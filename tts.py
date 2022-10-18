import torch
import inflect
import numpy as np
from g2p_en import G2p
from typing import Dict, List, Tuple
from torch.jit._script import ScriptModule
from utils.text import remove_invalid_chars
from utils.cleaners import (
    split_context_marker, 
    is_phone_marker,
    split_phone_marker, 
    strip_invalid_symbols)
from utils.text_norm import (
    remove_cont_whitespaces,
    EnglishTextNormalizer,
)
from utils.tokenization import (
    WordTokenizer,
    SentenceTokenizer,
    BertTokenizer,
)
from tqdm import tqdm
import soundfile as sf
from config.symbols import symbol2id
from models.uvinet import Generator
from g2p.dp.phonemizer import Phonemizer
from torch.jit._serialization import load
from config.configs import PreprocessingConfig, VocoderModelConfig



class TTSEngine:
    def __init__(self, model_name, speaker_id=0, use_finetuned_vocoder=True, talking_speed=1.0):
        self.speaker_id = speaker_id
        self.talking_speed = talking_speed
        self.lexicon = self.open_lexicon()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Models
        self.word_tokenizer = WordTokenizer(lang="en", remove_punct=False)
        self.sentence_tokenizer = SentenceTokenizer(lang="en")        
        self.g2p = Phonemizer.from_checkpoint('assets/torchscript/g2p.pt', lang_phoneme_dict=symbol2id)
        self.acoustic_model = load(f'assets/torchscript/{model_name}/acoustic_model.pt').eval().to(self.device)
        if use_finetuned_vocoder:
            self.vocoder = load(f'assets/torchscript/{model_name}/vocoder.pt').eval()
        else:
            self.vocoder = Generator(VocoderModelConfig(), PreprocessingConfig()).eval()
            self.vocoder.load_state_dict(torch.load('assets/torchscript/vocoder_org.pt')['generator'])
        self.style_predictor = load(f'assets/torchscript/{model_name}/style_predictor.pt').eval().to(self.device)
        self.text_normalizer = EnglishTextNormalizer()
        self.bert_tokenizer = BertTokenizer('assets')
    
    @property
    def chars_blacklist(self):
        return ['-','+','=','_','[',']','/','(',')']
    
    def open_lexicon(self, path='assets/lexicon.txt'):
        with open('assets/lexicon.txt', 'r') as f:
            lexicon = {line.split(" ")[0]:line.strip("\n").split(" ")[1:] for line in f.readlines()}
        return lexicon

    def preprocess_text(self, text):
        text = text.strip()
        text = remove_cont_whitespaces(text)
        text = remove_invalid_chars(text)
        text = text.split(" ")
        p = inflect.engine()
        p.number_to_words(99)
        for i, word in enumerate(text):
            if word.isnumeric():
                text[i] = p.number_to_words(word)
        return " ".join(text)

    def synthesize(self, text):
        text = self.preprocess_text(text)
        waves = []
        for sentence in tqdm(self.sentence_tokenizer.tokenize(text)):
            style_sentences = []
            symbol_ids = []
            for subsentence, context in split_context_marker(sentence):
                for subsubsentence in split_phone_marker(subsentence):
                    if is_phone_marker(subsubsentence):
                        for phone in subsubsentence.strip().split(" "):
                            if f"{phone}" in symbol2id:
                                symbol_ids.append(symbol2id[f"{phone}"])
                            style_sentences.append(context)
                    else:
                        subsubsentence = self.text_normalizer(subsubsentence)
                        style_sentences.append(self.text_normalizer(context))
                        for word in self.word_tokenizer.tokenize(subsubsentence):
                            word = word.lower()
                            if word.strip() == "":
                                continue
                            elif word in self.chars_blacklist:
                                continue
                            elif word in [".", "?", "!"]:
                                symbol_ids.append(symbol2id[word])
                            elif word in [",", ";"]:
                                symbol_ids.append(symbol2id["SILENCE"])
                            elif word in self.lexicon:
                                for phone in self.lexicon[word]:
                                    symbol_ids.append(symbol2id[phone])
                                symbol_ids.append(symbol2id["BLANK"])
                            else:
                                try:
                                    for phone in self.g2p(word):
                                        symbol_ids.append(symbol2id[phone])
                                    symbol_ids.append(symbol2id["BLANK"])
                                except:
                                    continue

            sentence_style = " ".join(style_sentences)
            encoding = self.bert_tokenizer([sentence_style])
            symbol_ids = torch.LongTensor([symbol_ids])
            speaker_ids = torch.LongTensor([self.speaker_id])

            with torch.no_grad():
                try:
                    style_embeds = self.style_predictor(
                        encoding["input_ids"].to(self.device),
                        encoding["attention_mask"].to(self.device),
                    )
                    mel = self.acoustic_model(
                        symbol_ids.to(self.device),
                        speaker_ids.to(self.device),
                        style_embeds.to(self.device),
                        1.0,
                        self.talking_speed,
                    )
                    wave = self.vocoder(mel.to('cpu'))
                    waves.append(wave.cpu().view(-1))
                except Exception as e:
                    print(e)

        wave_cat = torch.cat(waves).numpy()
        return wave_cat, 22050


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True)
    parser.add_argument('-si', '--speaker_id', default=0)
    parser.add_argument('-ts', '--talking_speed', default=1.0)
    parser.add_argument('-ftc', '--use_finetuned_vocoder', default=False)
    parser.add_argument('-dp', '--doc_path', type=str)
    parser.add_argument('-t', '--text', type=str)
    args, lf_args = parser.parse_known_args()

    tts = TTSEngine(args.model_name, 
                    int(args.speaker_id), 
                    args.use_finetuned_vocoder, 
                    args.talking_speed)
    if args.text is not None:
        audio, sr = tts.synthesize(args.text)
    if args.doc_path is not None:
        with open(args.doc_path, 'r') as f:
            text = f.read()
        audio, sr = tts.synthesize(text)
    sf.write("audio.wav", audio, sr)

