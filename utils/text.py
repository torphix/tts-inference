from config.symbols import symbol2id, pad
from typing import List
import re


def phones_to_token_ids(phones: List[str]) -> List[int]:
    token_ids = []
    for phone in phones:
        if _should_keep_symbol(phone):
            token_ids.append(symbol2id[phone])
        else:
            raise Exception(f"Symbol {phone} is not a valid phone ...")
    return token_ids


def _should_keep_symbol(s: str) -> bool:
    return s in symbol2id and s != pad


def strip_cont_whitespaces(text: str) -> str:
    new_text = ""
    last_whitespace = False
    for char in text:
        if char == " " and last_whitespace:
            continue
        new_text += char
        last_whitespace = char == " "
    return new_text

def remove_invalid_chars(text:str, invalid_chars:str=None) -> str:
    if invalid_chars is None:
        invalid_chars = '[@#+=-_\]'
    text = re.sub('\s{2,}', ' ', text) 
    text = re.sub('[-=]', '', text)
    return text