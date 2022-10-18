import re
from typing import List, Iterable, Tuple

def strip_invalid_symbols(text: str, pad_symbol: str, valid_symbols: List[str]) -> str:
    new_token = ""
    for char in text.lower():
        if char in valid_symbols and char != pad_symbol:
            new_token += char
    return new_token


def split_phone_marker(text: str) -> List[str]:
    splits = re.finditer(r"{{(([^}]+}?)*)}}", text)
    out = []
    start = 0
    for split in splits:
        non_arpa = text[start : split.start()]
        arpa = text[split.start() + 2 : split.end() - 2]
        if len(non_arpa) > 0:
            out.append(non_arpa)
        out.append(arpa)
        start = split.end()
    if start < len(text):
        out.append(text[start:])
    return out


def split_context_marker(text: str) -> Iterable[Tuple[str, str]]:
    splits = re.finditer(r"\[\[(([^\]]+\|\|[^\]]+)*)\]\]", text)
    sentences, contexts = [], []
    start = 0
    for split in splits:
        non_context = text[start : split.start()]
        matched_part = text[split.start() + 2 : split.end() - 2].lstrip()
        if len(non_context) > 0:
            sentences.append(non_context)
            contexts.append(non_context)
        s = matched_part.split("||")
        sentence, context = s[0], s[1]
        sentences.append(sentence)
        contexts.append(context)
        start = split.end()
    if start < len(text):
        sentences.append(text[start:])
        contexts.append(text[start:])
    return zip(sentences, contexts)


def is_phone_marker(text: str) -> bool:
    if len(text) < 5:
        return False
    return text[:2] == "{{" and text[-2:] == "}}"


def is_context_marker(text: str) -> bool:
    if len(text) < 7:
        return False
    return text[:2] == "[[" and text[-2:] == "]]" and "||" in text[2:-2]


def is_phone(text: str) -> bool:
    return len(text) > 1 and text[0] == "@"