import re


def parse_decklist(text: str) -> list[str]:
    lines = text.strip().split("\n")
    return [re.sub(r"^\\d+\\s*x?\\s*", "", line).strip() for line in lines if line.strip()]
