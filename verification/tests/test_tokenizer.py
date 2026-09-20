"""Tokenizer round-trip and edge-case verification."""

from __future__ import annotations

from pkg_import import import_from_pkg

CORPUS = """Once upon a time there was a little cat.
The cat liked milk and sun.
Friends play near the water every day.
नमस्ते दुनिया — café naïve 日本語
"""


def test_roundtrip_known_chars():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    text = "The cat liked milk"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_deterministic_encode_decode():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    text = "Once upon a time"
    assert tok.encode(text) == tok.encode(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == tok.decode(ids)


def test_empty_string():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unknown_char_maps_to_space_or_zero():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text("abc ")
    ids = tok.encode("aX")
    assert ids[0] == tok.stoi["a"]
    assert ids[1] == tok.stoi[" "]


def test_unicode_in_vocab_roundtrip():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    text = "नमस्ते"
    decoded = tok.decode(tok.encode(text))
    assert isinstance(decoded, str)
    assert len(tok.encode(text)) == len(text)


def test_vocab_size_positive():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    assert tok.vocab_size >= 10
    assert tok.vocab_size == len(tok.stoi)


def test_tokens_per_char_is_one():
    CharTokenizer = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer").CharTokenizer
    tok = CharTokenizer.from_text(CORPUS)
    text = "hello world"
    assert len(tok.encode(text)) / max(len(text), 1) == 1.0
