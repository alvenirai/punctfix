from typing import Tuple

from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast


def get_english_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    model = AutoModelForTokenClassification.from_pretrained("Alvenir/bert-punct-restoration-en")
    tokenizer = BertTokenizerFast.from_pretrained("Alvenir/bert-punct-restoration-en")
    return model, tokenizer


def get_danish_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, BertTokenizerFast]:
    tokenizer = BertTokenizerFast.from_pretrained("Alvenir/bert-punct-restoration-da")
    model = AutoModelForTokenClassification.from_pretrained("Alvenir/bert-punct-restoration-da")
    return model, tokenizer


def get_german_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    tokenizer = BertTokenizerFast.from_pretrained("Alvenir/bert-punct-restoration-de")
    model = AutoModelForTokenClassification.from_pretrained("Alvenir/bert-punct-restoration-de")
    return model, tokenizer


def get_custom_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer
