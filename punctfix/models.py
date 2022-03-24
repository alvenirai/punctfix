from typing import Tuple

from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast


def get_english_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """
    Gets English transformer model and tokenizer
    :return: Tuple with (model, tokenizer)
    """
    model_id = "Alvenir/bert-punct-restoration-en"
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    return model, tokenizer


def get_danish_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, BertTokenizerFast]:
    """
    Gets Danish transformer model and tokenizer
    :return: Tuple with (model, tokenizer)
    """
    model_id = "Alvenir/bert-punct-restoration-da"
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return model, tokenizer


def get_german_model_and_tokenizer() -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """
    Gets German transformer model and tokenizer
    :return: Tuple with (model, tokenizer)
    """
    model_id = "Alvenir/bert-punct-restoration-de"
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return model, tokenizer


def get_custom_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """
    Gets local transformer model and tokenizer
    :return: Tuple with (model, tokenizer)
    """
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer
