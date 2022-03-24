from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict, List

import torch
from transformers import TokenClassificationPipeline

from punctfix.models import get_custom_model_and_tokenizer, get_english_model_and_tokenizer, \
    get_danish_model_and_tokenizer, get_german_model_and_tokenizer


@dataclass
class WordPrediction:
    word: str
    labels: List[str]

    @property
    def label(self):
        return Counter(self.labels).most_common(1)[0][0]


class PunctFixer:
    """
    PunctFixer used to punctuate a given text.
    """

    def __init__(self, language: str = "da",
                 custom_model_path: str = None,
                 word_overlap: int = 70,
                 word_chunk_size: int = 100,
                 device: str = "cuda"):
        """
        :param language: Valid options are "da", "de", "en", for Danish, German and English, respectively.
        :param custom_model_path: If you have a trained model yourself. If parsed, then language param will be ignored.
        :param word_overlap: How many words should overlap in case text is too long. Defaults to 70.
        :param word_chunk_size: How many words should a single pass consist of. Defaults to 100.
        :param device: "cpu" or "cuda" to indicate where to run inference. Defaults to "cuda".
        """

        self.word_overlap = word_overlap
        self.word_chunk_size = word_chunk_size

        self.supported_languages = {
            "de": "German",
            "da": "Danish",
            "en": "English"
        }

        if custom_model_path:
            self.model, self.tokenizer = get_custom_model_and_tokenizer(custom_model_path)
        elif language == "en":
            self.model, self.tokenizer = get_english_model_and_tokenizer()
        elif language == "da":
            self.model, self.tokenizer = get_danish_model_and_tokenizer()
        elif language == "de":
            self.model, self.tokenizer = get_german_model_and_tokenizer()

        self.model = self.model.eval()
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1

        self.pipe = TokenClassificationPipeline(model=self.model,
                                                tokenizer=self.tokenizer,
                                                aggregation_strategy="first",
                                                device=self.device)

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get a dict containing supported languages for PunctFixer.

        :return: dict containing support languages for PunctFixer
        """
        return self.supported_languages

    def punctuate(self, text: str) -> str:
        """
        Punctuates given text.

        :param text: A lowercase text with no punctuation.
        :return: A punctuated text.
        """
        words = text.split(" ")

        chunks = []
        if len(words) >= self.word_chunk_size:
            chunks = [words[i:i + self.word_chunk_size]
                      for i in
                      range(0, len(words), self.word_chunk_size - self.word_overlap)]
        else:
            chunks.append(words)

        word_prediction_list = [WordPrediction(word=word, labels=[]) for word in words]

        for i, chunk_text in enumerate(chunks):
            output = self.pipe(" ".join(chunk_text))
            word_counter = 0
            for entity in output:
                label = entity["entity_group"]
                text = entity["word"]
                words_in_text = text.split(" ")

                for j, word in enumerate(words_in_text):
                    current_index = i * self.word_chunk_size + word_counter - (i * self.word_overlap)
                    assert word_prediction_list[current_index].word == word, \
                        f"Something went wrong due to handling of a long text... " \
                        f"Tried matching the word: {word} with {word_prediction_list[current_index].word}"
                    word_prediction_list[current_index].labels.append(label)
                    word_counter += 1

        # Combine final text
        final_text = []
        auto_upper_next = False
        for word_pred in word_prediction_list:
            punctuated_text, auto_upper_next = self._combine_label_and_word(word_pred.label,
                                                                            word_pred.word,
                                                                            auto_upper_next)
            final_text.append(punctuated_text)

        return " ".join(final_text)

    @staticmethod
    def _combine_label_and_word(label: str, word: str, auto_uppercase: bool = False) -> Tuple[str, bool]:

        next_auto_uppercase = False
        if label[-1] == "U":
            word = word.capitalize()

        if label[0] != "O":
            word += label[0]

        if auto_uppercase:
            word = word.capitalize()

        if label[0] != "0" and label[0] in [".", "!", "?"]:
            next_auto_uppercase = True

        return word, next_auto_uppercase
