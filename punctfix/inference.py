from collections import Counter
from dataclasses import dataclass
from typing import Tuple, Dict, List, Union, Optional
import warnings
import re

import torch
from transformers import TokenClassificationPipeline

from punctfix.models import get_custom_model_and_tokenizer, get_english_model_and_tokenizer, \
    get_danish_model_and_tokenizer, get_german_model_and_tokenizer


WORD_NORMALIZATION_PATTERN = re.compile(r"[\W_]+")

class NoLanguageOrModelSelect(Exception):
    """
    Exception raised if you fail to specify either a language or custom model path.
    """

class NonNormalizedTextWarning(RuntimeWarning):
    """
    Warning given when the input text does not follow the normalization used
    during training.
    """

@dataclass
class WordPrediction:
    """
    Dataclass to hold word and labels for inference.
    """
    word: str
    labels: List[str]

    @property
    def label(self):
        """
        Label property. When called, at least one label should always be present.

        :return: A single model label as a str
        """
        return Counter(self.labels).most_common(1)[0][0]


class PunctFixer:
    """
    PunctFixer used to punctuate a given text.
    """

    def __init__(self, language: str = "da",
                 custom_model_path: str = None,
                 use_auth_token: Optional[Union[bool, str]] = None,
                 word_overlap: int = 70,
                 word_chunk_size: int = 100,
                 device: Union[str, torch.device] = torch.device("cpu"),
                 skip_normalization=False,
                 warn_on_normalization=False,
                 batch_size: int = 1
                 ):
        """
        :param language: Valid options are "da", "de", "en", for Danish, German and English, respectively.
        :param custom_model_path: If you have a trained model yourself. If parsed, then language param will be ignored.
        :param word_overlap: How many words should overlap in case text is too long. Defaults to 70.
        :param word_chunk_size: How many words should a single pass consist of. Defaults to 100.
        :param device: A torch.device on which to perform inference. The strings "cpu" or "cuda" can also be given.
        :param skip_normalization: Don't check input text and don't normalize it.
        :param warn_on_normalization: Warn the user if the input text was normalized.
        :param batch_size: Number of text chunks to pass through token classification pipeline.
        """

        self.word_overlap = word_overlap
        self.word_chunk_size = word_chunk_size
        self.skip_normalization = skip_normalization
        self.warn_on_normalization = warn_on_normalization
        self.batch_size = batch_size

        self.supported_languages = {
            "de": "German",
            "da": "Danish",
            "en": "English"
        }

        if custom_model_path:
            self.model, self.tokenizer = get_custom_model_and_tokenizer(custom_model_path, use_auth_token)
        elif language == "en":
            self.model, self.tokenizer = get_english_model_and_tokenizer()
        elif language == "da":
            self.model, self.tokenizer = get_danish_model_and_tokenizer()
        elif language == "de":
            self.model, self.tokenizer = get_german_model_and_tokenizer()
        else:
            raise NoLanguageOrModelSelect("You need to specify either language or custom_model_path "
                                          "when instantiating a PunctFixer.")

        self.tokenizer.decoder.cleanup = False
        self.model = self.model.eval()
        if isinstance(device, str): # Backwards compatability
            self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        else:
            self.device = device


        self.pipe = TokenClassificationPipeline(model=self.model,
                                                tokenizer=self.tokenizer,
                                                aggregation_strategy="first",
                                                device=self.device,
                                                ignore_labels=[])

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get a dict containing supported languages for PunctFixer.

        :return: dict containing support languages for PunctFixer
        """
        return self.supported_languages

    @staticmethod
    def init_word_prediction_list(words: List[str]) -> List[WordPrediction]:
        """
        Initialize a word prediction list i.e. a list containing WordPrediction object for each word.
        :param words: List of words

        :return: List of Word predictions
        """
        return [WordPrediction(word=word, labels=[]) for word in words]

    def populate_word_prediction_with_labels(self, chunks: List[List[str]], word_prediction_list: List[WordPrediction]):
        """
        Performs predictions on all chunks of text, and adds labels to the relevant word predictions.

        :param chunks: List of List of words
        :param word_prediction_list: A list containing word predictions i.e. word and labels.
        :return: Word predictions list with all label predictions for each word
        """
        outputs = self.pipe([" ".join(chunk_text) for chunk_text in chunks], batch_size=self.batch_size)
        for i, output in enumerate(outputs):
            word_counter = 0
            for entity in output:
                label = entity["entity_group"]
                text = entity["word"]
                words_in_text = text.split(" ")

                for word in words_in_text:
                    current_index = i * self.word_chunk_size + word_counter - (i * self.word_overlap)

                    # Sanity check
                    assert word_prediction_list[current_index].word == word, \
                        f"Something went wrong while matching word list ... " \
                        f"Tried matching the word: {word} with {word_prediction_list[current_index].word}"
                    word_prediction_list[current_index].labels.append(label)
                    word_counter += 1

        return word_prediction_list

    def combine_word_predictions_into_final_text(self, word_prediction_list: List[WordPrediction]):
        """
        Combines all predictions for each word into a final string by checking label (majority vote or if equal
        predictions, it chooses however Counter from itertools chooses top_n.

        :param word_prediction_list: List of word predictions
        :return: A final string with punctuation
        """
        final_text = []
        auto_upper_next = False
        for word_pred in word_prediction_list:
            punctuated_text, auto_upper_next = self._combine_label_and_word(word_pred.label,
                                                                            word_pred.word,
                                                                            auto_upper_next)
            final_text.append(punctuated_text)

        return " ".join(final_text)

    def split_words_into_chunks(self, words: List[str]) -> List[List[str]]:
        """
        Simple method to split a list of words into chunks of words with overlap.

        :param words: List of words to split into chunks
        :return: List of List of words consisting of the chunks
        """
        return [words[i:i + self.word_chunk_size]
                for i in
                range(0, len(words), self.word_chunk_size - self.word_overlap)]

    def punctuate(self, text: str) -> str:
        """
        Punctuates given text.

        :param text: A lowercase text with no punctuation.
            If it has punctuatation, it will be removed.
        :return: A punctuated text.
        """
        words = self.split_input_text(text)

        # If we have a long sequence of text (measured by words), we split it into chunks
        chunks = []
        if len(words) >= self.word_chunk_size:
            chunks = self.split_words_into_chunks(words)
        else:
            chunks.append(words)

        # We create a word prediction list and then combine the predictions to to final text
        word_prediction_list = self.init_word_prediction_list(words)
        word_prediction_list = self.populate_word_prediction_with_labels(chunks, word_prediction_list)
        return self.combine_word_predictions_into_final_text(word_prediction_list)

    def split_input_text(self, text: str) -> List[str]:
        """
        Splits given text into words using whitespace tokenization, also performing normalization
        :param text: A lowercase text with no punctuation (otherwise normalized)
        :return: A list of the words in that text, splitted and normalized.
        """
        words = text.split(" ")
        if self.skip_normalization:
            return words

        normalized_words = []
        to_warn = []
        for word in words:
            if not word:
                to_warn.append("Additional whitespace was removed.")
            norm_word = WORD_NORMALIZATION_PATTERN.sub("", word)
            if not word:
                continue
            if len(norm_word) < len(word):
                to_warn.append(r"Non-word (r'\W') characters were removed.")
            # We might have removed the entire word
            if not norm_word:
                continue
            if not norm_word.islower():
                norm_word = norm_word.lower()
                to_warn.append("Text was lowercased.")
            normalized_words.append(norm_word)

        # Warn once for each type of normalization
        if self.warn_on_normalization and to_warn:
            warnings.warn(
                "The input text was modified to follow model normalization: " +
                " ".join(sorted(set(to_warn))) +
                " To avoid seeing this, set suppress_normalization_warning=True. "\
                "To entirely circumvent normalization, set skip_normalization=True. ",
                NonNormalizedTextWarning)
        return normalized_words

    @staticmethod
    def _combine_label_and_word(label: str, word: str, auto_uppercase: bool = False) -> Tuple[str, bool]:
        """
        Combines label and word into a single string by looking at the label to see what should be changed.

        :param label: Punctuation label from the model
        :param word: Word to handle
        :param auto_uppercase: Whether automatically uppercase independent of label
        :return: Tuple with a str consisting of the word with relevant punctuation and
        whether to auto capitalize next word
        """
        next_auto_uppercase = False
        if label[-1] == "U":
            word = word.capitalize()

        if label[0] != "O":
            word += label[0]

        if auto_uppercase:
            word = word.capitalize()

        if label[0] in {".", "!", "?"}:
            next_auto_uppercase = True

        return word, next_auto_uppercase
