from typing import Tuple, Dict

from transformers import TokenClassificationPipeline

from punctfix.models import get_custom_model_and_tokenizer, get_english_model_and_tokenizer, \
    get_danish_model_and_tokenizer, get_german_model_and_tokenizer


class PunctFixer:
    """
    PunctFixer used to punctuate a given text.
    """
    def __init__(self, language: str = "da",
                 custom_model_path: str = None,
                 device: str = "cuda"):
        """
        :param language: Valid options are "da", "de", "en", for Danish, German and English, respectively.
        :param custom_model_path: If you have a trained model yourself. If parsed, then language param will be ignored.
        :param device: "cpu" or "cuda" to indicate where to run inference.
        """

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
        self.device = -1 if device == "cpu" else 0

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
        output = self.pipe(text)
        final_text = []
        auto_upper_next = False
        for entity in output:
            label = entity["entity_group"]
            text = entity["word"]
            punctuated_text, auto_upper_next = self._combine_label_and_text(label, text, auto_upper_next)
            final_text.append(punctuated_text)

        return " ".join(final_text)

    @staticmethod
    def _combine_label_and_text(label: str, text: str, auto_uppercase: bool = False) -> Tuple[str, bool]:
        words = text.split(" ")

        next_auto_uppercase = False

        new_words = []
        for word in words:
            if label[-1] == "U":
                punct_wrd = word.capitalize()
            else:
                punct_wrd = word

            if label[0] != "O":
                punct_wrd += label[0]

            new_words.append(punct_wrd)

        if auto_uppercase and new_words[0]:
            new_words[0] = new_words[0].capitalize()

        if label[0] != "0" and label[0] in [".", "!", "?"]:
            next_auto_uppercase = True

        return " ".join(new_words), next_auto_uppercase
