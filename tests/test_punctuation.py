import unittest

from punctfix import PunctFixer


class DanishPunctuationRestorationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.model = PunctFixer(language="da")

    def test_sample01(self):
        model_input = "mit navn det er rasmus og jeg kommer fra firmaet alvenir " \
                      "det er mig som har trænet denne lækre model"
        expected_output = "Mit navn det er Rasmus og jeg kommer fra firmaet Alvenir. " \
                          "Det er mig som har trænet denne lækre model."

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)

    def test_sample02(self):
        model_input = "en dag bliver vi sku glade for at vi nu kan sætte punktummer " \
                      "og kommaer i en sætning det fungerer da meget godt ikke"
        expected_output = "En dag bliver vi sku glade for, at vi nu kan sætte punktummer " \
                          "og kommaer i en sætning. Det fungerer da meget godt, ikke?"

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)


class EnglishPunctuationRestorationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.model = PunctFixer(language="en")

    def test_sample01(self):
        model_input = "hello i come from denmark and i am very good at english"
        expected_output = "Hello! I come from Denmark and I am very good at English."

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)

    def test_sample02(self):
        model_input = "do you really want to know this is so weird to write text just to see if it works does it work"
        expected_output = "Do you really want to know? This is so weird to write text just to see " \
                          "if it works? Does it work?"

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)


class GermanPunctuationRestorationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.model = PunctFixer(language="de")

    def test_sample01(self):
        model_input = "oscar geht einkaufen in einen großen supermarkt seine einkaufsliste ist lang er " \
                      "kauft für das ganze wochenende ein außerdem kommen gäste für die er kochen wird"
        expected_output = "Oscar geht einkaufen in einen großen Supermarkt. Seine Einkaufsliste ist lang. " \
                          "Er kauft für das ganze Wochenende ein. Außerdem kommen Gäste, für die er kochen wird."

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)
