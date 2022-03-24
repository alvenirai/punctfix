import unittest
from unittest.mock import patch, MagicMock, ANY

from punctfix import PunctFixer


class DanishPunctuationRestorationTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = PunctFixer(language="da")

    def tearDown(self) -> None:
        super().tearDown()
        self.model = None

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

    def tearDown(self) -> None:
        super().tearDown()
        self.model = None

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

    def tearDown(self) -> None:
        super().tearDown()
        self.model = None

    def test_sample01(self):
        model_input = "oscar geht einkaufen in einen großen supermarkt seine einkaufsliste ist lang er " \
                      "kauft für das ganze wochenende ein außerdem kommen gäste für die er kochen wird"
        expected_output = "Oscar geht einkaufen in einen großen Supermarkt. Seine Einkaufsliste ist lang. " \
                          "Er kauft für das ganze Wochenende ein. Außerdem kommen Gäste, für die er kochen wird."

        actual_output = self.model.punctuate(model_input)

        self.assertEqual(actual_output, expected_output)


class HandlingOfLargeTextsTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = PunctFixer(language="da")
        self.long_text = "halløj derude og vel kom til en podcast med jacob og neviton du kan lige starte med at fortælle lidt om dig selv ja ja men det er så mig der jakob den ene halvdel af det her potkarsthold det er jo noget nyt for os det her det er mogens amatører det måske også lige onderestmidtet det var fald den første potkarstvidhverår så vi var i korte introducerer introns vase bare bare fortæller om det er jo næsten lige ja vi faldt næsten jævnaldrende vi bor i samme by ja vi bor i viborg vi bor og stort set helt års liv er det vær du det har jeg larlasesjeg ar altid sådan tilhørt vi går jeg kommer jo fra en lille landsby men jeg tror de kom til at høre mere om osten sener han jeg skal nok komme sådan lidt men gode personlige historier om nu vi jeg tog lidt mærkelig fyr som vi nu og vi har en hel del beatet er det må man sige en grund til vi egentlig startede mig på et kaste fordi at vi altid vi gør det dem på at snakke om alle mulige ting i flere timer faktisk så tog jeg mig selv lige tænk på tænk tanken om at staten podcast det er jo lidt maiestremehvertfald efterhånden her det er så en sjov det er sjovt når at se til bag på så tænkte jeg hvorfor ikke og det er faktisk hende det første ting vi kan huske vi latham jeg var så ikke en kot karst men men jeg skulle hente nyredtanengang efter en bitur og jeg havde ikke sådan jeg havde kendt ham en nu er det måske er og de ting vi snakker om det var ikke bare sådan en af de to ting det var virkelig det var lige for tegnefilmen til livet altså være mening med livet så er der er ikke noget bestemt i komme til at høre de klokken tre om natten efter sådan bytur koge rundt i en bil det er bare gode tider og penge til på den nu ja ser for i det korona nu har jeg så sammen med det lidt anvende ap vi sidst nogen der hører med her i to tusinde og niogtyve så kun leve en palmi der er så lidt til historie børn en ny anne de vi bare snart med denne uge men vi kan jo også komme op med et navn til podcasten ja det er og sådan for titemitersi jamen vi har venner kan læge for kort og godt for at sige i kor et godt ja ser han det års ugerne jamen jeg ved silende med at sige nej det var i hvert fald bare lige sådan en kort introduktion jeg har tænkt på hvad vi skal snakke om næste gang og det kan de vi lave i en ilespoyler ja til folket i gik jeg ned at for altså jakob et meget formelt når jeg trækker med amrepoasnakerom mer så så meget åben odsngetsonogeernår han snakker om at så gar krimeligterdetså men nå hvor sprængt om ja hvordan det ændrer så altså hvordan vi var kun så hvordan børn er nu han når der er jo sket ikke mere og så hvis der sket nogle sjove ting og brøndum som jeg lige kan huske sjove fortællinger jeg vil jo i stå jeg er lige sket som vil det tænker en bar og så kommer vi jo hundrede procent af indre så kom nok rundt omkring hvad der lige sker så det hvert fald det ja jeg har hvert fad ikke med at sige jamen skal vi afslutte den for i dag så skal du så sige os jeg ved ikke hvordan migarmename det bliver i hvert fald bedre bedre betidehkavisde sige vor velkendende ord gør når vi alle det er ikke så kendt fordi det er første gang men det bliver det for en på en tidspunkt færdig"

    def tearDown(self) -> None:
        super().tearDown()
        self.model = None
        self.long_text = None

    def test_multiple_chunk_size_and_padding_configs(self):
        configs_to_test = [
            (50, 5),
            (50, 10),
            (50, 20),
            (50, 40),
            (100, 5),
            (100, 20),
            (100, 40),
            (100, 80),
            (150, 10),
            (150, 20),
            (150, 50),
            (150, 80),
            (150, 120),
            (200, 50),
            (200, 100),
            (200, 150),
            (200, 180)
        ]

        for chunk_size, overlap in configs_to_test:
            self.model.word_overlap = overlap
            self.model.word_chunk_size = chunk_size

            actual_output = self.model.punctuate(self.long_text)
            self.assertIsNotNone(actual_output)


class GenerelFunctionalityTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        # Setup
        self.torch_cuda_patch = patch(
            'punctfix.inference.torch.cuda.is_available'
        )
        self.torch_cuda_mock: MagicMock = self.torch_cuda_patch.start()
        self.torch_cuda_mock.return_value = False

        self.token_classification_pipeline_patch = patch(
            'punctfix.inference.TokenClassificationPipeline'
        )
        self.token_classification_pipeline_mock: MagicMock = self.token_classification_pipeline_patch.start()

    def test_if_gpu_not_available_default_cpu(self):
        # When
        self.model = PunctFixer(language="da", device="cuda")

        # Expect
        self.token_classification_pipeline_mock.assert_called_once_with(model=ANY,
                                                                        tokenizer=ANY,
                                                                        aggregation_strategy="first",
                                                                        device=-1)

    def tearDown(self) -> None:
        super().tearDown()
        self.torch_cuda_patch.stop()
        self.token_classification_pipeline_patch.stop()


if __name__ == '__main__':
    unittest.main()
