# Punctuation restoration 
Adds punctuation and capitalization for a given text without punctuation.

Works on Danish, German and English. 

Models hosted on huggingface! ❤️  🤗

## Status with python 3.8
![example workflow](https://github.com/danspeech/punctfix/actions/workflows/run_tests.yml/badge.svg)
![example workflow](https://github.com/danspeech/punctfix/actions/workflows/pylint.yml/badge.svg)

## Installation
```
pip install punctfix
```

## Usage
Its quite simple to use! 

```python
>>> from punctfix import PunctFixer
>>> fixer = PunctFixer(language="da")

>>> example_text = "mit navn det er rasmus og jeg kommer fra firmaet alvenir det er mig som har trænet denne lækre model"
>>> print(fixer.punctuate(example_text))
'Mit navn det er Rasmus og jeg kommer fra firmaet Alvenir. Det er mig som har trænet denne lækre model.'

>>> example_text = "en dag bliver vi sku glade for, at vi nu kan sætte punktummer og kommaer i en sætning det fungerer da meget godt ikke"
>>> print(fixer.punctuate(example_text)) 
'En dag bliver vi sku glade for, at vi nu kan sætte punktummer og kommaer i en sætning. Det fungerer da meget godt, ikke?' 
```

Note that, per default, the input text will be normalied. See next section for more details.

## Parameters for PunctFixer
* Pass `device="cuda"` or `device="cpu"` to indicate where to run inference. Default is `device="cpu"`
* To handle long sequences, we use a chunk size and an overlap. These can be modified. For higher speed but 
lower acuracy use a chunk size of 150-200 and very little overlap i.e. 5-10. These parameters are set with 
default values `word_chunk_size=100`, `word_overlap=70` which makes it run a bit slow. The default parameters
will be updated when we have some results on variations. 
* Supported languages are "en" for English, "da" for Danish and "de" for German. Default is `language="da"`.
* Note that the fixer has been trained on normalized text (lowercase letters and numbers) and will per default normalize input text. You can instantiate the model with `skip_normalization=True` to disable this but this might yield errors on some input text.
* To raise warnings every time the input is normalied, set `warn_on_normalization=True`.

## Contribute
If you encounter issues, feel free to open issues in the repo and then we will fix. Even better, create issue and 
then a PR that fixes the issue! ;-)

Happy punctuating!
