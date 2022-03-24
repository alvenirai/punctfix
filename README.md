# Punctuation restoration 
Adds punctuation and capitalization for a given text.

Works on Danish, German and English. 

Models hosted on huggingface! <3 

## Installation
```
pip install punctfix
```

## Usage
Its quite simple to use! 

```python
>>> from punctfix import PunctFixer
>>> model = PunctFixer(language="da")

>>> example_text = "mit navn det er rasmus og jeg kommer fra firmaet alvenir det er mig som har trænet denne lækre model"
>>> print(model.punctuate(example_text))
'Mit navn det er Rasmus og jeg kommer fra firmaet Alvenir. Det er mig som har trænet denne lækre model.'

>>> example_text = "en dag bliver vi sku glade for, at vi nu kan sætte punktummer og kommaer i en sætning det fungerer da meget godt ikke"
>>> print(fixer.punctuate(example_text)) 
'En dag bliver vi sku glade for, at vi nu kan sætte punktummer og kommaer i en sætning. Det fungerer da meget godt, ikke?' 
```


## Parameters for PunctFixer
* Pass `device="cuda"` or `device="cpu"` to indicate where to run inference. Default is `device="cpu"`
* To handle long sequences, we use a chunk size and an overlap. These can be modified. For higher speed but 
lower acuracy use a chunk size of 150-200 and very little overlap i.e. 5-10. These parameters are set with 
default values `word_chunk_size=100`, `word_overlap=70` which makes it run a bit slow. The default parameters
will be updated when we have some results on variations. 
* Supported languages are "en" for English, "da" for Danish and "de" for German. Default is `language="da"`.


## Contribute
If you encounter issues, feel free to open issues in the repo and then we will fix. Even better, create issue and 
then a PR that fixes the issue! ;-) 

Happy punctuating!
