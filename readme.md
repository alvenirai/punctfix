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
