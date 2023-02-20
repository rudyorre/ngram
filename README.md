# N-Gram Language Model

An n-gram model is a type of probabilistic language model for predicting the next item in such a sequence in the form of a (n − 1)–order Markov model.

## Add-K Smoothing
Add-1 smoothing (also called as Laplace smoothing) is a simple smoothing technique that Add 1 to the count of all n-grams in the training set before normalizing into probabilities. Add-k smoothing is an extension of Laplace smoothing that allows us to add a specified positive k value.

## Sentence Generation (Greedy Search)
Using `greedy_search()`, we can generate the most probable sentence by doing greedy search on the model. When selecting the next word, the model will choose the word that maximizes the n-gram frequency given the previously added words. Depending on `n`, this can lead to varying different sentences, for example using the including training data:
```
n = 2:
"the company said"

n = 5:
"the company said the sale is part of a major program to divest several of its businesses representing about 200 mln dlrs in the first quarter of 1987"

n = 10:
"the company said the sale is subject to review by local authorities"
```

## Usage
Full usage of `language_model.py` which includes driver code for instantiating the n-gram `LanguageModel`:
```
usage: language_model.py [-h] --N N [--k K] [--verbose VERBOSE]

options:
  -h, --help         show this help message and exit
  --N N              Order of N-gram model to create (i.e. 1 for unigram, 2 for
                     bigram, etc.)
  --k K              Parameter for add-k smoothing (default is 0.0 (disabled) --
                     use 1 for add-1 smoothing)
  --verbose VERBOSE  Will print information that is helpful for debug if set to
                     True
```

Example running command:
```
python language_model.py --N 2 --k 0.1
```

Here, the arguments N is the order of the language model and k is the parameter for smoothing.


Unit test script:
```
python test.py 
```

If successful, you will see outputs like
```
....
----------------------------------------------------------------------
Ran 4 tests in 23.027s

OK
```

