# N-Gram Language Model

An n-gram model is a type of probabilistic language model for predicting the next item in such a sequence in the form of a (n − 1)–order Markov model.

## Add-K Smoothing
Add-1 smoothing (also called as Laplace smoothing) is a simple smoothing technique that Add 1 to the count of all n-grams in the training set before normalizing into probabilities. Add-k smoothing is an extension of Laplace smoothing that allows us to add a specified positive k value.

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

