# Lecture 2: Building Makemore

Lecture 2 introduces two character-level bigram language models: the "Bigram Language Model", which is a container for the conditional probabilities P(ch2 | ch1) for each (ch1, ch2) in the dataset, and the "Neural Net Model", which uses just one layer of weights to predict the next character from the previous, making it similar in design and outcome to the first model. These models and variations
are implemented in models.py. 

## Exercises

Exercises from video description. Observations may be replicated by running `python main.py`. 

### "E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?"

Trigram language models (both counting and neural net) decreased the loss (nll) of the dataset under the model relative to bigram models. 
* Counting:   bigram_nll=2.45 ---> trigram_nll=2.10
* Neural net: bigram_nll=2.46 ---> trigram_nll=2.26

### "E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?"

Results: 
* `Bigram LM  (smoothing=1):  train_nll=tensor(2.4546), dev_nll=tensor(2.4500), test_nll=tensor(2.4614)`
* `Trigram LM (smoothing=1): train_nll=tensor(2.0946), dev_nll=tensor(2.1270), test_nll=tensor(2.1352)`

For each model, the performance (nll of the split under the model) across train/dev/test splits is approximately equal. This is a nice example of a suboptimal tradeoff between bias and variance, 
where the models' simplicity causes high bias: the performance across the splits is constant because these models are quite simple and almost certainly underfit the training data. 


### "E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?"

Bigram and Trigram language models are so simple that on this dataset, they fail to demonstrate the typical interaction between regularization and train/test/dev loss. Since the loss across each split
is approximately equal, there's no evidence of overfitting. In fact, adding more regularization increases loss across all three splits, leading me to believe that the models have failed to fully 
capture the true distribution of this dataset. 

### "E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?"


The matrix multiplication in this dataset `(x @ W)` with one-hot `x` vectors looks something like this:

```
                [[w11 w21 w31 w41]
                 [w12 w22 w32 w42]
                 [w13 w23 w33 w43]    --> [w14 w24 w34 w44]
                 [w14 w24 w34 w44] 
[0 0 0 1 0]      [w15 w25 w35 w45]] 
```
Multiplying the rows of x with the columns of W in this example zeros out everything except the 4th row of W, which is equivalent to taking W[i] where i is the location of the 1 in the one-hot encoding.

Indexing is more straightforward but is actually slower than matrix multiplying one-hot x tensors on my macbook.


### "E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?"

f.cross_entropy() loss and nll loss can be equivalent, and the neural nets in this folder are a great example of this.

The `BigramNeuralNet` class computes nll loss with the following steps:
1. Run the input through the network to produce 'logits': `logits = xenc @ self.W`. We treat logits as the 'log-counts' of the output classes. 
2. Pull the counts out of log-space with `counts = logits.exp()`
3. Normalize the counts to produce probablities of each class: `probs = counts / counts.sum(dim=-1, keepdims=True)`
4. Multiply the generated probabilities with the true labels (done in this program by indexing, since labels are one-hot encoded) and take the log: `nll = -probs[example_indices, ys].log().mean()`

These steps are expressed mathematically as the definition of CrossEntropyLoss in PyTorch documentation. CrossEntropyLoss has the benefits of being more simpler (taking logits as input and doing
all the computation for you), more flexible (additional arguments), and faster since all of the steps are combined into one efficient implementation.

Time variance between training runs is too high to tell whether F.cross_entropy() is faster than my nll implementation without a high number of training runs. 
