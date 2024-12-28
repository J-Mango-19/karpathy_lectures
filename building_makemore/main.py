import torch
import random
import time
from models import BigramLanguageModel, TrigramLanguageModel, BigramNeuralNet, TrigramNeuralNet

# -------------------------------------------------------
# Setup
with open("data/names.txt", "r") as file:
    names = file.read().splitlines()

chars = sorted(list(set(''.join(names))))
n_chars = len(chars)

itos = {i+1:s for i, s in enumerate(chars)}
itos[0] = '.'
stoi = {s:i for i, s in itos.items()}

# -------------------------------------------------------
# Bigram (Conditional Probability) language model

print("----- Bigram Language Model -----\n")
blm = BigramLanguageModel(n_chars, itos, stoi)
blm.get_counts(names)
blm.compute_probs()
nll = blm.compute_nll(names)
print(f"NLL of training set after fitting: {nll.item():.4f}")
print("10 samples from the bigram langauge model: ")
for i in range(10):
    print("\t", blm.sample())

# -------------------------------------------------------
# Trigram (Conditional Probability) language model

print("\n----- Trigram Language Model -----\n")
tlm = TrigramLanguageModel(n_chars, itos, stoi)
tlm.get_counts(names)
tlm.compute_probs()
nll = tlm.compute_nll(names)
print(f"NLL of training set after fitting: {nll.item():.4f}")
print("10 samples from the trigram langauge model: ")
for i in range(10):
    print("\t", tlm.sample(blm))

# -------------------------------------------------------
# Bigram (Single-layer neural net) language model

print("\n----- Bigram Neural Net Language Model -----\n")
bnn = BigramNeuralNet(n_chars, itos, stoi)
xs, ys = bnn.create_dataset(names)
nll = bnn.train(xs, ys)
print(f"NLL of training set after training: {nll.item():.4f}")
print("10 samples from the bigram langauge model: ")
for i in range(10):
    print("\t", bnn.sample())

# -------------------------------------------------------
# Trigram (Single-layer neural net) language model

print("\n----- Trigram Neural Net Language Model -----\n")
tnn = TrigramNeuralNet(n_chars, itos, stoi)
xs, ys = tnn.create_dataset(names)
nll = tnn.train(xs, ys)
print(f"NLL of training set after training: {nll.item():.4f}")
print("10 samples from the bigram langauge model: ")
for i in range(10):
    print("\t", tnn.sample(blm))


# -------------------------------------------------------
# Split dataset into train/dev/test; evaluate train/dev/test loss
print("\n----- Train/dev/test model performance -----\n")

random.shuffle(names)
train_names = names[:int(0.8*len(names))]
dev_names = names[int(0.8*len(names)):int(0.9*len(names))]
test_names = names[int(0.9*len(names)):]

blm = BigramLanguageModel(n_chars, itos, stoi)
blm.get_counts(train_names, smoothing=1)
blm.compute_probs()
train_nll = blm.compute_nll(train_names); dev_nll = blm.compute_nll(dev_names); test_nll = blm.compute_nll(test_names)
print(f"Bigram LM (smoothing=1): {train_nll=}, {dev_nll=}, {test_nll=}")

tlm = TrigramLanguageModel(n_chars, itos, stoi)
tlm.get_counts(train_names, smoothing=1)
tlm.compute_probs()
train_nll = tlm.compute_nll(train_names); dev_nll = tlm.compute_nll(dev_names); test_nll = tlm.compute_nll(test_names)
print(f"Trigram LM (smoothing=1): {train_nll=}, {dev_nll=}, {test_nll=}")


# -------------------------------------------------------
# Use the dev set to tune the smoothing strength of the trigram model

print("\n----- Tuning performance with dev set -----\n")
print("Trigram LM - Loss for each split at different smoothing values: ")
min_dev_nll, min_dev_nll_smoothing_val = float('inf'), float('inf')
for smoothing_value in range(5):
    tlm = TrigramLanguageModel(n_chars, itos, stoi)
    tlm.get_counts(train_names, smoothing=smoothing_value)
    tlm.compute_probs()
    train_nll = tlm.compute_nll(train_names); dev_nll = tlm.compute_nll(dev_names); test_nll = tlm.compute_nll(test_names)
    print(f"Trigram LM (smoothing={smoothing_value}): {train_nll=}, {dev_nll=}, {test_nll=}")
    if dev_nll < min_dev_nll:
        min_dev_nll = dev_nll
        min_dev_nll_smoothing_val = smoothing_value
print(f"lowest dev loss was {min_dev_nll:.4f}, with smoothing={min_dev_nll_smoothing_val}")

print("Bigram LM - Loss for each split at different smoothing values: ")
min_dev_nll, min_dev_nll_smoothing_val = float('inf'), float('inf')
for smoothing_value in range(5):
    blm = BigramLanguageModel(n_chars, itos, stoi)
    blm.get_counts(train_names, smoothing=smoothing_value)
    blm.compute_probs()
    train_nll = blm.compute_nll(train_names); dev_nll = blm.compute_nll(dev_names); test_nll = blm.compute_nll(test_names)
    print(f"Bigram LM (smoothing={smoothing_value}): {train_nll=}, {dev_nll=}, {test_nll=}")
    if dev_nll < min_dev_nll:
        min_dev_nll = dev_nll
        min_dev_nll_smoothing_val = smoothing_value
print(f"lowest dev loss was {min_dev_nll:.4f}, with smoothing={min_dev_nll_smoothing_val}")


# -------------------------------------------------------
# Use the dev set to tune the smoothing strength of the trigram model

print("----- Bigram (single-layer neural net) LM: equivalence & efficiency of indexing over F.one_hot() -----\n")
bnn = BigramNeuralNet(n_chars, itos, stoi)
xs, ys = bnn.create_dataset(names)

bnn = BigramNeuralNet(n_chars, itos, stoi)
start_time = time.time()
nll = bnn.train(xs, ys)
end_time = time.time()
print(f"NLL of training set after training with F.one_hot(): {nll.item():.4f}, time: {(end_time - start_time):.4f}s")

# train with indexing W
start_time = time.time()
nll = bnn.train_by_indexing(xs, ys)
end_time = time.time()
print(f"NLL of training set after training with indexing: {nll.item():.4f}, time: {(end_time - start_time):.4f}s")
# train with matrix multiplying W with one-hot x tensors


# -------------------------------------------------------
# Demonstrate equivalence and speedup of F.cross_entropy() over computing nll loss manually
print("\n----- nll & F.cross_entropy(): equivalence and efficiency -----\n")

print("Training BigramNeuralNet with nll loss:")
bnn = BigramNeuralNet(n_chars, itos, stoi)
start_time = time.time()
nll = bnn.train(xs, ys)
end_time = time.time()
print(f"NLL of training set after training with nll loss: {nll.item():.4f}, time: {(end_time - start_time):.4f}s")

print("Training BigramNeuralNet with F.cross_entropy() loss:")
bnn = BigramNeuralNet(n_chars, itos, stoi)
start_time = time.time()
ce = bnn.train_by_CE_loss(xs, ys)
end_time = time.time()
print(f"Cross Entropy of training set after training with F.cross_entropy(): {ce.item():.4f}, time: {(end_time - start_time):.4f}s")
