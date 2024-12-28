import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TrigramLanguageModel():
    def __init__(self, data_dim, itos, stoi, special_char='.'):
        data_dim += 1# to account for the special char
        self.data_dim = data_dim
        self.itos = itos
        self.stoi = stoi
        self.special_char = special_char
        self.counts = torch.zeros((data_dim, data_dim, data_dim), dtype=torch.float32)
        self.probs = None

    def get_counts(self, words_list, smoothing=1):
        self.counts += smoothing
        for w in words_list:
            chars = [self.special_char] + list(w) + [self.special_char]
            for ch1, ch2, ch3 in zip(chars, chars[1:], chars[2:]):
                # increment the count of characters (ch1, ch2, ch3) found in that order
                ix1, ix2, ix3 = self.stoi[ch1], self.stoi[ch2], self.stoi[ch3]
                self.counts[ix1, ix2, ix3] += 1

    def compute_probs(self):
        self.probs = self.counts / self.counts.sum(2, keepdims=True)

    def sample(self, blm):
        first_char = self.special_char

        #second_char = self.itos[torch.multinomial(torch.ones(self.data_dim) / self.data_dim, num_samples=1, replacement=True).item()]
        second_char = self.itos[torch.multinomial(blm.probs[0], num_samples=1, replacement=True).item()]
        ix1, ix2 = self.stoi[first_char], self.stoi[second_char]
        output = []
        while True:
            ix3 = torch.multinomial(self.probs[ix1, ix2], num_samples=1, replacement=True).item()
            output.append(self.itos[ix3])
            if self.itos[ix3] == self.special_char:
                break
            ix1 = ix2; ix2 = ix3
        return ''.join(output)

    def compute_nll(self, words_list):
        log_likelihood = 0
        n = 0
        for w in words_list:
            w = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(w, w[1:], w[2:]):
                ix1, ix2, ix3 = self.stoi[ch1], self.stoi[ch2], self.stoi[ch3]
                log_prob = self.probs[ix1, ix2, ix3].log()
                log_likelihood += log_prob
                n += 1
        nll = -log_likelihood / n
        return nll


class BigramLanguageModel:
    def __init__(self, data_dim, itos, stoi, special_char='.'):
        data_dim += 1 # to account for the special char
        self.data_dim = data_dim
        self.itos = itos
        self.stoi = stoi
        self.special_char = special_char
        self.counts = torch.zeros((data_dim, data_dim), dtype=torch.float32)
        self.probs = None


    def get_counts(self, words_list, smoothing=1):
        self.counts += smoothing
        for w in words_list:
            chars = [self.special_char] + list(w) + [self.special_char]
            for ch1, ch2 in zip(chars, chars[1:]):
                # ch2 is always the next char wrt ch1
                ix1, ix2 = self.stoi[ch1], self.stoi[ch2]
                self.counts[ix1, ix2] += 1 # observed an instance of the pair tracked by counts[ix1, ix2]; increment count

    def compute_probs(self):
        self.probs = self.counts / self.counts.sum(1, keepdims=True)

    def sample(self):
        first_char = self.special_char
        ix = self.stoi[first_char]
        output = []
        while True:
            ix = torch.multinomial(self.probs[ix], num_samples=1, replacement=True).item()
            output.append(self.itos[ix])
            if self.itos[ix] == self.special_char:
                break
        return ''.join(output)

    def compute_nll(self, words_list):
        log_likelihood = 0
        n = 0
        for w in words_list:
            w = ['.'] + list(w) + ['.']
            for ch1, ch2, in zip(w, w[1:]):
                ix1, ix2 = self.stoi[ch1], self.stoi[ch2]
                log_prob = self.probs[ix1, ix2].log()
                log_likelihood += log_prob
                n += 1
        nll = -log_likelihood / n
        return nll

    def plot(self, filename="BigramLanguageModel.png"):
        plt.figure(figsize=(16,16))
        plt.imshow(self.counts, cmap="Blues")

        for i in range(self.data_dim):
            for j in range(self.data_dim):
                char_str = self.itos[i] + self.itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
                plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
        plt.axis("off")
        print(f"Saving plot under assets/{filename}")

class BigramNeuralNet:
    def __init__(self, data_dim, itos, stoi, special_char='.'):
        data_dim += 1 # for special character
        self.data_dim = data_dim
        self.itos = itos
        self.stoi = stoi
        self.special_char = special_char
        self.W = torch.randn((data_dim, data_dim), requires_grad=True)

    def create_dataset(self, words_list):
        xs, ys = [], []
        for w in words_list:
            w = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(w, w[1:]):
                ix1, ix2 = self.stoi[ch1], self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        return xs, ys

    def forward(self, xenc):
        logits = xenc @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(dim=-1, keepdims=True)
        return probs

    def train(self, xs, ys, num_iterations=250, lr=70):
        n = len(xs)
        xenc = F.one_hot(xs, num_classes=self.data_dim).float()
        example_indices = torch.arange(n)
        print("training...")
        for k in range(num_iterations):
            # forward pass
            probs = self.forward(xenc)

            # loss calc
            log_likelihood = probs[example_indices, ys].log().mean()
            loss = -log_likelihood

            # backward pass
            self.W.grad = None
            loss.backward()

            # update
            self.W.data += -lr * self.W.grad

        return loss

    def train_by_indexing(self, xs, ys, num_iterations=250, lr=70):
        # see exercise 4 in README
        n = len(xs)
        example_indices = torch.arange(n)
        print("training...")
        for k in range(num_iterations):
            # forward pass
            logits = self.W[xs]
            counts = logits.exp()
            probs = counts / counts.sum(dim=-1, keepdims=True)

            # loss calc
            log_likelihood = probs[example_indices, ys].log().mean()
            loss = -log_likelihood

            # backward pass
            self.W.grad = None
            loss.backward()

            # update
            self.W.data += -lr * self.W.grad

        return loss

    def train_by_CE_loss(self, xs, ys, num_iterations=250, lr=70):
        n = len(xs)
        xenc = F.one_hot(xs, num_classes=self.data_dim).float()
        example_indices = torch.arange(n)
        print("training...")
        for k in range(num_iterations):
            # forward pass
            logits = xenc @ self.W

            # loss calc
            loss = F.cross_entropy(logits, ys)

            # backward pass
            self.W.grad = None
            loss.backward()

            # update
            self.W.data += -lr * self.W.grad

        return loss



    def sample(self):
        first_char = self.special_char
        xenc = F.one_hot(torch.tensor(self.stoi[first_char]), num_classes=self.data_dim).float()
        output = []
        while True:
            probs = self.forward(xenc)
            ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
            output.append(self.itos[ix])
            if self.itos[ix] == self.special_char:
                break
            xenc = F.one_hot(torch.tensor(ix), num_classes=self.data_dim).float()
        return ''.join(output)


class TrigramNeuralNet:
    def __init__(self, data_dim, itos, stoi, special_char='.'):
        data_dim += 1 # for special character
        self.data_dim = data_dim
        self.itos = itos
        self.stoi = stoi
        self.special_char = special_char
        self.W = torch.randn((2*data_dim, data_dim), requires_grad=True)

    def create_dataset(self, words_list):
        xs, ys = [], []
        for w in words_list:
            w = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(w, w[1:], w[2:]):
                ix1, ix2, ix3 = self.stoi[ch1], self.stoi[ch2], self.stoi[ch3]
                xs.append((ix1, ix2))
                ys.append(ix3)
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        return xs, ys

    def forward(self, xenc):
        logits = xenc @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(dim=-1, keepdims=True)
        return probs

    def train(self, xs, ys, num_iterations=250, lr=25):
        n = len(xs)
        xenc = F.one_hot(xs, num_classes=self.data_dim).flatten(start_dim=-2).float()
        example_indices = torch.arange(n)
        print("training...")
        for k in range(num_iterations):
            # forward pass
            probs = self.forward(xenc)

            # loss calc
            log_likelihood = probs[example_indices, ys].log().mean()
            loss = -log_likelihood

            # backward pass
            self.W.grad = None
            loss.backward()

            # update
            self.W.data += -lr * self.W.grad

        return loss

    def sample(self, blm):
        ix1 = self.stoi[self.special_char]
        ix2 = torch.multinomial(blm.probs[0], num_samples=1, replacement=True).item()
        xenc1 = F.one_hot(torch.tensor(ix1), num_classes=self.data_dim).float()
        xenc2 = F.one_hot(torch.tensor(ix2), num_classes=self.data_dim).float()
        xenc = torch.concat((xenc1, xenc2), dim=-1)

        output = []
        while True:
            probs = self.forward(xenc)
            ix3 = torch.multinomial(probs, num_samples=1, replacement=True).item()
            output.append(self.itos[ix3])
            if self.itos[ix3] == self.special_char:
                break

            ix1 = ix2
            ix2 = ix3
            xenc1 = F.one_hot(torch.tensor(ix1), num_classes=self.data_dim).float()
            xenc2 = F.one_hot(torch.tensor(ix2), num_classes=self.data_dim).float()
            xenc = torch.concat((xenc1, xenc2), dim=-1)
        return ''.join(output)
