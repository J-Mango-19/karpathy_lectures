import torch
import matplotlib.pyplot as plt


#class LanguageModel():
#    def __init__(self, data_dim, itos, stoi, special_char='.'):

class BigramLanguageModel():
    def __init__(self, data_dim, itos, stoi, special_char='.'):
        data_dim += 1 # to account for the special char
        self.itos = itos
        self.stoi = stoi
        self.special_char = special_char
        self.counts = torch.zeros((data_dim, data_dim), dtype=torch.float32)
        self.probs = None

    def get_counts(self, words_list):
        for w in words_list:
            chars = [self.special_char] + list(w) + [self.special_char]
            for ch1, ch2 in zip(chars, chars[1:]):
                # ch2 is always the next char wrt ch1
                ix1, ix2 = self.stoi[ch1], self.stoi[ch2]
                self.counts[ix1, ix2] += 1 # observed an instance of the pair tracked by counts[ix1, ix2]; increment count

    def compute_probs(self):
        self.probs = self.counts / self.counts.sum(1, keepdims=True)

    def next_char_distr(current_char_idx):
        return self.probs[current_char_idx]

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




