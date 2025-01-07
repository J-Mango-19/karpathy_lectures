import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_LM(nn.Module):
    def __init__(self, emb_dim=2, ctx_len=3, hidden_dim=200, scale_params=False, direct_connections=False, num_classes=27):
        super().__init__()
        self.emb_dim = emb_dim
        self.ctx_len = ctx_len
        # I'll deviate from the lecture to use nn.Parameter rather than a list of parameters 
        # Torch will track these as model parameters, freeing me from having to keep a list

        # optionally scale parameters at initialization by 0.1
        scalar = 0.1 if scale_params is True else 1.0

        # embedding lookup table
        self.C = scalar * nn.Parameter(torch.randn((num_classes, emb_dim)))

        # MLP parameters
        self.W1 = scalar * nn.Parameter(torch.randn((emb_dim * ctx_len, hidden_dim)))
        self.b1 = scalar * nn.Parameter(torch.randn(hidden_dim))
        self.W2 = scalar * nn.Parameter(torch.randn((hidden_dim, num_classes)))
        self.b2 = scalar * nn.Parameter(torch.randn(num_classes))

        self.direct_connections = direct_connections
        if direct_connections is True:
            self.W3 = scalar * nn.Parameter(torch.randn((emb_dim * ctx_len, num_classes)))
            self.b3 = scalar * nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        # x is given as a tensor of indices
        # embed the indices
        emb = self.C[x]

        # squash the context embeddings into one vector for input to layer 1
        emb = emb.view(-1, self.ctx_len*self.emb_dim)

        # first layer
        h = torch.tanh(emb @ self.W1 + self.b1)

        # second layer
        logits = h @ self.W2 + self.b2
        print(logits.abs().mean())
        import sys; sys.exit()

        # optional direct connections
        if self.direct_connections is True:
            logits += emb @ self.W3 + self.b3

        return logits

    def sample(self, num_samples, itos):
        self.eval()
        with torch.no_grad():
            generated_words = []
            for i in range(num_samples):
                word_ch_idxs = []
                # create context consisting of special chars
                context = [0]*self.ctx_len

                new_ch_idx = None
                while new_ch_idx != 0:

                    logits = self.forward(torch.tensor(context))
                    probs = F.softmax(logits, dim=1)
                    new_ch_idx = torch.multinomial(probs, num_samples=1, replacement=True).item()
                    word_ch_idxs.append(new_ch_idx)

                    # maintain a fixed-length context window as chars are generated
                    context = context[1:] + [new_ch_idx]

                # compile the chars_idxs of each word into a string and append to words list
                word_ch_idxs = [itos[idx] for idx in word_ch_idxs]
                generated_words.append(''.join(word_ch_idxs))

            return generated_words
