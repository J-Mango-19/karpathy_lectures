import torch
import random
import argparse
import matplotlib.pyplot as plt

def make_dataset(words, context_len, train_split, dev_split, test_split, special_char='.'):
    print("Creating dataset...")
    assert train_split + dev_split + test_split == 1
    random.shuffle(words)

    # chars is the model's vocabulary
    # itos and stoi are mappings between characters and their integer forms
    chars = sorted(list(set(''.join(words))))
    num_classes = len(chars) + 1
    itos = {i+1 : s for i, s in enumerate(chars)}
    itos[0] = special_char
    stoi = {s : i for i, s in itos.items()}

    # create character-level dataset
    x, y = [], []
    for w in words:
        # initialize context to the special character (index 0)
        context = [0]*context_len

        # ensure every word ends with the special char
        for ch in w + special_char:
            y.append(stoi[ch])
            x.append(context)

            # before every new character, context array should:
            # 1) forget the least recent character
            # 2) incorporate the newest character
            context = context[1:] + [stoi[ch]]

    x = torch.tensor(x)
    y = torch.tensor(y)

    # print the number format of the dataset
    print(f'Dataset created: {y.shape[0]} examples of {context_len} characters')

    # ensure every x has a y
    assert x.shape[0] == y.shape[0]

    # partition dataset into splits
    n1 = int(train_split * x.shape[0])
    n2 = int((train_split + dev_split) * x.shape[0])

    x_train, y_train = x[:n1], y[:n1]
    x_dev, y_dev     = x[n1:n2], y[n1:n2]
    x_test, y_test   = x[n2:], y[n2:]

    return x_train, y_train, x_dev, y_dev, x_test, y_test, itos, stoi, num_classes

def evaluate(model, x, y):
    # return the nll of the given data under the model
    model.eval()
    with torch.no_grad():
        # manual implementation of nll for practice
        logits = model(x)

        # account for numerical overflow
        logits -= logits.max()

        # softmax
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdims=True)

        # calculate log-likelihood by indexing true value probabilities
        log_likelihood = probs[torch.arange(y.shape[0]), y].log().mean()
        return -log_likelihood.item()

def plot_embs(model, itos, filename='2dembeddings.png'):
    """
    For learned 2-dimensional character embeddings, plot them and their corresponding chars
    Gives an idea of the association the model learns between characters, eg vowels are grouped
    """
    C = model.C
    assert C.shape[1] == 2
    plt.figure(figsize=(8,8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
        plt.grid('minor')
    plt.savefig(filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctx_len',        type=float, default=3)
    parser.add_argument('--emb_dim',        type=int,   default=15)
    parser.add_argument('--num_samples',    type=int,   default=20)
    parser.add_argument('--hidden_dim',     type=int,   default=300)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--num_train_steps',type=int,   default=2*10**5)

    parser.add_argument('--lr',             type=float, default=0.1)
    parser.add_argument('--lr2',            type=float, default=0.001)
    parser.add_argument('--train_split',    type=float, default=0.8)
    parser.add_argument('--dev_split',      type=float, default=0.1)
    parser.add_argument('--test_split',     type=float, default=0.1)

    parser.add_argument('--plot_loss', action='store_false', default=True)
    parser.add_argument('--plot_embs', action='store_false', default=True)
    parser.add_argument('--scale_params', action='store_true', default=False)
    parser.add_argument('--direct_connections', action='store_true', default=False)

    args = parser.parse_args()
    return args


