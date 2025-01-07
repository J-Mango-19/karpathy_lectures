import torch
import matplotlib.pyplot as plt

from utils import make_dataset, evaluate, parse_args, plot_embs
from train import train
from model import MLP_LM

if __name__ == '__main__':
    args = parse_args()

    # make dataset
    with open('data/names.txt', 'r') as f:
        words = f.read().splitlines()

    x_train, y_train, x_dev, y_dev, x_test, y_test, itos, stoi, n_classes = make_dataset(words, args.ctx_len, args.train_split, args.dev_split, args.test_split)

    # initialize and train
    model = MLP_LM(args.emb_dim, args.ctx_len, args.hidden_dim, args.scale_params, args.direct_connections, n_classes)
    losses, lrs = train(model, x_train, y_train, args.num_train_steps, args.lr, args.lr2, args.batch_size)

    # evaluate negative log-likelihood
    train_nll, dev_nll, test_nll = evaluate(model, x_train, y_train), evaluate(model, x_dev, y_dev), evaluate(model, x_test, y_test)
    print(f'Train split nll: {train_nll:.3f} | Dev Split nll: {dev_nll:.3f} | Test Split nll: {test_nll:.3f}')

    # sample and plot
    samples = model.sample(args.num_samples, itos)
    print("Generated Samples: ")
    for s in samples:
        print(f"\t{s}")

    if args.plot_loss:
        plt.plot(losses)
        plt.savefig('loss.png')
        plt.close()
        plt.plot(lrs)
        plt.savefig('lrs.png')

    if args.plot_embs and args.emb_dim == 2:
        plot_embs(model, itos)

