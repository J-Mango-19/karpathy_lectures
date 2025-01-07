import torch
import math
import torch.nn.functional as F

def get_random_batch(x, y, batch_size):
    num_examples = x.shape[0]
    random_indices = torch.randint(low=0, high=num_examples, size=(batch_size,))
    return x[random_indices], y[random_indices]


def train(model, x, y, num_steps=200000, lr1=0.1, lr2=0.005, batch_size=32):
    print(f'Training for {num_steps=}')
    model.train()
    loss_list = []
    lr_list =[]

    for k in range(num_steps):
        x_batch, y_batch = get_random_batch(x, y, batch_size)

        # forward pass
        logits = model(x_batch)
        nll_loss = F.cross_entropy(logits, y_batch)

        # zero grad
        for p in model.parameters():
            p.grad = None

        # backprop
        nll_loss.backward()

        # parameter update
        lr = lr1 - (k**1.5 * (lr1 - lr2) / num_steps**1.5)
        for p in model.parameters():
            p.data += -lr * p.grad

        loss_list.append(nll_loss.item())
        lr_list.append(lr)

    return loss_list, lr_list


