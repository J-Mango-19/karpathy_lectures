from autograd import Value
from graphviz import Digraph
from draw import draw_dot

def sum_vector(vec):
    vsum = Value(0)
    for l in vec:
        vsum += l
    return vsum

x = Value(1.5)
w1 = Value(1.2); w2 = Value(1.0)

print("True distribution to fit: [0, 1]")

for i in range(5):
    print(f"\n ---------- training step {i + 1} ---------- ")
    logits = [w1 * x, w2 * x]
    print(f"Computed Logits: [{logits[0].data:.3f}, {logits[1].data:.3f}]")

    logits_exp = [logit.exp() for logit in logits]
    logits_exp_sum = sum_vector(logits_exp)
    probs = [logit_exp / logits_exp_sum for logit_exp in logits_exp]

    print(f"Output probs:    [{probs[0].data:.3f}, {probs[1].data:.3f}]")

    # output we want is dim 1
    log_likelihood = probs[1].log()

    nll = Value(-1) * log_likelihood
    print(f"nll: {nll.data:.3f}")
    nll.backward()

    if i == 0:
        dot = draw_dot(nll)
        dot.render('gout')

    w1.data -= w1.grad
    w2.data -= w2.grad


    nll.zero_grad()
