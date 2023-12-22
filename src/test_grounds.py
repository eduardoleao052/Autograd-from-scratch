import talos
import nn
import numpy as np
np.set_printoptions(precision=2)

loss_func = nn.CrossEntropyLoss()

x = talos.randn((8,16))
w = talos.randn((16,32), requires_grad=True)
w2 = talos.randn((32,8), requires_grad=True)
y = np.random.randint(0,8,8)

for _ in range(500000):
    z = x @ w
    z = z @ w2


    # flatten z to apply simple indexing:
    logits = talos.exp(z - talos.max(z, dim=1, keepdims=True))
    logits = logits / talos.sum(logits, dim= 1, keepdims=True)


    # get cross-entropy loss:
    log_losses = talos.log(logits[np.arange(8), y])
    loss = -talos.sum(log_losses) / 8

    print(loss._data)

    loss.backward()
        
    w = w - (w.grad * 0.05)
    w2 = w2 - (w2.grad * 0.05)

    loss.zero_grad_tree()
