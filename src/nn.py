import talos
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, z, y):
        return self.forward(z, y)

    def forward(self, z, y):
        *B, D = z.shape
        Batch_dims = np.prod(B)
        # flatten z to apply simple indexing:
        z = z.reshape(Batch_dims,D)
        #print("CROSS ENTROPY LOSS:")
        #print(z.shape)
        a = talos.max(z, dim=1, keepdims=False)
        #print(a.shape)
        logits = talos.exp(z - talos.max(z, dim=1, keepdims=True))
        #print(logits.shape)

        logits = logits / talos.sum(logits, dim= 1, keepdims=True)
        #print(logits.shape)

        y = y.reshape(Batch_dims)
            
        # get cross-entropy loss:
        log_losses = talos.log(logits[np.arange(Batch_dims), y])
        loss = -talos.sum(log_losses) / (Batch_dims)
        return loss