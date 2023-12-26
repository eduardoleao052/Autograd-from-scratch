import numpy as np

class SGD:
    def __init__(self, params, lr=1e-3, reg=0) -> None:
        self.params = params
        self.lr = lr
        self.reg = reg
        

    def step(self):
        for param in self.params:
            param._data = param._data - (self.lr * param.grad) - (self.lr * self.reg * param._data)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, reg=0) -> None:
        self.params = params
        self.lr = lr
        self.reg = reg
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.eps = eps
        self.reg = reg
        for param in self.params:
            param.m = np.zeros(param.shape)
            param.v = np.zeros(param.shape)

    def step(self):
            for param in self.params:
                param.m = (param.m * self.b1 + (1 - self.b1) * param.grad) 
                param.v = (param.v * self.b2 + (1 - self.b2) * np.square(param.grad))

                param._data = param._data - (self.lr * param.m) / (np.sqrt(param.v) + self.eps) - self.reg * self.lr * param._data

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()