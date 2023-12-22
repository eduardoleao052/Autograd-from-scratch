from tensor import *
import numpy as np

# Methods to iniate Tensors:
def tensor(data, requires_grad = False):
    '''
    Creates new instance of the Tensor class.

    @param data (Array-like): Iterable containing the data to be stored in the Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containing "data".
    '''
    return Tensor(data, requires_grad=requires_grad)

def zeros(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with zeros.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining zeros with "shape" shape.
    '''
    data = np.zeros(shape)
    return Tensor(data, requires_grad=requires_grad)

def ones(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with ones.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining ones with "shape" shape.
    '''
    data = np.ones(shape)
    return Tensor(data, requires_grad=requires_grad)

def randint(low: int = 0, high: int = None, shape: tuple = (1), requires_grad: bool = False):
    '''
    Creates new instance of the Tensor class, filled with random integers.

    @param low (int): lowest integer to be generated. [OPTIONAL]
    @param high (int): one above the highest integer to be generated.
    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining random integers with "shape" shape.
    '''
    if type(high).__name__ == 'int':
        data = np.random.randint(low, high, size=shape)
    else:
        data = np.random.randint(low, size=high)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, requires_grad = False):
    '''
    Creates new instance of the Tensor class, filled with floating point numbers in a normal distribution.

    @param shape (tuple): iterable with the shape of the resulting Tensor.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with "shape" shape.
    '''
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)

def zeros_like(other: Tensor, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor, and filled with zeros.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining zeros with other Tensor's shape.
    '''
    shape = other.shape
    return zeros(shape=shape, requires_grad=requires_grad)

def ones_like(other: Tensor, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor, and filled with ones.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining ones with other Tensor's shape.
    '''
    shape = other.shape
    return ones(shape=shape, requires_grad=requires_grad)

def randn_like(other: Tensor, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor,
    and filled with random floats in a normal distribution.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with other Tensor's shape.
    '''
    shape = other.shape
    return randn(shape=shape, requires_grad=requires_grad)

def randint_like(other: Tensor, low: int, high: int=0, requires_grad = False):
    '''
    Creates new instance of the Tensor class with same shape as given Tensor,
    and filled with random integers in the given distribution.
    @param other (Tensor): Tensor to copy shape from.
    @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.

    @returns Tensor (Tensor): Tensor containining normally distributed floats with other Tensor's shape.
    '''
    shape = other.shape
    if high == 0:
        return randint(low, shape, requires_grad=requires_grad)
    else:
        return randint(low, high, shape, requires_grad=requires_grad)

# Methods to work with Tensors:
def max(a, dim=-1, keepdims=False):
    return a.max(dim=dim, keepdims=keepdims)

def argmax(a, dim=-1, keepdims=False):
    return Tensor(np.argmax(a._data,axis=dim,keepdims=keepdims))

def sum(a, dim=-1, keepdims=False):
    return a.sum(dim=dim, keepdims=keepdims)

def exp(a):
    op = Exp()
    return op.forward(a)

def log(a):
    op = Log()
    return op.forward(a)

def reshape(a, shape: tuple):
    return a.reshape(*shape)

def transpose(a, dims: tuple):
    return a.transpose(*dims)