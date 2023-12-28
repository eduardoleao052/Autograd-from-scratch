from typing import List
import numpy as np

# Tensor class, with __init__, backward, magic methods, and utils:
class Tensor:
    def __init__(self, data, requires_grad = False, operation = None) -> None:
        self._data = array(data)
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.shape = self._data.shape
        if self.requires_grad:
            self.grad = np.zeros_like(data)

    def __repr__(self):
        return f"""({self._data}, requires_grad = {self.requires_grad})"""

    def data(self):
        return self._data
    
    def backward(self, grad = None, z = None):
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        if grad is None:
            grad = np.ones_like(self._data)

        self.grad += grad

        if z is not None:
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def tolist(self):
        ''' Turns the Tensor into a python list. '''
        return self._data.tolist()

    def toarray(self):
        ''' Turns the Tensor into a numpy array. '''
        return self._data
    
    def zero_grad(self):
        ''' Reset the Tensor's gradients to zero. '''
        self.grad = np.zeros_like(self._data)

    def zero_grad_tree(self):
        ''' Reset the gradients of this Tensor, and of all of the Tensors that led to it. '''
        self.zero_grad()
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grad_tree()
            self.operation = None

    def __add__(self, other):
        """ New = self + other """
        op = Add()
        return op.forward(self, tensor(other))

    def __radd__(self, other):
        """ New = other + self """
        op = Add()
        return op.forward(self, tensor(other))

    def __iadd__(self, other):
        """ self += other """
        op = Add()
        return op.forward(self, tensor(other))

    def __sub__(self, other):
        """ New = self - other """
        return self + -other

    def __rsub__(self, other):
        """ New = other - self """
        return other + -self

    def __isub__(self, other):
        """ self -= other """
        return self + -other
    
    def __neg__(self):
        """ self = -self """
        op = Neg()
        return op.forward(self) 

    def __mul__(self, other):
        """ New = self * other """
        op = Mul()
        return op.forward(self, tensor(other))

    def __rmul__(self, other):
        """ New = other * self """
        op = Mul()
        return op.forward(self, tensor(other))

    def __imul__(self, other):
        """ self *= other """
        op = Mul()
        return op.forward(self, tensor(other))

    def __matmul__(self, other):
        """ New = self @ other """
        op = MatMul()
        return op.forward(self, tensor(other))
    
    def __truediv__(self, other):
        """ New = self / other """
        op = Div()
        return op.forward(self, tensor(other))
    
    def __getitem__(self, index): 
        """ New = self[index] """
        op = Slice()
        return op.forward(self, index)

    def __gt__(self, other):
        """ New = self > other """
        return self._data > array(other)

    def max(self, dim, keepdims=False):
        """
        Returns the largest values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be reduced (only largest remains).
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Max()
        return op.forward(self, dim, keepdims=keepdims)

    def sum(self, dim=-1, keepdims=False):
        """
        Returns the sum of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be summed across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)

    def mean(self, dim=-1, keepdims=False):
        """
        Returns the mean of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention to be averaged across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Mean()
        return op.forward(self, dim, keepdims=keepdims)

    def var(self, dim=-1, keepdims=False):
        """
        Returns the variance of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention the variance will be computed across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Var()
        return op.forward(self, dim, keepdims=keepdims)

    def reshape(self, *shape):
        """
        Returns the original tensor reshaped to the new shape given.
        Example: (16, 8, 4), *shape =(2, 32, 8) -> (2, 32, 8)

        @param *shape (integers): new shape of the tensor.
        """
        op = Reshape()
        return op.forward(self, shape)

    def transpose(self, *dims):
        """
        Returns the original tensor with the two given dimentions transposed.
        Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8)

        @param *dims (integers): two dimentions to be transposed.
        """
        op = Transpose()
        return op.forward(self, *dims)

    def masked_fill(self, condition, value):
        """
        Returns the original tensor with the values where condition is True set to "value".

        @param condition (Array-like): two dimentions to be transposed.
        @param value (float): value to fill Tensor with, where condition is True.
        """
        op = MaskedFill()
        return op.forward(self, array(condition), value )

# Parameter subclass, inherits from Tensor:
class Parameter(Tensor):
    ''' Subclass of Tensor which always tracks gradients. '''
    def __init__(self, data, requires_grad = True, operation = None) -> None:
        super().__init__(data, requires_grad=requires_grad, operation=operation)

        
# Operations between two tensors:
class Add:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data + b._data
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z
    
    def backward(self, dz, z):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class Neg:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad
        data = - a._data 

        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)

        self.cache = a

        return z 
    
    def backward(self, dz, z):
        a = self.cache
        da = 0

        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = -dz
            a.backward(da, z)

class Mul:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data * b._data
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = dz * b._data
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            db = dz * a._data
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class Div:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data / b._data
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = dz / b._data
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            db = - dz * a._data / (b._data ** 2)
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class MatMul:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
        data = a._data @ b._data

        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            # Transpose the last 2 dimentions:
            da = dz @ b._data.swapaxes(-1,-2)
            
            # Get difference between "a" size and upstream "da" size, to broadcast grad into "a":
            in_dim = len(a.shape)
            grad_dim = len(da.shape)
            # print("DA")
            # print(da.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            # print("DA2")
            # print(da.shape)
            a.backward(da, z)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            db = a._data.swapaxes(-1,-2) @ dz

            # Get difference between "b" size and upstream "db" size, to broadcast grad into "b":
            in_dim = len(b.shape)
            grad_dim = len(db.shape)

            # print("DB")
            # print(db)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)
            # print("DB2")
            # print(db)
            b.backward(db, z)

# Statistics operations:
class Max:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims=False):
        requires_grad = a.requires_grad
        data = np.max(a._data, axis=dim, keepdims=keepdims)
        if keepdims:
            data = np.ones(a.shape) * data


        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data, dim, keepdims)

        return z
    
    def backward(self, dz, z):
        a, data, dim, keepdims =  self.cache
        da = dz
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            if a.shape != da.shape:
                # Brodcast upstream derivative to the size of "a":
                da = np.expand_dims(da, axis=dim)
                da = da * np.ones_like(a._data)
                # Brodcast upstream output (max) to the size of "a":
                max = np.expand_dims(data, axis=dim)
                max = max * np.ones_like(a._data)
            da = da * np.equal(a._data, max)
            a.backward(da, z)
       
class Sum:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad

        data = a._data.sum(axis=dim, keepdims=keepdims)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.ones(a.shape) * dz
            a.backward(da, z)

class Mean:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad

        data = a._data.mean(axis=dim, keepdims=keepdims)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim)

        return z
    
    def backward(self, dz, z):
        a, dim =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.ones(a.shape) * dz
            da /= np.prod(np.array(a.shape)[dim])
            a.backward(da, z)

class Var:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad

        data = a._data.var(axis=dim, keepdims=keepdims)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim)

        return z
    
    def backward(self, dz, z):
        a, dim =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.ones(a.shape) * dz
            da = da * 2 * (a._data - a._data.mean(axis=dim, keepdims=True)) / np.prod(np.array(a.shape)[dim])
            a.backward(da, z)

# Element-wise operations:
class Exp:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad

        data = np.exp(a._data)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = data * dz
            a.backward(da, z)

class Log:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad

        data = np.log(a._data)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = (1 / a._data) * dz
            a.backward(da, z)

class Sqrt:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad

        data = np.sqrt(a._data)
        
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = (1 / 2) * (1 / data) * dz
            a.backward(da, z)

# Tensor Operations:
class Reshape:
    def __init__(self) -> None:
        pass

    def forward(self, a, shape):
        requires_grad = a.requires_grad

        data = a._data.reshape(*shape)

        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:

            da = dz.reshape(a.shape)
 
            a.backward(da, z)

class Transpose:
    def __init__(self) -> None:
        pass

    def forward(self, a, *dims):
        requires_grad = a.requires_grad
        
        data = a._data.swapaxes(*dims)

        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:

            da = dz.swapaxes(*dims)
 
            a.backward(da, z)

class Cat:
    def __init__(self) -> None:
        pass

    def forward(self, tensors: tuple, dim: int):

        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
        print(tensors[0]._data.shape)
        data = np.concatenate([tensor._data for tensor in tensors], axis=dim)
        print(data.shape)
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache
        
        dz = np.split(dz, len(tensors), dim)

        # Find gradients relative to each tensor in "tensor", and make recursive calls if it requires gradients:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:

                di = dz[i]
    
                tensor.backward(di, z)

class Stack:
    def __init__(self) -> None:
        pass

    def forward(self, tensors: tuple, dim: int):

        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
        print(tensors[0]._data.shape)
        data = np.stack([tensor._data for tensor in tensors], axis=dim)
        print(data.shape)
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache
        print(dz.shape)
        dz = np.split(dz, len(tensors), dim)
        print(len(dz))
        # Find gradients relative to each tensor in "tensor", and make recursive calls if it requires gradients:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:

                di = dz[i].reshape(tensor._data.shape)
    
                tensor.backward(di, z)

class MaskedFill:
    def __init__(self) -> None:
        pass

    def forward(self, a, condition, value):
        requires_grad = a.requires_grad

        data = np.where(condition, a._data, value)


        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z 
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:

            da = dz
 
            a.backward(da, z)

class Slice:
    def __init__(self) -> None:
        pass

    def forward(self, a, index):
        requires_grad = a.requires_grad

        data = a._data[index]

        z = Tensor(data, requires_grad=requires_grad, operation=self) 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, index)

        return z
    
    def backward(self, dz, z):
        a, index =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.zeros_like(a._data)
            da[index] = dz
            a.backward(da, z)


# Some helper functions to transition between iterable data types:
def list(data):
    if isinstance(data, List):
        return data
    else: 
        return data.tolist()

def array(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, Tensor):
        return data.toarray()
    else: 
        return np.array(data)
    
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)