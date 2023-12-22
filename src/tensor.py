from typing import List
import numpy as np

# Tensor class, with __init__, backward, magic methods, and utils:
class Tensor:
    def __init__(self, data, requires_grad = False, operation = None) -> None:
        self._data = array(data)
        self.requires_grad = requires_grad
        self.operation = operation
        self.shape = self._data.shape
        if self.requires_grad:
            self.grad = np.zeros_like(data)

    def __repr__(self):
        return f"""({self._data}, requires_grad = {self.requires_grad})"""

    def data(self):
        return self._data
    
    def backward(self, grad = None):
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        if grad is None:
            grad = np.ones_like(self._data)
        # print('===========================')
        # print("grad")
        # print(grad)
        # print('self.grad')
        # print(self.grad)
        self.grad += grad

        if self.operation:
            #print(self.operation)
            self.operation.backward(grad)

    def tolist(self):
        return self._data.tolist()

    def toarray(self):
        return self._data
    
    def zero_grad(self):
        self.grad = np.zeros_like(self._data)

    def zero_grad_tree(self):
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
        op = MatMul()
        return op.forward(self, tensor(other))
    
    def __truediv__(self, other):
        op = Div()
        return op.forward(self, tensor(other))
    
    def __getitem__(self, index):
        op = Slice()
        return op.forward(self, index)

    def max(self, dim, keepdims=False):
        op = Max()
        return op.forward(self, dim, keepdims=keepdims)

    def sum(self, dim=-1, keepdims=False):
        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)

    def reshape(self, *shape):
        op = Reshape()
        return op.forward(self, shape)

    def transpose():
        pass


# Operation classes. Each has a forward and backward method:
class Add:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data + b._data
        
        self.parents = (a, b)
        self.cache = (a, b)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            out_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(out_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            out_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(out_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db)

class Neg:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad
        data = - a._data 

        self.parents = (a,)
        self.cache = a

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a = self.cache
        da = 0

        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = -dz
            a.backward(da)

class Mul:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data * b._data
        
        self.parents = (a, b)
        self.cache = (a, b)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = dz * b._data
            out_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(out_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            db = dz * a._data
            out_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(out_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db)

class Div:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data / b._data
        
        self.parents = (a, b)
        self.cache = (a, b)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = dz / b._data
            out_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(out_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            db = - dz * a._data / (b._data ** 2)
            out_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(out_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db)

class MatMul:
    def __init__(self) -> None:
        pass

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad

        data = a._data @ b._data
        
        self.parents = (a, b)
        self.cache = (a, b)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, b = self.cache
        da, db = dz, dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            b_dims = len(b.shape)
            # Transpose the last 2 dimentions:
            b_dims_transposed = [i for i in range(b_dims)][:-2] + [b_dims - 1, b_dims - 2]
            da = dz @ b._data.transpose(b_dims_transposed)
            a.backward(da)

        # Find gradients relative to "b", and make recursive calls if it requires gradients:
        if b.requires_grad:
            a_dims = len(a.shape)
            a_dims_transposed = [i for i in range(a_dims)][:-2] + [a_dims - 1, a_dims - 2]
            db = a._data.transpose(a_dims_transposed) @ dz
            b.backward(db)

class Slice:
    def __init__(self) -> None:
        pass

    def forward(self, a, index):
        requires_grad = a.requires_grad

        data = a._data[index]
        
        self.parents = (a,)
        self.cache = (a, index)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, index =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.zeros_like(a._data)
            da[index] = dz
            a.backward(da)

class Max:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims=False):
        requires_grad = a.requires_grad
        data = np.max(a._data, axis=dim, keepdims=keepdims)

        if keepdims:
            data = np.ones(a.shape) * data

        self.parents = (a,)
        self.cache = (a, data, dim, keepdims)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, max, dim, keepdims =  self.cache
        da = dz
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            if a.shape != da.shape:
                # Brodcast upstream derivative to the size of "a":
                da = np.expand_dims(da, axis=dim)
                da = da * np.ones_like(a._data)
                # Brodcast upstream output (max) to the size of "a":
                max = np.expand_dims(max, axis=dim)
                max = max * np.ones_like(a._data)
            da = da * np.equal(a._data, max)
            a.backward(da)
       
class Sum:
    def __init__(self) -> None:
        pass

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad

        data = a._data.sum(axis=dim, keepdims=keepdims)
        
        self.parents = (a,)
        self.cache = (a)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a =  self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = np.ones(a.shape) * dz
            a.backward(da)

class Exp:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad

        data = np.exp(a._data)
        
        self.parents = (a,)
        self.cache = (a, data)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, z = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = z * dz
            a.backward(da)

class Log:
    def __init__(self) -> None:
        pass

    def forward(self, a):
        requires_grad = a.requires_grad

        data = np.log(a._data)
        
        self.parents = (a,)
        self.cache = (a, data)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a, z = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = (1 / a._data) * dz
            a.backward(da)

class Reshape:
    def __init__(self) -> None:
        pass

    def forward(self, a, shape):
        requires_grad = a.requires_grad

        data = a._data.reshape(*shape)

        self.parents = (a,)
        self.cache = (a)

        return Tensor(data, requires_grad=requires_grad, operation=self) 
    
    def backward(self, dz):
        a = self.cache
        da = dz
        
        # Find gradients relative to "a", and make recursive calls if it requires gradients:
        if a.requires_grad:
            da = dz.reshape(a.shape)
            a.backward(da)


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