import math

# value wrapper for scalars automatically tracks gradients 
class Value:
    def __init__(self, data, children=(), op=None):
        self.data = data
        self._prev = set(children)
        self.op = op
        self.grad = 0
        self._backward = lambda : None

    def __repr__(self):
        return str(self.data)

    def __add__(self, other):
        new_val = Value(self.data + other.data, (self, other), op='+')
        def _backward():
            self.grad += new_val.grad
            other.grad += new_val.grad
        new_val._backward = _backward
        return new_val

    def __sub__(self, other):
        new_val = Value(self.data - other.data, (self, other), op='-')
        def _backward():
            self.grad += new_val.grad
            other.grad -= new_val.grad
        new_val._backward = _backward
        return new_val

    def __mul__(self, other):
        new_val = Value(self.data * other.data, (self, other), op='*')
        def _backward():
            self.grad += new_val.grad * other.data
            other.grad += new_val.grad * self.data
        new_val._backward = _backward
        return new_val

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        new_val = Value(self.data**other, (self,), op=f'**{other}')
        def _backward():
            self.grad += other.data * self.data**(other-1) * new_val.grad
        new_val._backward = _backward
        return new_val

    def exp(self):
        new_val = Value(math.exp(self.data), (self,), op='exp')
        def _backward():
            self.grad += new_val.data * new_val.grad
        new_val._backward = _backward
        return new_val

    def log(self):
        new_val = Value(math.log(self.data), (self,), op='log')
        def _backward():
            self.grad += 1 / self.data * new_val.grad
        new_val._backward=_backward
        return new_val

    def relu(self):
        new_val = Value(self.data if self.data > 0 else 0, (self,), 'relu')
        def _backward():
            self.grad += new_val.grad if new_val.data > 0 else 0
        new_val._backward = _backward
        return new_val

    def backward(self):
        # topological order all the children in the graph
        # This means we rearrange the DAG of computation such that each parent has its gradients computed before its children
        # This order is important since childrens' gradients depend on their parents' gradients
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # the last Value has no dependencies to alter its gradient, so derivative of something wrt itself is just 1
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 0
        for v in topo:
            v.grad = 0
