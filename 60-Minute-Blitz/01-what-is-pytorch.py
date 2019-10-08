# 01 - What is Pytorch
# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py


# There are 2 uses for Pytorch:
# A GPU-capable replacement for NumPy
# A flexible deep learning platform

# The core objects in Pytorch are Tensors - which are similar to NumPy arrays, i.e. multidimensional 
# grids of numbers.

import torch
print(torch.__version__)  # 1.1.0 as-of 8th October 2019


# We can create tensors through a variety of functions
 
#  e.g. uninitialised:
x = torch.empty(5,3)
print(x)
# tensor([[4.2318e+21, 1.8943e+23, 1.1727e-19],
#         [2.9538e+21, 1.8469e+25, 7.3986e+31],
#         [4.9571e+28, 3.6002e+27, 8.8674e-04],
#         [6.2717e+22, 4.7428e+30, 1.8179e+31],
#         [7.0948e+22, 3.1855e-12, 1.7565e+25]])

 
# random:
x = torch.rand(9)
print(x)
# tensor([0.5878, 0.6850, 0.7007, 0.3581, 0.9350, 0.0717, 0.5464, 0.2390, 0.4083])


# We can also specify the type of the array's contents - its 'dtype'
x = torch.zeros(2,3,4, dtype = torch.long)
print(x)
print(torch.long)  # torch.int64
# tensor([[[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 0, 0, 0]],

#         [[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 0, 0, 0]]])

# The above is a 3D tensor - effectively two 3x4 matrices; so the indices work inward. The first specifying 
# the number of outer-most arrays (here, 2), the second the number of arrays within each of these outermost
# ones (here, 3), then finally the size of the inner-most 1D array (here, 4) 

print(x.shape)  # torch.Size([2, 3, 4])

# So the first element of x will be the first outermost array - and have shape (3,4):
print(x[0])
# tensor([[0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0]])
print(x[0].shape)  # torch.Size([3, 4])

# Pytorch supports chained-indexing:
print(x[1][2])   # tensor([0, 0, 0, 0])
print(x[1][2].shape) # torch.Size([4])


# The base elements are still tensor objects - but '0D', as they have no size
print(x[1][2][2])   # tensor(0)
print(x[1][2][2].shape)  # torch.Size([])


# We can also construct tensors directly from data:
x = torch.tensor([[1,2,3],[4,5,6]])
print(x)
# tensor([[1, 2, 3],
#         [4, 5, 6]])


# We can also get tensor size via its .size() method (which returns a tuple):
x_size = x.size()
print(x_size)  # torch.Size([2, 3])
# print(x_size.count())


# We can create tensors based on existing ones, e.g:
print(x.dtype)  # torch.int64
y = x.new_ones(5, 3, dtype=torch.float)
print(y)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
print(y.dtype)  # torch.float32


# Or create tensors of the same dimension as existing ones:
z = torch.rand_like(x, dtype=torch.float)
print(z)
# tensor([[0.9277, 0.3080, 0.6307],
#         [0.2183, 0.5429, 0.4861]])
print(z.shape) # torch.Size([2, 3])
print(z.dtype) # torch.float32


# Various operations have multiple syntaxes; e.g. (element-wise) addition:
x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor([[3,2,1], [6,5,4]])
z = x + y
print(z)
# tensor([[ 4,  4,  4],
#         [10, 10, 10]])
print(z.size()) # torch.Size([2, 3])

z = torch.add(x, y)
print(z)
# tensor([[ 4,  4,  4],
#         [10, 10, 10]])
print(z.size()) # torch.Size([2, 3])


# Can also provide an output tensor to hold the results; although this will be reshape to the correct results dimensions:
result = torch.zeros(2*3, dtype=torch.long)
torch.add(x, y, out = result)
print('\nResult:')
print(result)
# tensor([[ 4,  4,  4],
#         [10, 10, 10]])
print(result.size()) # torch.Size([2, 3])


# Tensors also have an .add() method (that is not in-place)
z = y.add(x)
print(y)
# tensor([[3, 2, 1],
#         [6, 5, 4]])
print(z)
# tensor([[ 4,  4,  4],
#         [10, 10, 10]])


# And a "hidden" .add_() method that is in-place
y.add_(x)
print(y)
# tensor([[ 4,  4,  4],
#         [10, 10, 10]])

y = torch.tensor([[3,2,1], [6,5,4]])

# Note: all methods that end with '_' as above mutate tensors in-place; otherwise methods 
# do not mutate in-place

# e.g:
print(x)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
x.t_()
print(x)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])


# Tensors also support NumPy-style indexing:
x.t_()
print(x)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(x[:, 1])  # prints second column of x:  tensor([2, 5])

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 8]])

print(x[0, 1:3]) # first row; 2nd & 3rd columns: tensor([2, 3])


# The .view() method resizes tensors:
y = x.view(9,1)
print('\ny:')
print(y)
# y:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6],
#         [7],
#         [8],
#         [9]])


z = x.view(1,9)
print('\nz:')
print(z)
# z:
# tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

# cf: the distinction between a 2D array whose outermost array contains only 1 inner array; and a truly 1D array 
a = x.view(9)
print('\na:')
print(a) # tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a.size()) # torch.Size([9])
print(z.size()) # torch.Size([1, 9])

# Albeit, comparison is broadcasted:
print(a == z) # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8)
print(torch.all(a == z))  # tensor(1, dtype=torch.uint8)

# However true a equality-check fails:
print(a.equal(z)) # False


# The .item() method retrieves the element of a one-element tensor as a Python object:
print(a[0]) # tensor(1)
print(a[0].item()) # 1

# But only works with one-element tensors:
try:
    print(a.item())
except Exception as e:
    print(repr(e))   # ValueError('only one element tensors can be converted to Python scalars')


# We can swap between NumPy arrays and Pytorch tensors freely - and both will refer to the same undelying 
# memory locations (if the Torch tensor is on the CPU):
import numpy as np

b = a.numpy()
print(a)  # tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(b)  # [1 2 3 4 5 6 7 8 9]
print(type(b)) # <class 'numpy.ndarray'>

print(a is b) # False - not clear why this fails; perhaps changes to a and transmitted to b automatically via torch
print(id(a)) # 139891242054640
print(id(b)) # 139891242018176


a.add_(1)
print(a) # tensor([ 2,  3,  4,  5,  6,  7,  8,  9, 10]) -  the 1 is broadcasted to every array element
print(b) # [ 2  3  4  5  6  7  8  9 10] - b has changed too


# Can create Pytorch tensors from NumPy arrays:
a = np.ones(5)
b = torch.from_numpy(a)
c = torch.tensor(a)

print(a, type(a))  # [1. 1. 1. 1. 1.] <class 'numpy.ndarray'>
print(b, type(b))  # tensor([1., 1., 1., 1., 1.], dtype=torch.float64) <class 'torch.Tensor'>
print(c, type(c))  # tensor([1., 1., 1., 1., 1.], dtype=torch.float64) <class 'torch.Tensor'>

np.add(a, 1, out = a)
print(a, type(a))  # [2. 2. 2. 2. 2.] <class 'numpy.ndarray'>
print(b, type(b))  # tensor([2., 2., 2., 2., 2.], dtype=torch.float64) <class 'torch.Tensor'>
print(c, type(c))  # tensor([1., 1., 1., 1., 1.], dtype=torch.float64) <class 'torch.Tensor'>

# Note c above is a copy of a and so doesn't change when it is modified, whereas b does