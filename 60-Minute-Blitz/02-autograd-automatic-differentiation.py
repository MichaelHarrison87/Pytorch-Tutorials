# 02 - Autograd: Automatic Differentiation
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
import torch


# The autograd package underpins all neural networks in Pytorch - it provides automatic differentiation
# for all operations on tensors. 

# It is a define-by-run framework - with backprop defined by how the code is run, and each iteration can
# be different


# The central class of Pytorch is torch.Tensor. If its attribute require_grad is set to True, 
# it starts to track all operations on it. Then when the computation is finished, can 
# call the .backward() method to automatically calculate all gradients. These gradients for the Tensor 
# are accumulated in its .grad attribute.

# Tensors can be .detach()'d from the computation history, to prevent future computation being tracked.

# Tracking history can also be turned off by wrapping code in no_grad blocks as below:
with torch.no_grad():
    pass

# This will disable tracking even for Tensors with require_grad = True. This can be useful when 
# evaluating models - as may have trainable params for which we don't require the gradients.


# Another key class is Function - which together with Tensor forms the acyclic computation graph
# that encodes the full history of computation. 

# Each Tensor has a .grad_fn attribute, that references the Function that created the Tensor (except for 
# user-created tensors - where this is None).

# To calculate derivatives, call the .backward() method on the Tensor. If the Tensor only has one element,
# (i.e. its a scalar) then .backward() doesn't need any arguments. Otherwise, need to specify a gradient
# tensor of matching shape

x = torch.ones(2,2)
print(x)
# tensor([[1., 1.],
#         [1., 1.]])
print(x.requires_grad) # False  - requires_grad off by default

# Descendants inherit the requires_grad setting from their parents in the computation graph:
y = x +2
print(y)
# tensor([[3., 3.],
#         [3., 3.]])
print(y.requires_grad) # False


# But can specify it when creating the tensor:
x = torch.ones(2,2, requires_grad = True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
print(x.requires_grad) # True


# With requires_grad True, for simple operations the grad_fn is automatically added to the result
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
print(y.requires_grad) # True

z = x * 2
print(z)
# tensor([[2., 2.],
#         [2., 2.]], grad_fn=<MulBackward0>)
print(z.requires_grad) # True

z2 = x + z
print(z2)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
print(z2.requires_grad)  # True

out = z2.mean()
print(out)  # tensor(3., grad_fn=<MeanBackward0>)


# Note we can change the requires_grad attribute in-place using the '_' suffix:
a = torch.Tensor([1,2,3])
print(a)  # tensor([1., 2., 3.])
print(a.requires_grad)  # False
a.requires_grad_(True)
print(a) # tensor([1., 2., 3.], requires_grad=True)
print(a.requires_grad)  # True


# We can get gradients - out is a scalar so .backward() requires no arguments:
# Recall: z2 = x + z = x +  2*x = 3x
# out is then mean of z2 - i.e. out = (1/4) * sum (i from 1 to 4) of 3*x[i] 
# so d(out)/d(x[i]) = 3/4 - as seen below:
out.backward()
print(x.grad)
# tensor([[0.7500, 0.7500],
#         [0.7500, 0.7500]])


# For vectir-valued function y = f(x), where x in R^n and y in R^m - the gradient is 
# the  m x n Jacobian matrix:
# J = dy[1]/dx[1] ... dy[1]/dx[n]
#     .                   .
#     .                   .
#     dy[m]/dx[1] ... dy[m]/dx[n]


# torch.autograd is a tool for computing vector-Jacobian product, i.e. (v^T)*J - where v = (v[1],...,v[m])^T
# So the product is in R^n, and its i^th component is the dot product of v with the gradient of y wrt x[i].

# If v is the gradient of a scalar function, e.g. l = g(y)  (l is in R), with v = (dl/dy[1], ..., dl/dy[m])^T
# then we have vector-Jacobian product:

# (J^T)*v = dy[1]/dx[1] ... dy[m]/dx[1] * dl/dy[1]
#               .                  .        .
#               .                  .        .
#           dy[1]/dx[n] ... dy[m]/dx[n]   dl/dy[m]
#
#          = (Sum (i=1 to m) dy[i]/dx[1] * dl/dy[i], ..., Sum (i=1 to m) dy[i]/dx[n] * dl/dy[i])^T
#          = (dl/dx[1], ..., dl/dx[n])^T   by the Chain Rule



# The vector-Jacobian product lets us feed in external gradients to a model with non-scalar output.
# Consider:

print('\nVector-Jacobian Product')
x = torch.rand(3, requires_grad=True)

y = x * 2
print(x)  # tensor([0.3884, 0.0157, 0.1033], requires_grad=True)
print(y)  # tensor([0.7768, 0.0315, 0.2067], grad_fn=<MulBackward0>)
print(y.data.norm())  # tensor(0.8044)  - i.e. sqrt(0.7768^2 + 0.0315^2 + 0.2067^2)

# Double elements of y till y's norm exceeds 1000 - i.e. stretch the vector y in R^3 by factor of 2 
# repeatedly until its length > 1000
while y.data.norm() < 1000:
    y = y * 2
    
print(y)  # tensor([1590.8469,   64.5117,  423.3049], grad_fn=<MulBackward0>) -  norm is 1647.46;  423.3049/0.2067 = 2048 = 2^11
        

# Since y is not a scalar, autograd can't compute the full Jacobian directly, but we can pass a vector to
# the .backward() method to compute the vector-Jacobian product
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)  # tensor([4.0960e+02, 4.0960e+03, 4.0960e-01])

# Note: the final y here is x * 2^12  = x * 4096. More specifically, y is the 3-vector with components y[i] = 4096 * x[i] 
# So the Jacobian is just the square 3x3 matrix with 4096 along the diagonal, i.e. 4096 * I:
# J = dy[1]/dx[1] dy[1]/dx[2] dy[1]/dx[3] =  4096  0    0
#     dy[2]/dx[1] dy[2]/dx[2] dy[2]/dx[3]     0  4096   0
#     dy[3]/dx[1] dy[3]/dx[2] dy[3]/dx[3]     0    0   4096

# So the vector-Jacobian product (J^T)*v = (4096 * 0.1, 4096 * 1.0, 4096 * 1e-4)^T = (409.6, 4096, 0.4096)^T
# Which is what x.grad gives us