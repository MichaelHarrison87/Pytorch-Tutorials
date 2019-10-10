# 03 - Neural Networks
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


# Neural networks are constructed using the torch.nn package. This relies on autograd to define models
# and differentiate them.

# An nn.Module contains layers, and a method .forward(input) that returns the output of a forward pass.

# The typical training procedure for a neural network is:
# Define the network, and its trainable weights
# Iterate over a dataset of inputs
# Forward-pass the inputs through the network to obtain its output
# Calculate the loss of this output - how far it is from being correct (per the dataset's labels)
# Back-propagate gradients to the network's parameters
# Update the network's weights using some update rule, e.g: new_weight = old_weight - learning_rate * gradient 


# We demonstrate an example below:
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    All models need to subclass nn.Module
    The code for nn.Module requires its subclasses to implement a .forward() method
    Specifying this .forward() method means autograd will automatically define the .backward() method
    Can use any Tensor operations in the .forward() method 
    
    Note: the intended input image spatial dimensions are 32x32. 
    The conv layers have no padding by default, so these will each reduce the dimension by 2.
    The max pooling layers use 2x2 kernels, with stride defaulting to the kernel size. 
    So the max pooling layers will halve the spatial dimensions.
    
    So the sequence of spatial dimensions over the convolutional body is:
    input -     32x32  
    conv1 -     30x30
    maxpool1 -  15x15
    conv2 -     13x13
    maxpool2 -  6x6  (tutorial says 6x6, and PyTorch probably crops the last row/column for odd-dimensions, i.e. 13x13 above cropped to 12x12)
    """
    
    def __init__(self):
        """Set-up the network, then specify each trainable layer"""        
        super(Net, self).__init__()
        
        # CONVOLUTIONAL BODY
        # 1 input image channel, 6 output channels, 3x3 square conv filters:
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
        # NETWORK HEAD - DENSE LAYERS
        # affine operation: y = Wx + b; note the orig image spatial dims are 6x6; linear activation function
        self.fc1 = nn.Linear(6 * 6 * 16, 120)   # 16 input channels per conv2; we specify 120 output neurons; NB: biases included by default
        self.fc2 = nn.Linear(120, 84) 
        self.fc3 = nn.Linear(84, 10)
        
    
    def forward(self, x):
        """Specifies forward pass for entire network. Input x will be the original image"""
        
        # 1st Conv + Max Pooling Layer
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))  # 2x2 kernel size for the max pooling
        
        # 2nd Conv + Max Pooling Layer
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # pooling window assumed to be square if only 1 dimension supplied for kernel size
        
        # Resize output of 2nd conv block:
        x = x.view(-1, self.num_flat_features(x))
        
        # Now dense layers + output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def num_flat_features(self, x):
        """Returns the size of the 1D "flattened" vector, which reshapes the 2D (spatial dimensions) 
        output of the network's convolutional body""" 
        size = x.size()[1:]   # the first dimension indexes the elements in the minibatch; we flatten each element individually, so remove
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
# Net(
#   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
#   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
#   (fc1): Linear(in_features=576, out_features=120, bias=True)
#   (fc2): Linear(in_features=120, out_features=84, bias=True)
#   (fc3): Linear(in_features=84, out_features=10, bias=True)
# )

# Can get the network's learnable parameters:
print(net.parameters())  # <generator object Module.parameters at 0x7f68eb1cdd68>
params = list(net.parameters())
print(len(params))  # 10
print(params[0])  # printed below - 6 3x3x1 grids of params, per the conv1 layer
# Parameter containing:
# tensor([[[[-0.2730,  0.2236, -0.3239],
#           [ 0.2115, -0.0425,  0.2247],
#           [-0.0654,  0.2291, -0.3313]]],


#         [[[-0.3287, -0.1338,  0.2229],
#           [-0.1355,  0.0688,  0.1971],
#           [-0.3026, -0.1134,  0.3081]]],


#         [[[-0.3213,  0.1931, -0.3305],
#           [-0.0472,  0.2542,  0.1694],
#           [-0.1009, -0.0919,  0.2019]]],


#         [[[ 0.2740, -0.1071,  0.0963],
#           [ 0.3061, -0.1397,  0.2647],
#           [-0.2039, -0.0873,  0.1427]]],


#         [[[ 0.3043,  0.0427, -0.1479],
#           [ 0.1594,  0.2314,  0.0889],
#           [-0.1319,  0.1223,  0.2084]]],


#         [[[-0.1980,  0.2613, -0.1524],
#           [-0.0484, -0.0787, -0.0802],
#           [-0.0485, -0.1627,  0.2825]]]], requires_grad=True)
print(params[0].size())  # torch.Size([6, 1, 3, 3]) - 6 3x3x1 filter windows


print(params[1])  # These are the biases for the conv1 layer
# Parameter containing:
# tensor([ 0.2206,  0.2037, -0.1480,  0.1888, -0.1975, -0.2281],
#        requires_grad=True)


print(params[-1].size()) # torch.Size([10])  - 10 output neurons, per the fc3 layer

# The network has 5 layers - 2 conv2d and 3 dense layers; these are each stored in the params list, with 
# weights and biases separate. Hence: params list is of length 5*2=10 


# Try passing an input through the network:
image_input = torch.rand(1, 1, 32, 32)  # dimension are: (num_inputs, num_channels, height, width)
out = net(image_input)

# Note: nn.Module is callable, and its .__call__() method includes:  result = self.forward(*input, **kwargs)
# So calling net on image_input above invokes a forward pass through the network; we don't have to call 
# .forward() explicitly ourselves

# Also, when specifying the input - torch.nn assumes minibatches throughout, so we need to specify the number
# of items in the minibatch in the first input dimension (even if it's only 1)

# The .unsqueeze(0) method can be used to add a fake batch dimension (as the 0th dimension); e.g:
print('\nUnsqueeze:')
x = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])  # dummy single 3x3 image
print(x)
# tensor([[1., 2., 3.],
#         [4., 5., 6.],
#         [7., 8., 9.]])
print(x.size(), x[0].size()) # torch.Size([3, 3]) torch.Size([3]) - x is 2D 3x3; first element is 1D 3-element array

y = x.unsqueeze(0)
print(y)
# tensor([[[1., 2., 3.],
#          [4., 5., 6.],
#          [7., 8., 9.]]])
print(y.size(), y[0].size()) # torch.Size([1, 3, 3]) torch.Size([3, 3]) - y is 1x3x3; first element is 2D 3x3 array


print('\nInput:')
print(image_input.size()) # torch.Size([1, 1, 32, 32])

print('\nOutput:')
print(out.size()) # torch.Size([1, 10]) - as expected


# Since calling net() on the input invoked a forward pass, we can now perform a backward pass 
# (zero'ing the gradient buffers first) with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10), retain_graph=True)  # dimension of the vector (for vector-Jacobian product) must match out's


# We can now specify a loss, vs some (here, dummy) target:
print('\nLoss:')
target = torch.rand(10)
print(target.size())  # torch.Size([10])
target = target.view(1, -1) # reshape target to have a 1st dimesnion of size 1; -1 specifies all items 
print(target.size())  # torch.Size([1, 10])

# cf resize with 1st dimension of size of 2:
target_alt = target.view(2, -1)
print(target_alt.size())  # torch.Size([2, 5])

criterion = nn.MSELoss()   # instantiate the loss class first
loss = criterion(out, target)  # then call it on our specific output and targets; 
# Note simply doing loss = nn.MSELoss(out, target) resulted in an error, as can't instantiate the loss with these tensors; instead we /call/ the instance on them, as here  


print(out) # tensor([[-0.0464,  0.0528,  0.0878,  0.0641,  0.0924,  0.0426,  0.0344, -0.0467, 0.0205,  0.1240]], grad_fn=<AddmmBackward>)
print(target) # tensor([[0.1165, 0.0041, 0.6117, 0.3702, 0.8794, 0.4346, 0.0966, 0.8392, 0.1809, 0.9308]])
print(loss)  # tensor(0.2636, grad_fn=<MseLossBackward>)   - verified this is indeed the mean of the squared elementwise differences between output and target above


# Note that loss is a scalar, and we can do a backward pass to calc the gradients of all params wrt the loss.
# First 0 the gradient buffers:
net.zero_grad()

# (Note we can access weights/biases and their gradients as attributes of the layers, e.g. nn.Conv2d etc)
print('\nConv1:')
conv1_weight = net.conv1.weight
conv1_bias = net.conv1.bias
print(conv1_weight.size(), conv1_bias.size())  # torch.Size([6, 1, 3, 3]) torch.Size([6])

# And can access the gradient attributes of each of these:
print(conv1_bias.grad) # tensor([0., 0., 0., 0., 0., 0.])  six 0's, as expected
print(conv1_weight.grad.size())  # torch.Size([6, 1, 3, 3])


# Now, as mentioned, we can do a backward pass from loss without specifying a vector argument (as loss is scalar):
loss.backward()

# And obtain updated gradients:
print(conv1_bias.grad) # tensor([-0.0022, -0.0035, -0.0002, -0.0018,  0.0000, -0.0010])

# We could then iterate over all params, get their gradients and do a weight update:
learning_rate = 0.01
for param in net.parameters():
    param.data.sub_(param.grad.data * learning_rate)   # sub_ subtracts in-place - i.e. w = w - grad * lr
    

# Although PyTorch has a package that with a variety of optimisers:
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

# We can now use this:
optimizer.zero_grad()
output = net(image_input)   # Forward Pass
loss = criterion(output, target)
loss.backward()    # Backward Pass
optimizer.step()    # Weight Update


# Note: we have to repeatedly clear out the gradients after each step using .zero_grad(), since otherwise
# they are accumulated (useful for minibatches - accumualte gradients across all items in the minibatch)