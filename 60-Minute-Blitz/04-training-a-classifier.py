# 04 - Training a Classifier
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


# The torchvision package has data loaders for various common image datasets, e.g. MNIST, CIFAR, ImageNet

# The CIFAR-10 dataset contains 3x32x32 images of 10 different categories (e.g. airplane, cat, bird)
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 10

# torchvision datasets are PILImage images, with pixel values in range [0,1], which we want to normalise to [-1,1]
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
# The args of Normalize() are mean & sd, we specify mean & sd 0.5 for each of the 3 colour channels of the original image

# Import Data
trainset = torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10('./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Functions to display an image
def imshow(img):
    # Undo the normalisation:
    img = 0.5 + img/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# Now get some images and show them; note below code shows 4 images since trainload batched the data into batches of 4
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
# imshow(torchvision.utils.make_grid(images))


# Copying the neural network from last chapter (modified to accept 3-channel input images):
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
        self.conv1 = nn.Conv2d(3, 6, 3)
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

# And set up the loss function and optimiser - cross-entropy loss and SGD with momentum:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Now train the network, looping over the data:
print(f'trainset size: {len(trainset)}, num batches: {len(trainset)/BATCH_SIZE}')
print(f'testset size: {len(testset)}, num batches: {len(trainset)/BATCH_SIZE}')
for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data   # data is a minibatch
        
        # zero the gradient buffers
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = net(inputs)
        
        # Loss
        loss = criterion(outputs, labels)
        
        # Backward Pass
        loss.backward()
        
        # Weight Update
        optimizer.step()
        
        # Print statistics:
        running_loss += loss.item()
        if i % 500 == 499:
            print(f'Epoch {epoch+1:d}, Batch {i+1:4d} - loss: {running_loss/500}')

print('Training Finished')


# Now save the model:
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# Test the network on test data
testiter = iter(testloader)
images, labels = next(testiter)

net = Net()
net.load_state_dict(torch.load(PATH))
output = net(images)

_, predicted = torch.max(outputs, 0)

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
print('Predicted  : ', ' '.join('%5s' % classes[predicted[j]] for j in range(BATCH_SIZE)))
imshow(torchvision.utils.make_grid(images))


# Can get accuracy (by class) across entire test set
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Aggregate
        total += labels.size(0)     # number of items in the minibatch
        correct += (predicted == labels).sum().item()
        
        # by Class
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        
        
print(f'Accuracy across entire test set is: {correct/total:.1%}')

print('\nAccuracy by Class:')
for i in range(10):
    print(f'{classes[i]}: {class_total[i]} {class_correct[i]/class_total[i]:.1%}')