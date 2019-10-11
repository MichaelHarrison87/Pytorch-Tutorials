# Custom datasets, dataloaders and transformers
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

landmarks_frame = pd.read_csv('./data/faces/face_landmarks.csv')


# Inspect landmarks data:
n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print(f'Image Name: {img_name}')
print(f'Landmarks shape: {landmarks.shape}')
print(f'First 4 landmarks: {landmarks[:4]}') # landmarks is a list of (x,y) coords in the image's spatial dimensions

# Helper function to plot landmarks on the image
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.show()

# plt.figure()
# img = io.imread(os.path.join('data/faces/', img_name))
# show_landmarks(img, landmarks)


# torch.utils.data.Dataset is an abstract class representing a dataset. Custom datasets should inherit from
# it. Such inheritor concrete classes required .__len__() and .__getitem__() methods - to enable users to 
# get the size of the dataset, and to index into it, e.g. dataset[i], respectively.


# So we'll create a custom dataset for the faces landmarks data. The __init__() method reads the csv, 
# but does not get the image data - to avoid loading the entire dataset into memory upon initialisation.
# Getting the image is left to .__getitem__().

# Items in the dataset will be stored as dicts of image: landmark pairs.
# We also give it an optional transform argument, to allow for transformations.

class FaceLandmarksDataset(Dataset):
    """Face Landmarks Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with landmarks data
            root_dir (string): Directory containing the images
            transform (callable, optional): Optional transformation to apply to the sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    

# Now instantiate it with our data:
face_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces/')

fig = plt.figure()

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)
    
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title(f'Sample #{i}')
    ax.axis('off')
    # show_landmarks(**sample)
    
    if i == 3:
        # plt.show()
        break


# Now we'll create some useful Transformation classes - to let us, e.g., rescale all images to same size, 
# perform data augmentation.

# The dataset class above expects these as callables - and classes are preferable to functions so that 
# we don't need to pass arguments each time we call the transform

class Rescale(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        
        # If output size only an int, assume the desired output is square with that side length
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size  # when output_size is a tuple
        new_h, new_w = int(new_h), int(new_w)
        
        
        # Now resize
        img = transform.resize(image, (new_h, new_w))
        
        # For landmarks, height & width are swapped, since the x-axis is the width while y-axis the height:
        # The resizing below retains the landmarks' relative positions in the image
        landmarks = landmarks * [new_w/w, new_h/h]
        
        return {'image': img, 'landmarks': landmarks}
    

# Now instantiate it with our data:
rescale = Rescale((200, 100))
face_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces/', rescale)

fig = plt.figure()

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)
    
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title(f'Sample #{i}')
    ax.axis('off')
    # show_landmarks(**sample)
    
    if i == 3:
        # plt.show()
        break



class RandomCrop(object):
    """Randomly crops the image - used as a data augmentation tactic"""
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size    
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        
        new_h, new_w = self.output_size
        
        # Find a random position for the crop - specified by its top-left corner (the height & width of
        # the crop is then specified by output_size)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top: top + new_h, 
                      left: left + new_w]
        
        # The landmark positions just need to be shifted so that the new (0,0) is the top-left corner of the crop
        # Presumably some landmarks will be out-of-bounds of the new cropped image - but presumably these are just not plotted
        landmarks = landmarks - [left, top]
        
        return {'image': image, 'landmarks': landmarks}
    

random_crop = RandomCrop((100, 100))
face_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces/', random_crop)

fig = plt.figure()

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)
    
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title(f'Sample #{i}')
    ax.axis('off')
    # show_landmarks(**sample)
    
    if i == 3:
        # plt.show()
        break


class ToTensor(object):
    """Converts numpy ndarrays in the sample to Pytorch Tensors"""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        
        # Numpy stores images as: HxWxC
        # Pytorch as: CxHxW
        # So permute the numpy axes to get Pytorch's format
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 
                'landmarks': torch.from_numpy(landmarks)}
        

# Note we can use torchvision.transforms.Compose to compose transformations
compose = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])

transformed_dataset = FaceLandmarksDataset('data/faces/face_landmarks.csv', 'data/faces/', compose)


# Dataloaders can then be used to provide various operations on the dataset, e.g. batching, shuffling,
# loading the data in parallel using mutliprocessing workers. Note Dataloaders are iterators

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

# Helper function to show a batch of data
def show_landmarks_batch(batch):
    images_batch, landmarks_batch = batch['image'], batch['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose(1,2,0)) # CxHxW to HxWxC
    
    for i in range(batch_size):
        lm_x = landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size
        lm_y = landmarks_batch[i, :, 1].numpy() + grid_border_size
        plt.scatter(lm_x, lm_y, s=10, marker='.', c='r')
        plt.title('Batch from Dataloader')
        

# Now plot batches from the dataloader:
for i, sample in enumerate(dataloader):
    print(i, sample['image'].size(), sample['landmarks'].size())
    
    # plot 4th batch then break
    if i == 3:
        plt.figure()
        show_landmarks_batch(sample)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break