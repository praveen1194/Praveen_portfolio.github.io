---
layout: project
title: CNN Image classifier
projectname: Image Classifier
description: Trained Convolution Neural Network (CNN) to classify images into categories
started: 2020
status: completed
category: project
techstack: Github, Python, Numpy, Pandas, Matplotlib, Pytorch, Sklearn, OpenCV2
---

# Image classification using traditional Convolution Neural Networks (CNNs)

## Objectives:
- Train an AI model to classify images.
- Calculate its accuracy

## Outcomes:
Multiple models were trained with different parameters. I selected the parameters manually and tried to increase the possibility of reaching global minima. The final model achieved 51% accuracy. Below are the details:
- Batch size: for Train and validation data 1 0
- Accuracy: 51%
- Dropout: 0.1
- No of Epochs: 10

## Implementation:


```python
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from  torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
from skimage import io, transform

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import csv
import os
import math
import cv2
```

### Training data

I used an open source dataset to train the model.


```python
! git clone https://github.com/MohammedAlghamdi/imagenet10.git
```

    Cloning into 'imagenet10'...
    remote: Enumerating objects: 10019, done.[K
    remote: Total 10019 (delta 0), reused 0 (delta 0), pack-reused 10019 (from 1)[K
    Receiving objects: 100% (10019/10019), 966.71 MiB | 16.91 MiB/s, done.
    Resolving deltas: 100% (2/2), done.
    Updating files: 100% (10002/10002), done.
    

Check that the repository is there:


```python
%ls
```

    imagenet10/  sample_data/
    


```python
root_dir = "imagenet10/train_set/"
class_names = [
  "baboon",
  "banana",
  "canoe",
  "cat",
  "desk",
  "drill",
  "dumbbell",
  "football",
  "mug",
  "orange",
]
```

A helper function for reading in images and assigning labels.


```python
def get_meta(root_dir, dirs):
    """ Fetches the meta data for all the images and assigns labels.
    """
    paths, classes = [], []
    for i, dir_ in enumerate(dirs):
        for entry in os.scandir(root_dir + dir_):
            if (entry.is_file()):
                paths.append(entry.path)
                classes.append(i)

    return paths, classes
```

Now we create a dataframe using all the data.


```python
# Benign images we will assign class 0, and malignant as 1
paths, classes = get_meta(root_dir, class_names)

data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
```

View some sample data.


```python
print("Found", len(data_df), "images.")
data_df.head()
```

    Found 9000 images.
    





  <div id="df-50b5ca1f-16a0-4190-80eb-0183b40d8d53" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>imagenet10/train_set/desk/n03179701_7724.JPEG</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>imagenet10/train_set/drill/n03239726_13670.JPEG</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>imagenet10/train_set/drill/n03239726_2072.JPEG</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>imagenet10/train_set/dumbbell/n03255030_6588.JPEG</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>imagenet10/train_set/mug/n03797390_23464.JPEG</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-50b5ca1f-16a0-4190-80eb-0183b40d8d53')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-50b5ca1f-16a0-4190-80eb-0183b40d8d53 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-50b5ca1f-16a0-4190-80eb-0183b40d8d53');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




Now we will create the Dataset class.


```python
class ImageNet10(Dataset):
    """ ImageNet10 dataset class """

    def __init__(self, df, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images
            df (DataFrame object): Dataframe containing the images, paths and classes
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load image from path and get label
        x = Image.open(self.df['path'][index])
        try:
          x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
          pass
        y = torch.tensor(int(self.df['class'][index]))

        if self.transform:
            x = self.transform(x)

        return x, y
```

Compute what we should normalise the dataset to.


```python
def compute_img_mean_std(image_paths):
    """
        Computing the mean and std of three channel on the whole dataset,
        first we should normalise the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs

```


```python
norm_mean, norm_std = compute_img_mean_std(paths)
```

    100%|██████████| 9000/9000 [00:21<00:00, 414.94it/s]
    

    (224, 224, 3, 9000)
    normMean = [np.float32(0.52283657), np.float32(0.47988173), np.float32(0.40605167)]
    normStd = [np.float32(0.29770696), np.float32(0.28884032), np.float32(0.31178236)]
    

Now we create the transforms to normalise and turn our data into tensors.


```python
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
```

Now I split the data into train and test sets and instantiate the new Dataset objects.


```python
train_split = 0.70 # Defines the ratio of train/valid/test data.
valid_split = 0.10

train_size = int(len(data_df)*train_split)
valid_size = int(len(data_df)*valid_split)

ins_dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

ins_dataset_valid = ImageNet10(
    df=data_df[train_size:(train_size + valid_size)].reset_index(drop=True),
    transform=data_transform,
)

ins_dataset_test = ImageNet10(
    df=data_df[(train_size + valid_size):].reset_index(drop=True),
    transform=data_transform,
)
```

Now to create DataLoaders for the datasets.


```python
train_set_loader = torch.utils.data.DataLoader(
    ins_dataset_train,
    batch_size=10,
    shuffle=True,
    num_workers=2
)

test_set_loader = torch.utils.data.DataLoader(
    ins_dataset_test,
    batch_size=24,
    shuffle=False,
    num_workers=2
)

validation_set_loader = torch.utils.data.DataLoader(
    ins_dataset_valid,
    batch_size=10,
    shuffle=True,
    num_workers=2
)

classes = np.arange(0, 10)
```

Defining the framework for the ConvNet model:


```python
# Convolutional neural network
class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
      super(ConvNet, self).__init__()

        # Add network layers here
      self.conv1 = nn.Sequential(
          nn.Conv2d(3,16,3),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.1)
        )

      self.conv2 = nn.Sequential(
          nn.Conv2d(16,24,4),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.1)
        )

      self.conv3 = nn.Sequential(
          nn.Conv2d(24,32,4),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.1)
      )

      self.conv4 = nn.Sequential(
          nn.Conv2d(32,128,4),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.1)
      )

      self.conv5 = nn.Sequential(
          nn.Conv2d(128,128,4),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout2d(p=0.1)
      )

      self.fc2 = nn.Sequential(
          nn.Linear(21632,512),
          nn.Linear(512, num_classes)
      )

    def forward(self, x):
      out = self.conv1(x)
      out = self.conv2(out)
      out = self.conv3(out)
      out = self.conv4(out)
      #out = self.conv5(out)
      out = out.view(out.size(0),-1)
      out = self.fc2(out)

      # Complete the graph

      return out
```


```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_gpu = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()

# Stochastic gradient descent
optimizer = optim.SGD(model_gpu.parameters(), lr=0.001, momentum=0.9)
```


```python
def validate_and_plot(tensor, channel, num_cols=6):

  from matplotlib import pyplot as plt

  if not tensor.ndim==4:
    raise Exception("assumes a 4D tensor")
  if not tensor.shape[-1]==3:
    raise Exception("last dim needs to be 3 to plot")
  num_kernels = tensor.shape[0]
  num_rows = 1+ num_kernels // num_cols
  fig = plt.figure(figsize=(num_cols,num_rows))

  for i in range(tensor.shape[0]):
    ax1 = fig.add_subplot(num_rows,num_cols,i+1)
    ax1.imshow(tensor[i][channel], cmap="gray")
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  plt.show()
```


```python
def vis_filters(channel):
  import torch
  import torchvision.models as models

  mm = model_gpu.cpu()
  filters = mm.modules
  body_model = [i for i in mm.children()][0]
  print(mm.children())
  layer1 = body_model[0]
  tensor = layer1.weight.data.numpy()
  # print(tensor)
  validate_and_plot(tensor, channel)
```

### Plotting untrained model layers (filters) in red, blue and green channels:
These filters, once trined would internally be used to extract features in all three channels, the results would later be combined together.


```python
def visualize():

  red_channel = 0
  blue_channel = 1
  green_channel = 2
  vis_filters(red_channel)                          #visualize red channel filters
  vis_filters(blue_channel)                         #visualize blue channel filters
  vis_filters(green_channel)                        #visualize green channel filters


print('Visualize filters before training:')
visualize()
```

    Visualize filters before training:
    <generator object Module.children at 0x7e9eadd0db10>
    


    
![png]({{ site.baseurl }}/assets/img/output_31.png)
    


    <generator object Module.children at 0x7e9ead673c60>
    


    
![png]({{ site.baseurl }}/assets/img/output_31_3.png)
    


    <generator object Module.children at 0x7e9eaddd82b0>
    


    
![png]({{ site.baseurl }}/assets/img/output_31_5.png)
    



```python
def train_model_epochs(num_epochs):
    """ Trains the model for a given number of epochs on the training set. """
    for epoch in range(num_epochs):

        # Visualising filters
        if epoch == 5:
          print('Visualize filters during training:')
          visualize()
          model_gpu.to(device)
        running_loss = 0.0

        # training dataset
        for i, data in enumerate(train_set_loader, 0):
            images, labels = data

             # Explicitly specifies that data is to be copied onto the device!
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients means to reset them from
            # any previous values. By default, gradients accumulate!
            optimizer.zero_grad()

            # Passing inputs to the model calls the forward() function of
            # the Module class, and the outputs value contains the return value
            # of forward()
            outputs = model_gpu(images)

            # Compute the loss based on the true labels
            loss = criterion(outputs, labels)

            # Backpropagate the error with respect to the loss
            loss.backward()

            # Updates the parameters based on current gradients and update rule;
            # in this case, defined by SGD()
            optimizer.step()

            # Print our loss
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Epoch / Batch [%d / %d] - Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        # Validation dataset
        for i, data in enumerate(validation_set_loader, 0):
            images, labels = data

             # Explicitly specifies that data is to be copied onto the device!
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients means to reset them from
            # any previous values. By default, gradients accumulate!
            optimizer.zero_grad()

            # Passing inputs to the model calls the forward() function of
            # the Module class, and the outputs value contains the return value
            # of forward()
            outputs = model_gpu(images)

            # Compute the loss based on the true labels
            loss = criterion(outputs, labels)

            # Print our loss
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('Epoch / Validation Batch [%d / %d] - Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
```

## Start training:


```python
model_gpu = model_gpu.to(device)
import timeit
gpu_train_time = timeit.timeit(
    "train_model_epochs(num_epochs)",
    setup="num_epochs=10",
    number=1,
    globals=globals(),
)
```

    Epoch / Batch [1 / 10] - Loss: 0.023
    Epoch / Batch [1 / 20] - Loss: 0.023
    Epoch / Batch [1 / 30] - Loss: 0.023
    Epoch / Batch [1 / 40] - Loss: 0.023
    Epoch / Batch [1 / 50] - Loss: 0.023
    Epoch / Batch [1 / 60] - Loss: 0.023
    Epoch / Batch [1 / 70] - Loss: 0.023
    Epoch / Batch [1 / 80] - Loss: 0.023
    Epoch / Batch [1 / 90] - Loss: 0.023
    Epoch / Batch [1 / 100] - Loss: 0.023
    Epoch / Batch [1 / 110] - Loss: 0.023
    Epoch / Batch [1 / 120] - Loss: 0.023
    Epoch / Batch [1 / 130] - Loss: 0.023
    Epoch / Batch [1 / 140] - Loss: 0.023
    Epoch / Batch [1 / 150] - Loss: 0.023
    Epoch / Batch [1 / 160] - Loss: 0.023
    Epoch / Batch [1 / 170] - Loss: 0.023
    Epoch / Batch [1 / 180] - Loss: 0.023
    Epoch / Batch [1 / 190] - Loss: 0.022
    Epoch / Batch [1 / 200] - Loss: 0.022
    Epoch / Batch [1 / 210] - Loss: 0.022
    Epoch / Batch [1 / 220] - Loss: 0.022
    Epoch / Batch [1 / 230] - Loss: 0.022
    Epoch / Batch [1 / 240] - Loss: 0.021
    Epoch / Batch [1 / 250] - Loss: 0.022
    Epoch / Batch [1 / 260] - Loss: 0.020
    Epoch / Batch [1 / 270] - Loss: 0.020
    Epoch / Batch [1 / 280] - Loss: 0.021
    Epoch / Batch [1 / 290] - Loss: 0.021
    Epoch / Batch [1 / 300] - Loss: 0.020
    Epoch / Batch [1 / 310] - Loss: 0.021
    Epoch / Batch [1 / 320] - Loss: 0.021
    Epoch / Batch [1 / 330] - Loss: 0.020
    Epoch / Batch [1 / 340] - Loss: 0.021
    Epoch / Batch [1 / 350] - Loss: 0.021
    Epoch / Batch [1 / 360] - Loss: 0.021
    Epoch / Batch [1 / 370] - Loss: 0.021
    Epoch / Batch [1 / 380] - Loss: 0.019
    Epoch / Batch [1 / 390] - Loss: 0.019
    Epoch / Batch [1 / 400] - Loss: 0.020
    Epoch / Batch [1 / 410] - Loss: 0.020
    Epoch / Batch [1 / 420] - Loss: 0.019
    Epoch / Batch [1 / 430] - Loss: 0.021
    Epoch / Batch [1 / 440] - Loss: 0.021
    Epoch / Batch [1 / 450] - Loss: 0.020
    Epoch / Batch [1 / 460] - Loss: 0.020
    Epoch / Batch [1 / 470] - Loss: 0.019
    Epoch / Batch [1 / 480] - Loss: 0.020
    Epoch / Batch [1 / 490] - Loss: 0.021
    Epoch / Batch [1 / 500] - Loss: 0.020
    Epoch / Batch [1 / 510] - Loss: 0.021
    Epoch / Batch [1 / 520] - Loss: 0.019
    Epoch / Batch [1 / 530] - Loss: 0.019
    Epoch / Batch [1 / 540] - Loss: 0.019
    Epoch / Batch [1 / 550] - Loss: 0.020
    Epoch / Batch [1 / 560] - Loss: 0.020
    Epoch / Batch [1 / 570] - Loss: 0.020
    Epoch / Batch [1 / 580] - Loss: 0.021
    Epoch / Batch [1 / 590] - Loss: 0.020
    Epoch / Batch [1 / 600] - Loss: 0.020
    Epoch / Batch [1 / 610] - Loss: 0.020
    Epoch / Batch [1 / 620] - Loss: 0.021
    Epoch / Batch [1 / 630] - Loss: 0.019
    Epoch / Validation Batch [1 / 10] - Loss: 0.019
    Epoch / Validation Batch [1 / 20] - Loss: 0.019
    Epoch / Validation Batch [1 / 30] - Loss: 0.019
    Epoch / Validation Batch [1 / 40] - Loss: 0.021
    Epoch / Validation Batch [1 / 50] - Loss: 0.019
    Epoch / Validation Batch [1 / 60] - Loss: 0.020
    Epoch / Validation Batch [1 / 70] - Loss: 0.018
    Epoch / Validation Batch [1 / 80] - Loss: 0.019
    Epoch / Validation Batch [1 / 90] - Loss: 0.021
    Epoch / Batch [2 / 10] - Loss: 0.019
    Epoch / Batch [2 / 20] - Loss: 0.020
    Epoch / Batch [2 / 30] - Loss: 0.020
    Epoch / Batch [2 / 40] - Loss: 0.022
    Epoch / Batch [2 / 50] - Loss: 0.020
    Epoch / Batch [2 / 60] - Loss: 0.019
    Epoch / Batch [2 / 70] - Loss: 0.020
    Epoch / Batch [2 / 80] - Loss: 0.020
    Epoch / Batch [2 / 90] - Loss: 0.019
    Epoch / Batch [2 / 100] - Loss: 0.018
    Epoch / Batch [2 / 110] - Loss: 0.019
    Epoch / Batch [2 / 120] - Loss: 0.018
    Epoch / Batch [2 / 130] - Loss: 0.019
    Epoch / Batch [2 / 140] - Loss: 0.018
    Epoch / Batch [2 / 150] - Loss: 0.020
    Epoch / Batch [2 / 160] - Loss: 0.018
    Epoch / Batch [2 / 170] - Loss: 0.019
    Epoch / Batch [2 / 180] - Loss: 0.019
    Epoch / Batch [2 / 190] - Loss: 0.019
    Epoch / Batch [2 / 200] - Loss: 0.019
    Epoch / Batch [2 / 210] - Loss: 0.019
    Epoch / Batch [2 / 220] - Loss: 0.019
    Epoch / Batch [2 / 230] - Loss: 0.019
    Epoch / Batch [2 / 240] - Loss: 0.018
    Epoch / Batch [2 / 250] - Loss: 0.019
    Epoch / Batch [2 / 260] - Loss: 0.019
    Epoch / Batch [2 / 270] - Loss: 0.019
    Epoch / Batch [2 / 280] - Loss: 0.017
    Epoch / Batch [2 / 290] - Loss: 0.018
    Epoch / Batch [2 / 300] - Loss: 0.019
    Epoch / Batch [2 / 310] - Loss: 0.019
    Epoch / Batch [2 / 320] - Loss: 0.019
    Epoch / Batch [2 / 330] - Loss: 0.019
    Epoch / Batch [2 / 340] - Loss: 0.019
    Epoch / Batch [2 / 350] - Loss: 0.018
    Epoch / Batch [2 / 360] - Loss: 0.020
    Epoch / Batch [2 / 370] - Loss: 0.019
    Epoch / Batch [2 / 380] - Loss: 0.018
    Epoch / Batch [2 / 390] - Loss: 0.018
    Epoch / Batch [2 / 400] - Loss: 0.018
    Epoch / Batch [2 / 410] - Loss: 0.019
    Epoch / Batch [2 / 420] - Loss: 0.019
    Epoch / Batch [2 / 430] - Loss: 0.018
    Epoch / Batch [2 / 440] - Loss: 0.019
    Epoch / Batch [2 / 450] - Loss: 0.019
    Epoch / Batch [2 / 460] - Loss: 0.019
    Epoch / Batch [2 / 470] - Loss: 0.017
    Epoch / Batch [2 / 480] - Loss: 0.018
    Epoch / Batch [2 / 490] - Loss: 0.018
    Epoch / Batch [2 / 500] - Loss: 0.018
    Epoch / Batch [2 / 510] - Loss: 0.018
    Epoch / Batch [2 / 520] - Loss: 0.018
    Epoch / Batch [2 / 530] - Loss: 0.018
    Epoch / Batch [2 / 540] - Loss: 0.019
    Epoch / Batch [2 / 550] - Loss: 0.018
    Epoch / Batch [2 / 560] - Loss: 0.018
    Epoch / Batch [2 / 570] - Loss: 0.021
    Epoch / Batch [2 / 580] - Loss: 0.018
    Epoch / Batch [2 / 590] - Loss: 0.017
    Epoch / Batch [2 / 600] - Loss: 0.017
    Epoch / Batch [2 / 610] - Loss: 0.018
    Epoch / Batch [2 / 620] - Loss: 0.019
    Epoch / Batch [2 / 630] - Loss: 0.018
    Epoch / Validation Batch [2 / 10] - Loss: 0.018
    Epoch / Validation Batch [2 / 20] - Loss: 0.017
    Epoch / Validation Batch [2 / 30] - Loss: 0.018
    Epoch / Validation Batch [2 / 40] - Loss: 0.018
    Epoch / Validation Batch [2 / 50] - Loss: 0.019
    Epoch / Validation Batch [2 / 60] - Loss: 0.018
    Epoch / Validation Batch [2 / 70] - Loss: 0.018
    Epoch / Validation Batch [2 / 80] - Loss: 0.018
    Epoch / Validation Batch [2 / 90] - Loss: 0.019
    Epoch / Batch [3 / 10] - Loss: 0.018
    Epoch / Batch [3 / 20] - Loss: 0.018
    Epoch / Batch [3 / 30] - Loss: 0.020
    Epoch / Batch [3 / 40] - Loss: 0.020
    Epoch / Batch [3 / 50] - Loss: 0.019
    Epoch / Batch [3 / 60] - Loss: 0.017
    Epoch / Batch [3 / 70] - Loss: 0.018
    Epoch / Batch [3 / 80] - Loss: 0.019
    Epoch / Batch [3 / 90] - Loss: 0.018
    Epoch / Batch [3 / 100] - Loss: 0.020
    Epoch / Batch [3 / 110] - Loss: 0.018
    Epoch / Batch [3 / 120] - Loss: 0.019
    Epoch / Batch [3 / 130] - Loss: 0.018
    Epoch / Batch [3 / 140] - Loss: 0.019
    Epoch / Batch [3 / 150] - Loss: 0.017
    Epoch / Batch [3 / 160] - Loss: 0.019
    Epoch / Batch [3 / 170] - Loss: 0.016
    Epoch / Batch [3 / 180] - Loss: 0.017
    Epoch / Batch [3 / 190] - Loss: 0.018
    Epoch / Batch [3 / 200] - Loss: 0.016
    Epoch / Batch [3 / 210] - Loss: 0.017
    Epoch / Batch [3 / 220] - Loss: 0.016
    Epoch / Batch [3 / 230] - Loss: 0.018
    Epoch / Batch [3 / 240] - Loss: 0.017
    Epoch / Batch [3 / 250] - Loss: 0.017
    Epoch / Batch [3 / 260] - Loss: 0.017
    Epoch / Batch [3 / 270] - Loss: 0.019
    Epoch / Batch [3 / 280] - Loss: 0.018
    Epoch / Batch [3 / 290] - Loss: 0.016
    Epoch / Batch [3 / 300] - Loss: 0.015
    Epoch / Batch [3 / 310] - Loss: 0.020
    Epoch / Batch [3 / 320] - Loss: 0.017
    Epoch / Batch [3 / 330] - Loss: 0.018
    Epoch / Batch [3 / 340] - Loss: 0.020
    Epoch / Batch [3 / 350] - Loss: 0.016
    Epoch / Batch [3 / 360] - Loss: 0.018
    Epoch / Batch [3 / 370] - Loss: 0.017
    Epoch / Batch [3 / 380] - Loss: 0.019
    Epoch / Batch [3 / 390] - Loss: 0.018
    Epoch / Batch [3 / 400] - Loss: 0.019
    Epoch / Batch [3 / 410] - Loss: 0.019
    Epoch / Batch [3 / 420] - Loss: 0.019
    Epoch / Batch [3 / 430] - Loss: 0.018
    Epoch / Batch [3 / 440] - Loss: 0.017
    Epoch / Batch [3 / 450] - Loss: 0.016
    Epoch / Batch [3 / 460] - Loss: 0.017
    Epoch / Batch [3 / 470] - Loss: 0.016
    Epoch / Batch [3 / 480] - Loss: 0.017
    Epoch / Batch [3 / 490] - Loss: 0.016
    Epoch / Batch [3 / 500] - Loss: 0.018
    Epoch / Batch [3 / 510] - Loss: 0.018
    Epoch / Batch [3 / 520] - Loss: 0.017
    Epoch / Batch [3 / 530] - Loss: 0.017
    Epoch / Batch [3 / 540] - Loss: 0.017
    Epoch / Batch [3 / 550] - Loss: 0.018
    Epoch / Batch [3 / 560] - Loss: 0.017
    Epoch / Batch [3 / 570] - Loss: 0.017
    Epoch / Batch [3 / 580] - Loss: 0.017
    Epoch / Batch [3 / 590] - Loss: 0.017
    Epoch / Batch [3 / 600] - Loss: 0.019
    Epoch / Batch [3 / 610] - Loss: 0.017
    Epoch / Batch [3 / 620] - Loss: 0.017
    Epoch / Batch [3 / 630] - Loss: 0.018
    Epoch / Validation Batch [3 / 10] - Loss: 0.018
    Epoch / Validation Batch [3 / 20] - Loss: 0.018
    Epoch / Validation Batch [3 / 30] - Loss: 0.017
    Epoch / Validation Batch [3 / 40] - Loss: 0.019
    Epoch / Validation Batch [3 / 50] - Loss: 0.019
    Epoch / Validation Batch [3 / 60] - Loss: 0.017
    Epoch / Validation Batch [3 / 70] - Loss: 0.018
    Epoch / Validation Batch [3 / 80] - Loss: 0.018
    Epoch / Validation Batch [3 / 90] - Loss: 0.018
    Epoch / Batch [4 / 10] - Loss: 0.017
    Epoch / Batch [4 / 20] - Loss: 0.017
    Epoch / Batch [4 / 30] - Loss: 0.017
    Epoch / Batch [4 / 40] - Loss: 0.018
    Epoch / Batch [4 / 50] - Loss: 0.018
    Epoch / Batch [4 / 60] - Loss: 0.018
    Epoch / Batch [4 / 70] - Loss: 0.017
    Epoch / Batch [4 / 80] - Loss: 0.016
    Epoch / Batch [4 / 90] - Loss: 0.019
    Epoch / Batch [4 / 100] - Loss: 0.018
    Epoch / Batch [4 / 110] - Loss: 0.018
    Epoch / Batch [4 / 120] - Loss: 0.017
    Epoch / Batch [4 / 130] - Loss: 0.016
    Epoch / Batch [4 / 140] - Loss: 0.017
    Epoch / Batch [4 / 150] - Loss: 0.017
    Epoch / Batch [4 / 160] - Loss: 0.016
    Epoch / Batch [4 / 170] - Loss: 0.017
    Epoch / Batch [4 / 180] - Loss: 0.018
    Epoch / Batch [4 / 190] - Loss: 0.017
    Epoch / Batch [4 / 200] - Loss: 0.017
    Epoch / Batch [4 / 210] - Loss: 0.017
    Epoch / Batch [4 / 220] - Loss: 0.015
    Epoch / Batch [4 / 230] - Loss: 0.017
    Epoch / Batch [4 / 240] - Loss: 0.016
    Epoch / Batch [4 / 250] - Loss: 0.014
    Epoch / Batch [4 / 260] - Loss: 0.017
    Epoch / Batch [4 / 270] - Loss: 0.015
    Epoch / Batch [4 / 280] - Loss: 0.017
    Epoch / Batch [4 / 290] - Loss: 0.018
    Epoch / Batch [4 / 300] - Loss: 0.018
    Epoch / Batch [4 / 310] - Loss: 0.018
    Epoch / Batch [4 / 320] - Loss: 0.018
    Epoch / Batch [4 / 330] - Loss: 0.017
    Epoch / Batch [4 / 340] - Loss: 0.016
    Epoch / Batch [4 / 350] - Loss: 0.016
    Epoch / Batch [4 / 360] - Loss: 0.014
    Epoch / Batch [4 / 370] - Loss: 0.017
    Epoch / Batch [4 / 380] - Loss: 0.017
    Epoch / Batch [4 / 390] - Loss: 0.017
    Epoch / Batch [4 / 400] - Loss: 0.016
    Epoch / Batch [4 / 410] - Loss: 0.016
    Epoch / Batch [4 / 420] - Loss: 0.017
    Epoch / Batch [4 / 430] - Loss: 0.017
    Epoch / Batch [4 / 440] - Loss: 0.016
    Epoch / Batch [4 / 450] - Loss: 0.017
    Epoch / Batch [4 / 460] - Loss: 0.016
    Epoch / Batch [4 / 470] - Loss: 0.017
    Epoch / Batch [4 / 480] - Loss: 0.016
    Epoch / Batch [4 / 490] - Loss: 0.016
    Epoch / Batch [4 / 500] - Loss: 0.017
    Epoch / Batch [4 / 510] - Loss: 0.017
    Epoch / Batch [4 / 520] - Loss: 0.017
    Epoch / Batch [4 / 530] - Loss: 0.015
    Epoch / Batch [4 / 540] - Loss: 0.016
    Epoch / Batch [4 / 550] - Loss: 0.018
    Epoch / Batch [4 / 560] - Loss: 0.017
    Epoch / Batch [4 / 570] - Loss: 0.015
    Epoch / Batch [4 / 580] - Loss: 0.018
    Epoch / Batch [4 / 590] - Loss: 0.018
    Epoch / Batch [4 / 600] - Loss: 0.017
    Epoch / Batch [4 / 610] - Loss: 0.017
    Epoch / Batch [4 / 620] - Loss: 0.016
    Epoch / Batch [4 / 630] - Loss: 0.017
    Epoch / Validation Batch [4 / 10] - Loss: 0.017
    Epoch / Validation Batch [4 / 20] - Loss: 0.017
    Epoch / Validation Batch [4 / 30] - Loss: 0.019
    Epoch / Validation Batch [4 / 40] - Loss: 0.016
    Epoch / Validation Batch [4 / 50] - Loss: 0.016
    Epoch / Validation Batch [4 / 60] - Loss: 0.019
    Epoch / Validation Batch [4 / 70] - Loss: 0.017
    Epoch / Validation Batch [4 / 80] - Loss: 0.017
    Epoch / Validation Batch [4 / 90] - Loss: 0.016
    Epoch / Batch [5 / 10] - Loss: 0.016
    Epoch / Batch [5 / 20] - Loss: 0.015
    Epoch / Batch [5 / 30] - Loss: 0.015
    Epoch / Batch [5 / 40] - Loss: 0.015
    Epoch / Batch [5 / 50] - Loss: 0.017
    Epoch / Batch [5 / 60] - Loss: 0.015
    Epoch / Batch [5 / 70] - Loss: 0.016
    Epoch / Batch [5 / 80] - Loss: 0.017
    Epoch / Batch [5 / 90] - Loss: 0.016
    Epoch / Batch [5 / 100] - Loss: 0.015
    Epoch / Batch [5 / 110] - Loss: 0.016
    Epoch / Batch [5 / 120] - Loss: 0.016
    Epoch / Batch [5 / 130] - Loss: 0.017
    Epoch / Batch [5 / 140] - Loss: 0.017
    Epoch / Batch [5 / 150] - Loss: 0.015
    Epoch / Batch [5 / 160] - Loss: 0.016
    Epoch / Batch [5 / 170] - Loss: 0.017
    Epoch / Batch [5 / 180] - Loss: 0.016
    Epoch / Batch [5 / 190] - Loss: 0.016
    Epoch / Batch [5 / 200] - Loss: 0.017
    Epoch / Batch [5 / 210] - Loss: 0.017
    Epoch / Batch [5 / 220] - Loss: 0.015
    Epoch / Batch [5 / 230] - Loss: 0.015
    Epoch / Batch [5 / 240] - Loss: 0.017
    Epoch / Batch [5 / 250] - Loss: 0.015
    Epoch / Batch [5 / 260] - Loss: 0.017
    Epoch / Batch [5 / 270] - Loss: 0.018
    Epoch / Batch [5 / 280] - Loss: 0.014
    Epoch / Batch [5 / 290] - Loss: 0.017
    Epoch / Batch [5 / 300] - Loss: 0.016
    Epoch / Batch [5 / 310] - Loss: 0.016
    Epoch / Batch [5 / 320] - Loss: 0.015
    Epoch / Batch [5 / 330] - Loss: 0.015
    Epoch / Batch [5 / 340] - Loss: 0.017
    Epoch / Batch [5 / 350] - Loss: 0.016
    Epoch / Batch [5 / 360] - Loss: 0.015
    Epoch / Batch [5 / 370] - Loss: 0.015
    Epoch / Batch [5 / 380] - Loss: 0.015
    Epoch / Batch [5 / 390] - Loss: 0.017
    Epoch / Batch [5 / 400] - Loss: 0.016
    Epoch / Batch [5 / 410] - Loss: 0.015
    Epoch / Batch [5 / 420] - Loss: 0.015
    Epoch / Batch [5 / 430] - Loss: 0.016
    Epoch / Batch [5 / 440] - Loss: 0.015
    Epoch / Batch [5 / 450] - Loss: 0.015
    Epoch / Batch [5 / 460] - Loss: 0.015
    Epoch / Batch [5 / 470] - Loss: 0.015
    Epoch / Batch [5 / 480] - Loss: 0.015
    Epoch / Batch [5 / 490] - Loss: 0.015
    Epoch / Batch [5 / 500] - Loss: 0.015
    Epoch / Batch [5 / 510] - Loss: 0.017
    Epoch / Batch [5 / 520] - Loss: 0.017
    Epoch / Batch [5 / 530] - Loss: 0.014
    Epoch / Batch [5 / 540] - Loss: 0.014
    Epoch / Batch [5 / 550] - Loss: 0.015
    Epoch / Batch [5 / 560] - Loss: 0.017
    Epoch / Batch [5 / 570] - Loss: 0.014
    Epoch / Batch [5 / 580] - Loss: 0.018
    Epoch / Batch [5 / 590] - Loss: 0.017
    Epoch / Batch [5 / 600] - Loss: 0.013
    Epoch / Batch [5 / 610] - Loss: 0.015
    Epoch / Batch [5 / 620] - Loss: 0.013
    Epoch / Batch [5 / 630] - Loss: 0.016
    Epoch / Validation Batch [5 / 10] - Loss: 0.017
    Epoch / Validation Batch [5 / 20] - Loss: 0.017
    Epoch / Validation Batch [5 / 30] - Loss: 0.015
    Epoch / Validation Batch [5 / 40] - Loss: 0.015
    Epoch / Validation Batch [5 / 50] - Loss: 0.015
    Epoch / Validation Batch [5 / 60] - Loss: 0.015
    Epoch / Validation Batch [5 / 70] - Loss: 0.015
    Epoch / Validation Batch [5 / 80] - Loss: 0.016
    Epoch / Validation Batch [5 / 90] - Loss: 0.015
    Visualize filters during training:
    <generator object Module.children at 0x7e9eadd0c450>
    


    
![png]({{ site.baseurl }}/assets/img/output_34_1.png)
    


    <generator object Module.children at 0x7e9fc116bd30>
    


    
![png]({{ site.baseurl }}/assets/img/output_34_3.png)
    


    <generator object Module.children at 0x7e9fc0fab440>
    


    
![png]({{ site.baseurl }}/assets/img/output_34_5.png)
    


    Epoch / Batch [6 / 10] - Loss: 0.013
    Epoch / Batch [6 / 20] - Loss: 0.013
    Epoch / Batch [6 / 30] - Loss: 0.015
    Epoch / Batch [6 / 40] - Loss: 0.015
    Epoch / Batch [6 / 50] - Loss: 0.015
    Epoch / Batch [6 / 60] - Loss: 0.017
    Epoch / Batch [6 / 70] - Loss: 0.016
    Epoch / Batch [6 / 80] - Loss: 0.014
    Epoch / Batch [6 / 90] - Loss: 0.015
    Epoch / Batch [6 / 100] - Loss: 0.014
    Epoch / Batch [6 / 110] - Loss: 0.014
    Epoch / Batch [6 / 120] - Loss: 0.013
    Epoch / Batch [6 / 130] - Loss: 0.014
    Epoch / Batch [6 / 140] - Loss: 0.016
    Epoch / Batch [6 / 150] - Loss: 0.016
    Epoch / Batch [6 / 160] - Loss: 0.013
    Epoch / Batch [6 / 170] - Loss: 0.016
    Epoch / Batch [6 / 180] - Loss: 0.015
    Epoch / Batch [6 / 190] - Loss: 0.014
    Epoch / Batch [6 / 200] - Loss: 0.013
    Epoch / Batch [6 / 210] - Loss: 0.015
    Epoch / Batch [6 / 220] - Loss: 0.014
    Epoch / Batch [6 / 230] - Loss: 0.017
    Epoch / Batch [6 / 240] - Loss: 0.014
    Epoch / Batch [6 / 250] - Loss: 0.015
    Epoch / Batch [6 / 260] - Loss: 0.014
    Epoch / Batch [6 / 270] - Loss: 0.014
    Epoch / Batch [6 / 280] - Loss: 0.014
    Epoch / Batch [6 / 290] - Loss: 0.014
    Epoch / Batch [6 / 300] - Loss: 0.016
    Epoch / Batch [6 / 310] - Loss: 0.017
    Epoch / Batch [6 / 320] - Loss: 0.015
    Epoch / Batch [6 / 330] - Loss: 0.014
    Epoch / Batch [6 / 340] - Loss: 0.013
    Epoch / Batch [6 / 350] - Loss: 0.013
    Epoch / Batch [6 / 360] - Loss: 0.017
    Epoch / Batch [6 / 370] - Loss: 0.014
    Epoch / Batch [6 / 380] - Loss: 0.018
    Epoch / Batch [6 / 390] - Loss: 0.016
    Epoch / Batch [6 / 400] - Loss: 0.016
    Epoch / Batch [6 / 410] - Loss: 0.016
    Epoch / Batch [6 / 420] - Loss: 0.016
    Epoch / Batch [6 / 430] - Loss: 0.014
    Epoch / Batch [6 / 440] - Loss: 0.016
    Epoch / Batch [6 / 450] - Loss: 0.014
    Epoch / Batch [6 / 460] - Loss: 0.014
    Epoch / Batch [6 / 470] - Loss: 0.014
    Epoch / Batch [6 / 480] - Loss: 0.016
    Epoch / Batch [6 / 490] - Loss: 0.014
    Epoch / Batch [6 / 500] - Loss: 0.015
    Epoch / Batch [6 / 510] - Loss: 0.015
    Epoch / Batch [6 / 520] - Loss: 0.015
    Epoch / Batch [6 / 530] - Loss: 0.015
    Epoch / Batch [6 / 540] - Loss: 0.013
    Epoch / Batch [6 / 550] - Loss: 0.013
    Epoch / Batch [6 / 560] - Loss: 0.013
    Epoch / Batch [6 / 570] - Loss: 0.016
    Epoch / Batch [6 / 580] - Loss: 0.015
    Epoch / Batch [6 / 590] - Loss: 0.014
    Epoch / Batch [6 / 600] - Loss: 0.016
    Epoch / Batch [6 / 610] - Loss: 0.015
    Epoch / Batch [6 / 620] - Loss: 0.014
    Epoch / Batch [6 / 630] - Loss: 0.015
    Epoch / Validation Batch [6 / 10] - Loss: 0.015
    Epoch / Validation Batch [6 / 20] - Loss: 0.015
    Epoch / Validation Batch [6 / 30] - Loss: 0.015
    Epoch / Validation Batch [6 / 40] - Loss: 0.015
    Epoch / Validation Batch [6 / 50] - Loss: 0.014
    Epoch / Validation Batch [6 / 60] - Loss: 0.015
    Epoch / Validation Batch [6 / 70] - Loss: 0.015
    Epoch / Validation Batch [6 / 80] - Loss: 0.015
    Epoch / Validation Batch [6 / 90] - Loss: 0.015
    Epoch / Batch [7 / 10] - Loss: 0.014
    Epoch / Batch [7 / 20] - Loss: 0.013
    Epoch / Batch [7 / 30] - Loss: 0.014
    Epoch / Batch [7 / 40] - Loss: 0.014
    Epoch / Batch [7 / 50] - Loss: 0.014
    Epoch / Batch [7 / 60] - Loss: 0.014
    Epoch / Batch [7 / 70] - Loss: 0.014
    Epoch / Batch [7 / 80] - Loss: 0.014
    Epoch / Batch [7 / 90] - Loss: 0.015
    Epoch / Batch [7 / 100] - Loss: 0.015
    Epoch / Batch [7 / 110] - Loss: 0.016
    Epoch / Batch [7 / 120] - Loss: 0.014
    Epoch / Batch [7 / 130] - Loss: 0.013
    Epoch / Batch [7 / 140] - Loss: 0.013
    Epoch / Batch [7 / 150] - Loss: 0.012
    Epoch / Batch [7 / 160] - Loss: 0.013
    Epoch / Batch [7 / 170] - Loss: 0.015
    Epoch / Batch [7 / 180] - Loss: 0.013
    Epoch / Batch [7 / 190] - Loss: 0.015
    Epoch / Batch [7 / 200] - Loss: 0.011
    Epoch / Batch [7 / 210] - Loss: 0.014
    Epoch / Batch [7 / 220] - Loss: 0.013
    Epoch / Batch [7 / 230] - Loss: 0.013
    Epoch / Batch [7 / 240] - Loss: 0.013
    Epoch / Batch [7 / 250] - Loss: 0.013
    Epoch / Batch [7 / 260] - Loss: 0.013
    Epoch / Batch [7 / 270] - Loss: 0.012
    Epoch / Batch [7 / 280] - Loss: 0.012
    Epoch / Batch [7 / 290] - Loss: 0.014
    Epoch / Batch [7 / 300] - Loss: 0.012
    Epoch / Batch [7 / 310] - Loss: 0.013
    Epoch / Batch [7 / 320] - Loss: 0.014
    Epoch / Batch [7 / 330] - Loss: 0.014
    Epoch / Batch [7 / 340] - Loss: 0.016
    Epoch / Batch [7 / 350] - Loss: 0.013
    Epoch / Batch [7 / 360] - Loss: 0.014
    Epoch / Batch [7 / 370] - Loss: 0.013
    Epoch / Batch [7 / 380] - Loss: 0.014
    Epoch / Batch [7 / 390] - Loss: 0.015
    Epoch / Batch [7 / 400] - Loss: 0.013
    Epoch / Batch [7 / 410] - Loss: 0.014
    Epoch / Batch [7 / 420] - Loss: 0.012
    Epoch / Batch [7 / 430] - Loss: 0.014
    Epoch / Batch [7 / 440] - Loss: 0.016
    Epoch / Batch [7 / 450] - Loss: 0.015
    Epoch / Batch [7 / 460] - Loss: 0.015
    Epoch / Batch [7 / 470] - Loss: 0.012
    Epoch / Batch [7 / 480] - Loss: 0.014
    Epoch / Batch [7 / 490] - Loss: 0.015
    Epoch / Batch [7 / 500] - Loss: 0.012
    Epoch / Batch [7 / 510] - Loss: 0.013
    Epoch / Batch [7 / 520] - Loss: 0.011
    Epoch / Batch [7 / 530] - Loss: 0.016
    Epoch / Batch [7 / 540] - Loss: 0.013
    Epoch / Batch [7 / 550] - Loss: 0.016
    Epoch / Batch [7 / 560] - Loss: 0.014
    Epoch / Batch [7 / 570] - Loss: 0.013
    Epoch / Batch [7 / 580] - Loss: 0.014
    Epoch / Batch [7 / 590] - Loss: 0.014
    Epoch / Batch [7 / 600] - Loss: 0.013
    Epoch / Batch [7 / 610] - Loss: 0.012
    Epoch / Batch [7 / 620] - Loss: 0.014
    Epoch / Batch [7 / 630] - Loss: 0.015
    Epoch / Validation Batch [7 / 10] - Loss: 0.015
    Epoch / Validation Batch [7 / 20] - Loss: 0.016
    Epoch / Validation Batch [7 / 30] - Loss: 0.015
    Epoch / Validation Batch [7 / 40] - Loss: 0.015
    Epoch / Validation Batch [7 / 50] - Loss: 0.016
    Epoch / Validation Batch [7 / 60] - Loss: 0.015
    Epoch / Validation Batch [7 / 70] - Loss: 0.015
    Epoch / Validation Batch [7 / 80] - Loss: 0.017
    Epoch / Validation Batch [7 / 90] - Loss: 0.013
    Epoch / Batch [8 / 10] - Loss: 0.013
    Epoch / Batch [8 / 20] - Loss: 0.013
    Epoch / Batch [8 / 30] - Loss: 0.010
    Epoch / Batch [8 / 40] - Loss: 0.012
    Epoch / Batch [8 / 50] - Loss: 0.013
    Epoch / Batch [8 / 60] - Loss: 0.011
    Epoch / Batch [8 / 70] - Loss: 0.013
    Epoch / Batch [8 / 80] - Loss: 0.014
    Epoch / Batch [8 / 90] - Loss: 0.013
    Epoch / Batch [8 / 100] - Loss: 0.014
    Epoch / Batch [8 / 110] - Loss: 0.013
    Epoch / Batch [8 / 120] - Loss: 0.013
    Epoch / Batch [8 / 130] - Loss: 0.015
    Epoch / Batch [8 / 140] - Loss: 0.012
    Epoch / Batch [8 / 150] - Loss: 0.014
    Epoch / Batch [8 / 160] - Loss: 0.013
    Epoch / Batch [8 / 170] - Loss: 0.013
    Epoch / Batch [8 / 180] - Loss: 0.012
    Epoch / Batch [8 / 190] - Loss: 0.013
    Epoch / Batch [8 / 200] - Loss: 0.013
    Epoch / Batch [8 / 210] - Loss: 0.012
    Epoch / Batch [8 / 220] - Loss: 0.013
    Epoch / Batch [8 / 230] - Loss: 0.013
    Epoch / Batch [8 / 240] - Loss: 0.010
    Epoch / Batch [8 / 250] - Loss: 0.012
    Epoch / Batch [8 / 260] - Loss: 0.013
    Epoch / Batch [8 / 270] - Loss: 0.013
    Epoch / Batch [8 / 280] - Loss: 0.015
    Epoch / Batch [8 / 290] - Loss: 0.014
    Epoch / Batch [8 / 300] - Loss: 0.013
    Epoch / Batch [8 / 310] - Loss: 0.011
    Epoch / Batch [8 / 320] - Loss: 0.014
    Epoch / Batch [8 / 330] - Loss: 0.012
    Epoch / Batch [8 / 340] - Loss: 0.013
    Epoch / Batch [8 / 350] - Loss: 0.013
    Epoch / Batch [8 / 360] - Loss: 0.012
    Epoch / Batch [8 / 370] - Loss: 0.011
    Epoch / Batch [8 / 380] - Loss: 0.011
    Epoch / Batch [8 / 390] - Loss: 0.012
    Epoch / Batch [8 / 400] - Loss: 0.013
    Epoch / Batch [8 / 410] - Loss: 0.012
    Epoch / Batch [8 / 420] - Loss: 0.011
    Epoch / Batch [8 / 430] - Loss: 0.013
    Epoch / Batch [8 / 440] - Loss: 0.013
    Epoch / Batch [8 / 450] - Loss: 0.011
    Epoch / Batch [8 / 460] - Loss: 0.013
    Epoch / Batch [8 / 470] - Loss: 0.014
    Epoch / Batch [8 / 480] - Loss: 0.014
    Epoch / Batch [8 / 490] - Loss: 0.014
    Epoch / Batch [8 / 500] - Loss: 0.014
    Epoch / Batch [8 / 510] - Loss: 0.015
    Epoch / Batch [8 / 520] - Loss: 0.013
    Epoch / Batch [8 / 530] - Loss: 0.013
    Epoch / Batch [8 / 540] - Loss: 0.012
    Epoch / Batch [8 / 550] - Loss: 0.014
    Epoch / Batch [8 / 560] - Loss: 0.014
    Epoch / Batch [8 / 570] - Loss: 0.013
    Epoch / Batch [8 / 580] - Loss: 0.014
    Epoch / Batch [8 / 590] - Loss: 0.012
    Epoch / Batch [8 / 600] - Loss: 0.013
    Epoch / Batch [8 / 610] - Loss: 0.013
    Epoch / Batch [8 / 620] - Loss: 0.013
    Epoch / Batch [8 / 630] - Loss: 0.013
    Epoch / Validation Batch [8 / 10] - Loss: 0.013
    Epoch / Validation Batch [8 / 20] - Loss: 0.013
    Epoch / Validation Batch [8 / 30] - Loss: 0.014
    Epoch / Validation Batch [8 / 40] - Loss: 0.015
    Epoch / Validation Batch [8 / 50] - Loss: 0.013
    Epoch / Validation Batch [8 / 60] - Loss: 0.013
    Epoch / Validation Batch [8 / 70] - Loss: 0.016
    Epoch / Validation Batch [8 / 80] - Loss: 0.015
    Epoch / Validation Batch [8 / 90] - Loss: 0.014
    Epoch / Batch [9 / 10] - Loss: 0.012
    Epoch / Batch [9 / 20] - Loss: 0.011
    Epoch / Batch [9 / 30] - Loss: 0.011
    Epoch / Batch [9 / 40] - Loss: 0.012
    Epoch / Batch [9 / 50] - Loss: 0.012
    Epoch / Batch [9 / 60] - Loss: 0.014
    Epoch / Batch [9 / 70] - Loss: 0.011
    Epoch / Batch [9 / 80] - Loss: 0.011
    Epoch / Batch [9 / 90] - Loss: 0.010
    Epoch / Batch [9 / 100] - Loss: 0.011
    Epoch / Batch [9 / 110] - Loss: 0.010
    Epoch / Batch [9 / 120] - Loss: 0.009
    Epoch / Batch [9 / 130] - Loss: 0.012
    Epoch / Batch [9 / 140] - Loss: 0.012
    Epoch / Batch [9 / 150] - Loss: 0.013
    Epoch / Batch [9 / 160] - Loss: 0.012
    Epoch / Batch [9 / 170] - Loss: 0.012
    Epoch / Batch [9 / 180] - Loss: 0.012
    Epoch / Batch [9 / 190] - Loss: 0.012
    Epoch / Batch [9 / 200] - Loss: 0.013
    Epoch / Batch [9 / 210] - Loss: 0.013
    Epoch / Batch [9 / 220] - Loss: 0.013
    Epoch / Batch [9 / 230] - Loss: 0.013
    Epoch / Batch [9 / 240] - Loss: 0.013
    Epoch / Batch [9 / 250] - Loss: 0.013
    Epoch / Batch [9 / 260] - Loss: 0.011
    Epoch / Batch [9 / 270] - Loss: 0.012
    Epoch / Batch [9 / 280] - Loss: 0.013
    Epoch / Batch [9 / 290] - Loss: 0.013
    Epoch / Batch [9 / 300] - Loss: 0.012
    Epoch / Batch [9 / 310] - Loss: 0.011
    Epoch / Batch [9 / 320] - Loss: 0.014
    Epoch / Batch [9 / 330] - Loss: 0.012
    Epoch / Batch [9 / 340] - Loss: 0.011
    Epoch / Batch [9 / 350] - Loss: 0.011
    Epoch / Batch [9 / 360] - Loss: 0.011
    Epoch / Batch [9 / 370] - Loss: 0.011
    Epoch / Batch [9 / 380] - Loss: 0.013
    Epoch / Batch [9 / 390] - Loss: 0.011
    Epoch / Batch [9 / 400] - Loss: 0.012
    Epoch / Batch [9 / 410] - Loss: 0.011
    Epoch / Batch [9 / 420] - Loss: 0.012
    Epoch / Batch [9 / 430] - Loss: 0.011
    Epoch / Batch [9 / 440] - Loss: 0.012
    Epoch / Batch [9 / 450] - Loss: 0.010
    Epoch / Batch [9 / 460] - Loss: 0.011
    Epoch / Batch [9 / 470] - Loss: 0.012
    Epoch / Batch [9 / 480] - Loss: 0.012
    Epoch / Batch [9 / 490] - Loss: 0.013
    Epoch / Batch [9 / 500] - Loss: 0.013
    Epoch / Batch [9 / 510] - Loss: 0.011
    Epoch / Batch [9 / 520] - Loss: 0.011
    Epoch / Batch [9 / 530] - Loss: 0.012
    Epoch / Batch [9 / 540] - Loss: 0.012
    Epoch / Batch [9 / 550] - Loss: 0.012
    Epoch / Batch [9 / 560] - Loss: 0.012
    Epoch / Batch [9 / 570] - Loss: 0.012
    Epoch / Batch [9 / 580] - Loss: 0.014
    Epoch / Batch [9 / 590] - Loss: 0.013
    Epoch / Batch [9 / 600] - Loss: 0.011
    Epoch / Batch [9 / 610] - Loss: 0.012
    Epoch / Batch [9 / 620] - Loss: 0.010
    Epoch / Batch [9 / 630] - Loss: 0.011
    Epoch / Validation Batch [9 / 10] - Loss: 0.012
    Epoch / Validation Batch [9 / 20] - Loss: 0.015
    Epoch / Validation Batch [9 / 30] - Loss: 0.015
    Epoch / Validation Batch [9 / 40] - Loss: 0.017
    Epoch / Validation Batch [9 / 50] - Loss: 0.016
    Epoch / Validation Batch [9 / 60] - Loss: 0.014
    Epoch / Validation Batch [9 / 70] - Loss: 0.015
    Epoch / Validation Batch [9 / 80] - Loss: 0.011
    Epoch / Validation Batch [9 / 90] - Loss: 0.015
    Epoch / Batch [10 / 10] - Loss: 0.009
    Epoch / Batch [10 / 20] - Loss: 0.011
    Epoch / Batch [10 / 30] - Loss: 0.009
    Epoch / Batch [10 / 40] - Loss: 0.011
    Epoch / Batch [10 / 50] - Loss: 0.011
    Epoch / Batch [10 / 60] - Loss: 0.009
    Epoch / Batch [10 / 70] - Loss: 0.009
    Epoch / Batch [10 / 80] - Loss: 0.011
    Epoch / Batch [10 / 90] - Loss: 0.013
    Epoch / Batch [10 / 100] - Loss: 0.011
    Epoch / Batch [10 / 110] - Loss: 0.010
    Epoch / Batch [10 / 120] - Loss: 0.011
    Epoch / Batch [10 / 130] - Loss: 0.009
    Epoch / Batch [10 / 140] - Loss: 0.010
    Epoch / Batch [10 / 150] - Loss: 0.013
    Epoch / Batch [10 / 160] - Loss: 0.009
    Epoch / Batch [10 / 170] - Loss: 0.011
    Epoch / Batch [10 / 180] - Loss: 0.010
    Epoch / Batch [10 / 190] - Loss: 0.013
    Epoch / Batch [10 / 200] - Loss: 0.009
    Epoch / Batch [10 / 210] - Loss: 0.010
    Epoch / Batch [10 / 220] - Loss: 0.011
    Epoch / Batch [10 / 230] - Loss: 0.011
    Epoch / Batch [10 / 240] - Loss: 0.010
    Epoch / Batch [10 / 250] - Loss: 0.010
    Epoch / Batch [10 / 260] - Loss: 0.012
    Epoch / Batch [10 / 270] - Loss: 0.010
    Epoch / Batch [10 / 280] - Loss: 0.009
    Epoch / Batch [10 / 290] - Loss: 0.013
    Epoch / Batch [10 / 300] - Loss: 0.011
    Epoch / Batch [10 / 310] - Loss: 0.010
    Epoch / Batch [10 / 320] - Loss: 0.009
    Epoch / Batch [10 / 330] - Loss: 0.012
    Epoch / Batch [10 / 340] - Loss: 0.011
    Epoch / Batch [10 / 350] - Loss: 0.011
    Epoch / Batch [10 / 360] - Loss: 0.011
    Epoch / Batch [10 / 370] - Loss: 0.011
    Epoch / Batch [10 / 380] - Loss: 0.012
    Epoch / Batch [10 / 390] - Loss: 0.012
    Epoch / Batch [10 / 400] - Loss: 0.008
    Epoch / Batch [10 / 410] - Loss: 0.013
    Epoch / Batch [10 / 420] - Loss: 0.013
    Epoch / Batch [10 / 430] - Loss: 0.009
    Epoch / Batch [10 / 440] - Loss: 0.012
    Epoch / Batch [10 / 450] - Loss: 0.011
    Epoch / Batch [10 / 460] - Loss: 0.012
    Epoch / Batch [10 / 470] - Loss: 0.011
    Epoch / Batch [10 / 480] - Loss: 0.010
    Epoch / Batch [10 / 490] - Loss: 0.013
    Epoch / Batch [10 / 500] - Loss: 0.012
    Epoch / Batch [10 / 510] - Loss: 0.012
    Epoch / Batch [10 / 520] - Loss: 0.010
    Epoch / Batch [10 / 530] - Loss: 0.013
    Epoch / Batch [10 / 540] - Loss: 0.012
    Epoch / Batch [10 / 550] - Loss: 0.010
    Epoch / Batch [10 / 560] - Loss: 0.013
    Epoch / Batch [10 / 570] - Loss: 0.013
    Epoch / Batch [10 / 580] - Loss: 0.010
    Epoch / Batch [10 / 590] - Loss: 0.010
    Epoch / Batch [10 / 600] - Loss: 0.012
    Epoch / Batch [10 / 610] - Loss: 0.011
    Epoch / Batch [10 / 620] - Loss: 0.010
    Epoch / Batch [10 / 630] - Loss: 0.012
    Epoch / Validation Batch [10 / 10] - Loss: 0.016
    Epoch / Validation Batch [10 / 20] - Loss: 0.014
    Epoch / Validation Batch [10 / 30] - Loss: 0.012
    Epoch / Validation Batch [10 / 40] - Loss: 0.014
    Epoch / Validation Batch [10 / 50] - Loss: 0.016
    Epoch / Validation Batch [10 / 60] - Loss: 0.013
    Epoch / Validation Batch [10 / 70] - Loss: 0.014
    Epoch / Validation Batch [10 / 80] - Loss: 0.014
    Epoch / Validation Batch [10 / 90] - Loss: 0.015
    


```python
print('Visualize Filters after training:')
visualize()
```

    Visualize Filters after training:
    <generator object Module.children at 0x7e9eadd0c2b0>
    


    
![png]({{ site.baseurl }}/assets/img/output_35_1.png)
    


    <generator object Module.children at 0x7e9e9f32fd30>
    


    
![png]({{ site.baseurl }}/assets/img/output_35_3.png)
    


    <generator object Module.children at 0x7e9fc0b07c60>
    


    
![png]({{ site.baseurl }}/assets/img/output_35_5.png)
    


## Visualising feature maps

### Feature maps from first convolution layer:


```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_gpu.conv1[0].register_forward_hook(get_activation('conv1'))
data, _ = ins_dataset_train[0]
data.unsqueeze_(0)
model_cpu = model_gpu.cpu()
output = model_gpu(data)

act = activation['conv1'].squeeze()
print(act.size())
fig, axarr = plt.subplots(2,2)
for idx in range(act.size(0)):
    axarr[0,0].imshow(act[0],cmap="gray")
    axarr[0,1].imshow(act[1],cmap="gray")
    axarr[1,0].imshow(act[2],cmap="gray")
    axarr[1,1].imshow(act[3], cmap="gray")
```

    torch.Size([16, 254, 254])
    


    
![png]({{ site.baseurl }}/assets/img/output_37_1.png)
    


### Feature maps from second convolution layer:


```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model_gpu.conv2[0].register_forward_hook(get_activation('conv2'))
data, _ = ins_dataset_train[0]
data.unsqueeze_(0)
model_cpu = model_gpu.cpu()
output = model_cpu(data)

act = activation['conv2'].squeeze()
fig, axarr = plt.subplots(2,2)
for idx in range(act.size(0)):
    axarr[0,0].imshow(act[0], cmap="gray")
    axarr[0,1].imshow(act[1], cmap="gray")
    axarr[1,0].imshow(act[2], cmap="gray")
    axarr[1,1].imshow(act[3], cmap="gray")
```


    
![png]({{ site.baseurl }}/assets/img/output_39_0.png)
    


### Feature maps from third convolution layer:


```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model_gpu.conv3[0].register_forward_hook(get_activation('conv3'))
data, _ = ins_dataset_train[0]
data.unsqueeze_(0)
model_cpu = model_gpu.cpu()
output = model_cpu(data)

act = activation['conv3'].squeeze()
fig, axarr = plt.subplots(2,2)
for idx in range(act.size(0)):
    axarr[0,0].imshow(act[0], cmap="gray")
    axarr[0,1].imshow(act[1], cmap="gray")
    axarr[1,0].imshow(act[2], cmap="gray")
    axarr[1,1].imshow(act[3], cmap="gray")
```


    
![png]({{ site.baseurl }}/assets/img/output_41_0.png)
    


### Feature maps from fourth convolution layer:


```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model_gpu.conv4[0].register_forward_hook(get_activation('conv4'))
data, _ = ins_dataset_train[0]
data.unsqueeze_(0)
model_cpu = model_gpu.cpu()
output = model_cpu(data)

act = activation['conv4'].squeeze()
fig, axarr = plt.subplots(2,2)
for idx in range(act.size(0)):
    axarr[0,0].imshow(act[0], cmap="gray")
    axarr[0,1].imshow(act[1], cmap="gray")
    axarr[1,0].imshow(act[2], cmap="gray")
    axarr[1,1].imshow(act[3], cmap="gray")
```


    
![png]({{ site.baseurl }}/assets/img/output_43_0.png)
    



```python
correct = 0
total = 0

with torch.no_grad():

    # Iterate over the test set
    for data in test_set_loader:
        model_gpu.to(device)
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        outputs = model_gpu(images)

        # torch.max is an argmax operation
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
```

    Accuracy of the network on the test images: 51 %
    


```python
torch.save(model_gpu.state_dict(), './my_mnist_model.pt')

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(labels.cpu(), predicted.cpu())
```


```python
import itertools

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix very prettily.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # Specify the tick marks and axis text
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # The data formatting
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
```


```python
plot_confusion_matrix(cm, classes)
```

### Confusion matrix, without normalization
    


    
![png]({{ site.baseurl }}/assets/img/output_47_1.png)
    

