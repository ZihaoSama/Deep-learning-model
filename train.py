#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


train_path = './traindata/'
transform_rgb = transforms.Compose(
        [transforms.RandomRotation(10),
         transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
         #prevent overfiting
         transforms.RandomHorizontalFlip()
         ])
trainset = torchvision.datasets.ImageFolder(train_path, transform=transform_rgb)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
#do not need split train, test. only train
#testloader = torch.utils.data.DataLoader(test_set, batch_size=10)


# ### cnn structure by using pre_train Model feature

# In[5]:


class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()
        #using vgg16 feature(including convolution layer and pooling layer)
        self.features = torchvision.models.vgg16_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            #3 layers 
            nn.Linear(512*7*7, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 3, bias=False)
        )
        # Initialize Weights of the liner layers
        #set the initial weights for trainng
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #Kaiming Normal initialization method
                #zhengtaifenbu
                nn.init.kaiming_normal_(m.weight.data)
    
    def forward(self, x):
        x = self.features(x)  # The forward propagation
        x = x.view(-1, 512*7*7)#output of convolutional layer into a one-dimensional vector
        x = self.classifier(x)  # Then splice the results of features (feature layer of network output) to the classifier
        return x
'''training net'''
model = VGGNet16().to(device)  # Define the network model for training
lossF = nn.CrossEntropyLoss() # make output more smoother(incorporated softmax())
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ### Traing(epoch=10)

# In[7]:


print("start training")
##optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times
    train_num = 0.0
    train_accuracy = 0.0
    train_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = lossF(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        train_loss += abs(loss.item()) * inputs.size(0)
        outputs = torch.argmax(outputs, 1)
        accuracy = torch.sum(outputs == labels)
        train_accuracy = train_accuracy + accuracy
        train_num += inputs.size(0)
    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(epoch + 1, train_loss / train_num, train_accuracy / train_num))
print('Finished Training')


# In[8]:


print(model)


# In[9]:


PATH = './model.pth'
torch.save(model.state_dict(), PATH)

