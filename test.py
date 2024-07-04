#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


# In[2]:


test_path = './testdata/'
model_path= './model.pth'


# In[3]:


class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()
        #using vgg16 feature(including convolution layer and pooling layer)
        self.features = torchvision.models.vgg16_bn(pretrained=True).features
        self.classifier = nn.Sequential(
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
model = VGGNet16() # Define the network model for training
lossF = nn.CrossEntropyLoss() # make output more smoother(incorporated softmax())
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[4]:


if __name__ == '__main__':
    transform_rgb = transforms.Compose(
        [
         transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
         ])
    testset = torchvision.datasets.ImageFolder(test_path, transform=transform_rgb)
    #train_set, test_set = train_test_split(trainset, train_size=0.8,random_state=309)
    #batch_size=32 Speed up trainning
    #trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10)
    model=VGGNet16()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    classes = ('cherry','strawberry','tomato')
    correct = 0
    total = 0
    pred_correct = {classname: 0 for classname in classes}
    pred_total = {classname: 0 for classname in classes}
    print('Start testing.')
    with torch.no_grad():
        for data in testloader:
            model.eval()
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    pred_correct[classes[label]] += 1
                pred_total[classes[label]] += 1
    #print the accuracy for each class            
    print(f'Overall accuracy: {100 * correct // total} %')
    #print the accuracy for each class
    for classname, count in pred_correct.items():
        accuracy = 100 * float(count) / pred_total[classname]
        print(f'Accuracy for class {classname:5s} is {accuracy:.1f} %')
    print('Finish testing.')

