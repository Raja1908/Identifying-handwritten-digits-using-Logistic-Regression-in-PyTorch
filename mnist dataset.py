#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
from torch.autograd import Variable 


# In[13]:


# Hyper Parameters 
input_size = 784
num_classes = 10
num_epochs = 6
batch_size = 100
learning_rate = 0.011


# In[14]:


# MNIST Dataset (Images and Labels) 
train_dataset = dsets.MNIST(root ='./data', 
							train = True, 
							transform = transforms.ToTensor(), 
							download = True) 

test_dataset = dsets.MNIST(root ='./data', 
						train = False, 
						transform = transforms.ToTensor()) 

# Dataset Loader (Input Pipline) 
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
										batch_size = batch_size, 
										shuffle = True) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
										batch_size = batch_size, 
										shuffle = False) 


# In[15]:


class LogisticRegression(nn.Module): 
	def __init__(self, input_size, num_classes): 
		super(LogisticRegression, self).__init__() 
		self.linear = nn.Linear(input_size, num_classes) 

	def forward(self, x): 
		out = self.linear(x) 
		return out 


# In[16]:


model = LogisticRegression(input_size, num_classes) 


# In[17]:


criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 


# In[18]:


# Training the Model 
for epoch in range(num_epochs): 
	for i, (images, labels) in enumerate(train_loader): 
		images = Variable(images.view(-1, 28 * 28)) 
		labels = Variable(labels) 

		# Forward + Backward + Optimize 
		optimizer.zero_grad() 
		outputs = model(images) 
		loss = criterion(outputs, labels) 
		loss.backward() 
		optimizer.step() 

		if (i + 1) % 100 == 0: 
			print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data)) 


# In[19]:


# Test the Model 
correct = 0
total = 0
for images, labels in test_loader: 
	images = Variable(images.view(-1, 28 * 28)) 
	outputs = model(images) 
	_, predicted = torch.max(outputs.data, 1) 
	total += labels.size(0) 
	correct += (predicted == labels).sum() 

print('Accuracy of the model on the 10000 test images: % d %%' % ( 100 * correct / total)) 


# In[ ]:




