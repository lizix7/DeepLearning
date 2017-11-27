import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

# Hyper Parameters
input_size = 10
output_size = 3
hidden_dimension = 64
num_epochs = 60
batch_size = 50
learning_rate = 0.001

# fix random seed for reproducibility
# use for random shuffling 
seed = 7
np.random.seed(seed)

# load gas dataset
metadata = np.loadtxt('HT_Sensor_metadata.dat', skiprows=1, dtype=str)

dataset = np.loadtxt('HT_Sensor_dataset.dat', skiprows=1)

banana_id = np.array(metadata[metadata[:,2]=="banana",0],dtype=float)

wine_id = np.array(metadata[metadata[:,2]=="wine",0],dtype=float)

background_id = np.array(metadata[metadata[:,2]=="background",0],dtype=float)

for index in range(len(dataset)):
	if dataset[index,0] in banana_id:
		dataset[index,0] = 0
		
	elif dataset[index,0] in wine_id:
		dataset[index,0] = 1
		
	else:
		dataset[index,0] = 2

# split into input (X) and output (Y) variables
X = np.array(dataset[:,2:12],dtype=float)
Y = np.array(dataset[:,0],dtype=int)

#print (X)
#print (Y)

# data processing

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# split into 80% for training and 20% for test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=seed)

#print (X_train)
#print (Y_train)

X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)

training_samples = utils_data.TensorDataset(X_train,Y_train)
data_loader = utils_data.DataLoader(training_samples, batch_size=50)

# create model using torch's nn module

class _classifier(nn.Module):
    def __init__(self,input_size,output_size):
        super(_classifier, self).__init__()
        self.f1 = nn.Linear(input_size,hidden_dimension)
        self.f2 = nn.Linear(hidden_dimension,10)
        self.f3 = nn.Linear(10,10)
        self.f4 = nn.Linear(10,10)
        self.f5 = nn.Linear(10,10)
        self.f6 = nn.Linear(10,10)
        self.f7 = nn.Linear(10,output_size)
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        x = F.relu(x)
        x = self.f4(x)
        x = F.relu(x)
        x = self.f5(x)
        x = F.relu(x)
        x = self.f6(x)
        x = F.relu(x)
        x = self.f7(x)
        x = F.softmax(x)
        return x

model = _classifier(input_size,output_size)
# Optimizer and Loss function
optimizer = optim.Adam(model.parameters())
#criterion = nn.CrossEntropyLoss()
criterion = nn.MultiLabelSoftMarginLoss()

# Train the model

for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):
        # Get each batch
        inputs, labels = data
        # Convert numpy array to torch Variable
        inputs = Variable(inputs.type(torch.FloatTensor))
        # in pytorch, the lable has to be LongTensor
        labels = torch.LongTensor(labels).view(-1,1)
        #print (inputs)
        #print (labels)
        #print (len(labels))

        # Convert the labels into one hot encoding
        labels_onehot = torch.FloatTensor(len(labels),output_size)

        labels_onehot.zero_()
        labels_onehot.scatter_(1,labels,1.)
        labels_onehot = Variable(labels_onehot)
        # Check the one hot encoding
        #print(labels_onehot)

        # Forward + Backward + Optimize    
        outputs = model(inputs)
        #print (outputs)
        # For the loss function, both inputs should be torch.Variable
        loss = criterion(outputs, labels_onehot)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        #print (outputs)

    if (epoch+1) % 5 == 0:
        print ('Epoch [%d/%d], Loss: %.4f' 
               %(epoch+1, num_epochs, loss.data[0]))

# Test the model
inputs_test = Variable(torch.FloatTensor(X_test))
targets_test = torch.LongTensor(Y_test)
outputs_test = model(inputs_test)
_, predicts = torch.max(outputs_test.data,1)
    
# Calculate accuration
print('Accuracy of the network %d %%'
     % (100*torch.sum(targets_test==predicts)/len(X_test)))
