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
num_epochs = 80
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
# Transform the data into Tensor
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
# Build a dataset with features and labels
training_samples = utils_data.TensorDataset(X_train,Y_train)
# Why do we need to use DataLoader?
# Because we need the mini batch feature
data_loader = utils_data.DataLoader(training_samples, batch_size=50)

# create model using torch's nn module

class _classifier(nn.Module):
    def __init__(self,input_size,output_size):
        super(_classifier, self).__init__()
        self.f1 = nn.Linear(input_size,hidden_dimension)
        self.f2 = nn.Linear(hidden_dimension,10)
        self.f3 = nn.Linear(10,output_size)
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)
        x = self.f3(x)
        # In the last layer, you cannot add activation function
        # Because CrossEntropy has softmax function built-in
        # If you add x = F.relu(x), the accuracy will be really bad, like 50%. 
        return x

model = _classifier(input_size,output_size)
# Optimizer and Loss function
optimizer = optim.Adam(model.parameters())
# For Cross Entropy loss function, the softmax is internally computed.
criterion = nn.CrossEntropyLoss()
# For multilabel loss function, the last layer is softmax?
# At least we can say relu is much worse than softmax.
#criterion = nn.MultiLabelSoftMarginLoss()

# Train the model

for epoch in range(num_epochs):
    for i, data in enumerate(data_loader):
        # Get each batch
        inputs, labels = data
        # Convert Tensors into torch Variable
        # inputs has to be FloatTensor. So we change the Tensor type.
        inputs = Variable(inputs.type(torch.FloatTensor))
        labels = Variable(labels)
        # in pytorch, the lable has to be LongTensor
        # This .view(-1,1) is a must if the lost function is MLSML.
        #labels = torch.LongTensor(labels).view(-1,1)
        #print (inputs)
        #print (labels)
        #print (len(labels))

        # Convert the labels into one hot encoding
        # One hot encoding is necessary if loss function is MLSML.
        #labels_onehot = torch.FloatTensor(len(labels),output_size)
        #labels_onehot = torch.LongTensor(len(labels),output_size)
        #labels_onehot.zero_()
        #labels_onehot.scatter_(1,labels,1)
        # Variable() should never be put in loss criterion(). -- learnt from BUG
        # It will lose the information if doing so.
        #labels_onehot = Variable(labels_onehot)
        # Check the one hot encoding
        #print(labels_onehot)

        # Forward + Backward + Optimize   
        optimizer.zero_grad()  
        outputs = model(inputs)
        #print (outputs)
        # For the loss function, both inputs should be torch.Variable
        loss = criterion(outputs, labels)
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
# The index with largest probability is the predicted result
_, predicts = torch.max(outputs_test.data,1)
    
# Calculate accuration
print('Accuracy of the network %d %%'
     % (100*torch.sum(targets_test==predicts)/len(X_test)))
