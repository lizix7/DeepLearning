# Pytorch

This is my study notes for Pytorch.

#### 1. How to use Data Loader.

I learnt this part from the source: http://kevin-ho.website/Make-a-Acquaintance-with-Pytorch/


#### 2. How to one-hot encode your labels

```
labels = torch.LongTensor(labels).view(-1,1)
labels_onehot = torch.FloatTensor(len(labels),output_size)
labels_onehot = torch.LongTensor(len(labels),output_size)
labels_onehot.zero_()
labels_onehot.scatter_(1,labels,1)

labels_onehot = Variable(labels_onehot)
```

#### 3. How to write down your own module 
```
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
criterion = nn.CrossEntropyLoss()
```

#### 4. Different loss function
If using nn.MultiLabelSoftMarginLoss as your loss function, there are several points to notice:

1. The last layer should be softmax. I have tried to use relu, but the performance is bad.
2. 

