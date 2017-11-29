# Gas Sensors Data Set

The data set used here is downloaded from UCI machine learning repository. 
 
Link is http://archive.ics.uci.edu/ml/datasets/gas+sensors+for+home+activity+monitoring 

Tools are:
Python2+Keras+Theano (2016)

Flow:

1. Data parsing
2. Data preprocessing
3. Data shuffling 
4. One hot coding for multi-label classification
5. Model creating
6. Model compiling
7. Model training (80% of original data)
8. Model evaluating (20% of original data)


Training accuracy: 95.43%

Test accuracy: 96%

Now it is 2017, **Python3+PyTorch** will be the mainstream.

In Pytorch, if you use the loss function as `CrossEntropyLoss`, it is not necessary to do one-hot encoding for labels.

Test accuracy is: 96%, which is the same with previous tools combination.

Loss is 0.026

There are only three layers of the network in Pytorch to reach this accuracy. But my hidden dimension is 64, which is way larger than the previous one, which is 10.
