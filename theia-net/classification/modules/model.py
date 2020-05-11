import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(torch.nn.Module):
    """
    1D CNN model architecture.
        
    Attributes
    ----------
    num_in : int
        Exposure in seconds.
        
    n_classes : int
        Number of stellar evolutionary states to classify.
        
    log : _io.TextIOWrapper
        Log file.
        
    kernel1, kernel2 : int
        Kernel width of first and second convolution, respectively.
        
    stride1, stride2 : int
        Stride of first and second convolution, respectively.
        
    padding1, padding2 : int
        Zero-padding of first and second convolution, respectively.
        
    dropout : float
        Dropout probability applied to fully-connected part of network.
        
    hidden1, hidden2, hidden3 : int
        Number of hidden units in the first, second, and third fully-connected
        layers, respectively.
        
        
    Methods
    -------
    forward(x, s)
        Forward pass through the model architecture.
    """
    def __init__(self, num_in, n_classes, log, kernel1, kernel2, stride1, \
                 stride2, padding1, padding2, dropout, hidden1=2048, \
                 hidden2=1024, hidden3=256):
       
        super(CNN, self).__init__()
        
        self.log = log
        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_in = num_in
        self.n_classes = n_classes
        print(self.num_in, file=log)

        OUT_CHANNELS_1 = 64
        dilation1 = 1
        poolsize1 = 4
        
        OUT_CHANNELS_2 = 16
        dilation2 = 1
        poolsize2 = 2
        
        # first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=OUT_CHANNELS_1,
                               kernel_size=kernel1,
                               dilation=dilation1,
                               stride=stride1,
                               padding=padding1)
        self.num_out = ((self.num_in+2*padding1-dilation1* \
                         (kernel1-1)-1)/stride1)+1
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        self.bn1 = nn.BatchNorm1d(num_features=OUT_CHANNELS_1)
        self.pool1 = nn.AvgPool1d(kernel_size=poolsize1)
        self.num_out = (self.num_out/poolsize1)
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        
        # second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=OUT_CHANNELS_1,
                               out_channels=OUT_CHANNELS_2,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.num_out = ((self.num_out+2*padding2-dilation2* \
                         (kernel2-1)-1)/stride2)+1
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        self.bn2 = nn.BatchNorm1d(num_features=OUT_CHANNELS_2)
        self.pool2 = nn.AvgPool1d(kernel_size=poolsize2)
        self.num_out = (self.num_out/poolsize2)
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        
        # fully-connected layers
        self.num_out = OUT_CHANNELS_2*self.num_out
        assert str(self.num_out)[-1] == '0'
        print(self.num_out, file=log)
        self.num_out = int(self.num_out)
        
        self.linear1 = nn.Linear(self.num_out+1, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)

        # output prediction
        self.predict = nn.Linear(hidden3, self.n_classes)
    

    def forward(self, x, s):
        """
        Forward pass through the model architecture.
            
        Parameters
        ----------
        x : array_like
            Input time series data.
            
        s : array_like
            Standard deviation array.
            
        Returns
        ----------
        x : array_like
            Output prediction.
        """
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
       
        x = x.view(-1, self.num_out)
        x = torch.cat((x, s), 1)

        x = self.dropout(F.tanh(self.linear1(x)))
        x = self.dropout(F.tanh(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = self.predict(x)
        
        return x
