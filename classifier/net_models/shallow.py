import torch.nn as nn

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class shallow(nn.Module):
    def __init__(self, num_classes=(5, 5)):
        super().__init__()
        
        # Convolutional-Batchnorm-Relu 1
        self.unit1 = Unit(in_channels=3, out_channels=12)
        
        # Convolutional-Batchnorm-Relu 2
        self.unit2 = Unit(in_channels=12, out_channels=24)
        
        # Max Pool 1 224->112
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Convolutional-Batchnorm-Relu 3
        self.unit3 = Unit(in_channels=24, out_channels=36)
        
        # Max Pool 2 112->56
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Convolutional-Batchnorm-Relu 4
        self.unit4 = Unit(in_channels=36, out_channels=48)
        
        # Max Pool 3 56->28
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Convolutional-Batchnorm-Relu 5
        self.unit5 = Unit(in_channels=48, out_channels=48)
        
        # Max Pool 4 28->7
        self.pool4 = nn.MaxPool2d(kernel_size=4)
        
        # Convolutional-Batchnorm-Relu 6 7->3
        self.unit6 = Unit(in_channels=48, out_channels=96, kernel_size=5, stride=1, padding=0)

        self.dropout = nn.Dropout(0.5)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit3, 
                                 self.pool2, self.unit4, self.pool3, self.unit5,
                                 self.pool4, self.unit6, self.dropout)
        
        
        # Fully connected - disease
        if type(num_classes) is not tuple:
            self.fc1 = nn.Linear(in_features=3*3*96, out_features=num_classes)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(in_features=3*3*96, out_features=num_classes[0])
            self.fc2 = nn.Linear(in_features=3*3*96, out_features=num_classes[1])
            
    def forward(self, inp):
        out = self.net(inp)
        out = out.view(-1, 3*3*96)
        out1 = self.fc1(out)
        
        if self.fc2 == None:
            return out1
        else:
            out2 = self.fc2(out)
            return out1, out2