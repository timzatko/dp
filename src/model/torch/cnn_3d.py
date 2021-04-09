import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Net3DCNN(nn.Module):
    def __init__(self):
        super(Net3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32, affine=True, eps=0.001, momentum=0.99)
        self.mpool1 = nn.MaxPool3d(2, stride=2, padding=0)
        
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64, affine=True, eps=0.001, momentum=0.99)
        self.mpool2 = nn.MaxPool3d(3, stride=3)
        
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128, affine=True, eps=0.001, momentum=0.99)
        self.mpool3 = nn.MaxPool3d(4, stride=4)
        
        self.flt = nn.Flatten()
        self.dp1 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(10240, 256)
        self.dp2 = nn.Dropout(p=0.1)
        
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mpool1(x)
        torch.cuda.empty_cache()
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mpool2(x)
        torch.cuda.empty_cache()
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mpool3(x)      
        
        x = torch.transpose(x, 1, 4) # n, 4, 4, 5, 128 
        x = torch.transpose(x, 2, 3) # n, 4, 5, 4, 128 
        x = torch.transpose(x, 1, 3) # n, 4, 5, 4, 128
        
        x = self.flt(x)
        x = self.dp1(x)
        
        x = F.relu(self.fc1(x))
        torch.cuda.empty_cache()
        x = self.dp2(x)
        
        x = F.softmax(self.fc2(x), dim=1)
        
        return x
    
    
def load_weights(net, weights, weights_bn):
    print(list(net.children()))

    print("copy trainable parameters...\n")

    for (name, param), (tf_name, tf_param) in zip(net.named_parameters(), weights):
        if 'conv' in tf_name and 'kernel' in tf_name:
            tf_param = np.transpose(tf_param, axes=(4, 3, 0, 1, 2))
        else:
            tf_param = tf_param.transpose()
        print(f"\n--- {name} / {tf_name} ---")
        print(tf_param.shape)
        print(param.shape)
        param.data = torch.from_numpy(tf_param)

    print("\n\ncopy batch norm parameters...\n")

    i = 0

    for child in net.children():
        if isinstance(child, torch.nn.BatchNorm3d):
            var = weights_bn[i + 1]
            mean = weights_bn[i]
            print(f'{var.shape} ({len(var)}) {mean.shape} ({len(mean)})')
            child.running_var = torch.from_numpy(var)
            child.running_mean = torch.from_numpy(mean)
            i += 2
    
    return net