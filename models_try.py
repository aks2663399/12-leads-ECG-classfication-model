#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
from transformer.transformer_block import TRANSFORMER
#%% Functions
class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer,  self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=(50, 1),
                stride=1,
                padding=0,
            ),
            nn.GroupNorm(2,32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (20, 1), 1, 0),
            nn.GroupNorm(2,64),
            nn.GELU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),
            nn.GroupNorm(2,64),
            nn.GELU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),
            nn.GroupNorm(2,64),
            nn.GELU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0),
            nn.GroupNorm(2,128),
            nn.GELU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),
            nn.GroupNorm(2,128),
            nn.GELU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 256, (2, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.GroupNorm(2, 256),
            nn.GELU(),
            nn.MaxPool2d((1, 1))
        )
        self.network1 = nn.Sequential(
            nn.Linear(256, 200),  # 卷积层后加一个普通的神经网络
            # nn.BatchNorm1d(256),
            nn.Dropout(0.7),
            nn.GELU(),
            nn.Linear(200, 128),  # 卷积层后加一个普通的神经网络
            # nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(128, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.networks2 = nn.Sequential(
            nn.Linear(256 * 2, 200),
            nn.Dropout(0.7),
            nn.Linear(200, 128),
        )
        self.transformer = TRANSFORMER()

        self.networks3 = nn.Sequential(
            nn.Linear(128, 100),
            nn.Dropout(0.7),
            nn.Linear(100, 9),
        )
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.reshape(x.shape[0], -1)  # win_num * feature
        x = self.network1(x)
        x = x.unsqueeze(0)
        x = self.dropout(x)
        x, (h_n, c_n) = self.bilstm(self.relu(x))
        x = h_n[:2, :, :]
        x = h_n.reshape(1, -1)
        x = self.networks2(x)
        x = x.unsqueeze(0)
        x = self.transformer(self.relu(x))
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.networks3(x)
        return x

#%% Main Function
if __name__ == '__main__':
    x = np.random.randn(3, 12, 3000, 1)
    x = torch.Tensor(x)
    cnn_bilstm = CNN_Transformer()
    y = cnn_bilstm(x)
    print(y.shape)