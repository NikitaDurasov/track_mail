import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Тут делаем свою сеть ResNet
"""

class Net(nn.Module):

    # Слои, в которых нет параметров для обучения можно не создавать, а брать из переменной F
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential()


    def forward(self, x):
        return self.net(x)