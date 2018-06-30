# Comments:
# использовал стандартный подход в архитектуре сети: пачка сверток + батчнорм + релу
# после делаем пулинг + дропаут
# в конце делаем два полносвязных слоя с батчнормом и дропаутом 

# при обучении использовал Adam оптимайзер с weight_decay = 1e-4 и lr = 0.001
# так же поковырялся с аугментацией картинок: флипал по горизонтали и крутил на стучайные углы

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.classifier2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.bn(F.dropout(self.classifier(out), p=0.5))
        out = self.classifier2(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.Dropout2d(0.25)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
