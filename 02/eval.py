# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.autograd import Variable
from cifar_model import Net
from optparse import OptionParser
from torch.utils.data import DataLoader
from torchvision.datasets import cifar
import torchvision.transforms as transforms
from tqdm import *

# разбираем аргументы коммандной строки
parser = OptionParser("Train cifar10 neural network")

parser.add_option("-i", "--input", dest="input", default='./',
                  help="Cifar data root directory")  # рутовый каталог откуда беруться данные

parser.add_option('-m',"--model", dest="model", default='./model_save/final.model_chekpoint_000.pth',
                  help="Model base path ") # путь к файлу модели

parser.add_option("-o", "--out", dest="out", type='string', default='solution.csv',
                  help="Path to tensorboard log")   # куда складывать результаты

def eval(options):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # Создаем модель, нужно сделать иплементацию    
    print("Creating model...")
    net = Net()#.cuda()
    net.eval()
    # Критерий кросс энтропия проверим, что у нас вск сходится
    criterion = nn.CrossEntropyLoss()#.cuda()

    # загружаем сеть
    cp_dic = torch.load(options.model)
    print(cp_dic.keys())
    net.load_state_dict(cp_dic['net'])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # данные для теста
    testset = cifar.CIFAR10(options.input, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    test_loss = 0
    print('Test model: ')

    ofile = open(options.out, /'w')
    print("id,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", file=ofile)

    for bid, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        inputs, labels = data

        # получаем переменные Variable
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True) # Variable(inputs, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
        outputs = net(inputs)

        # считаем ошибку
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        # считаем какие классы мы предсказали и сохраняем для
        # последующего расчета accuracy
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels.data).squeeze()
        for i in range(outputs.size(0)):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

        # печатаем для каждого класса вероятности
        probs = softmax(outputs.data)
        for sid, sample in enumerate(probs.data):
            s = '%d' % ((bid * 16)+sid)
            for prob in sample:
                s += ',%f'%prob
            print(s, file=ofile)

    test_loss /= len(testloader)
    # расчитываем accuracy
    accuracy = {}
    avg_accuracy = 0
    for i in range(10):
        accuracy[classes[i]] = 100 * class_correct[i] / class_total[i]
        avg_accuracy += accuracy[classes[i]]

    print("Final cross entropy loss: %0.5f"%test_loss, "Final accuracy: %0.3f"%(avg_accuracy/10) )

if __name__ == '__main__':
    (options, args) = parser.parse_args()
    print(options.model)
    if options.model is None or not os.path.exists( options.model ):
        print ('Model file does not exist or empty')
        exit(1)
    eval(options)