"""
Ипользуем обученную модель, для того, что бы сделать предикт для конкурса
"""
import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.autograd import Variable
from cifar_resnet import Net
from optparse import OptionParser
from torch.utils.data import DataLoader
from custom_dataset import  CustomC100Dataset
import torchvision.transforms as transforms
from tqdm import *

# разбираем аргументы коммандной строки
parser = OptionParser("Train cifar10 neural network")

parser.add_option("-i", "--input", dest="input", default='./cifar_100_custom/test.npy',
                  help="Cifar data directory")  # рутовый каталог откуда беруться данные

parser.add_option('-m',"--model", dest="model",
                  help="Model base path ") # путь к файлу модели

parser.add_option("-o", "--out", dest="out", type='string', default='solution.csv',
                  help="Path to tensorboard log")   # куда складывать результаты

def eval(options):
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm')
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    # Создаем модель, нужно сделать иплементацию
    print("Creating model...")
    net = Net().cuda()
    net.eval()
    # Критерий кросс энтропия проверим, что у нас вск сходится
    criterion = nn.CrossEntropyLoss().cuda()

    # загружаем сеть
    cp_dic = torch.load(options.model)
    net.load_state_dict(cp_dic['net'])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # нормализация посчитанная по всему множеству
    ])

    # данные для теста
    testset = CustomC100Dataset(options.input, 'test', 10000, transform=transform_test)
    testloader = DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    print('Test model: ')

    ofile = open(options.out, 'w')
    print("Id,Prediction", file=ofile)

    for bid, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        inputs, labels = data

        # получаем переменные Variable
        inputs, labels = Variable(inputs, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
        outputs = net(inputs)

        # печатаем для каждого пункта предсказанный класс
        max_probs, classes_id = torch.max(softmax(outputs.data).data, 1)
        for sid, class_id in enumerate(classes_id):
            s = '%d' % ((bid * 16)+sid) + ',%d'%class_id
            print(s, file=ofile)


if __name__ == '__main__':
    (options, args) = parser.parse_args()
    if options.model is None or not os.path.exists( options.model ):
        print ('Model file does not exist or empty')
        exit(1)
    eval(options)