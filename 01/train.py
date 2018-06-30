import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import torchvision.transforms as transforms
import os
from model import Net
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from optparse import OptionParser
from tqdm import *

# разбираем аргументы коммандной строки
parser = OptionParser("Train cifar10 neural network")

parser.add_option("-i", "--input", dest="input", default='./',
                  help="Cifar data root directory")  # рутовый каталог откуда беруться данные

parser.add_option('-m',"--model", dest="model", default='./model_save/final.model',
                  help="Model base path ") # базовый путь куда будет сохранятся модель и ее чекпойнты

parser.add_option('-c',"--checkpoint", dest="checkpoint",
                  help="Check point for load model and continue train")  # если продолжаем обучения с такого чекпойнта

parser.add_option('-e',"--epoch", dest="epoch", default='10', type=int,
                  help="Count epoch")  # количество эпох, которое будем обучаться

parser.add_option("-l", "--log", dest="log", type='string', default='./log',
                  help="Path to tensorboard log")   # куда складывать лог tensorboard


def adjust_learning_rate(optimizer, epoch, base_lr, loss_diff):
    """
     Реализует политику уменьшения коэффициента обучения 
     в зависимости от номера эпохи
    :param optimizer:  ссылка на класс оптимизатора сети
    :param epoch:      номер эпохи 
    :param base_lr:    базовый коэффициент обучения
    :return: 
    """
    lr = base_lr*(0.1**(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return  lr

def train(options):
    """
     Обучаем нашу модель, которую нужно реализовать в файле cifar_model.py
    :param options: 
    :return: 
    """
    base_lr = 0.01 # задаем базовый коэффициент обучения
    # список классов  cifar 10
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # иниициализация writer для записи в tensorboard
    writer = SummaryWriter(log_dir=options.log)

    #
    # тут можно сделать аугментацию
    # трансформации, шум ...
    # https://www.programcreek.com/python/example/104832/torchvision.transforms.Compose
    transform = transforms.Compose(
        [   
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(240), 
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), #нормализация данных
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])

    #
    # Загружаем данные, если данных еще нет, то нужно указать флаг download=True
    # torchvision реализует Dataset для CIFAR, MNIST, ImageNet...
    print("Loading data....")
    trainset = mnist.MNIST(options.input, download=True, train=True, transform=transform)

    # теперь можно использовать DataLoader для доступа к данным
    # Dataset, shuffle = True - доступ рандомный
    # можно загружать данные в несколько потоков, если скорость загрузки
    # меньше чем скорость обновления сети
    trainloader = DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # данные для теста
    testset = mnist.MNIST(options.input, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    # Создаем модель, нужно сделать иплементацию
    print("Creating model...")
    #net = Net().cuda()
    net = Net()

    # Критерий кросс энтропия
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    # тут создаем оптимайзер, который нужен
    learning_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-5) # 

    start_from_epoch = 0
    # Если указан чекпойнт то загружаем сеть
    if options.checkpoint is not None and os.path.exists(options.checkpoint):
        cp_dic = torch.load(options.checkpoint)
        net.load_state_dict(cp_dic['net'])
        optimizer.load_state_dict(cp_dic['optimizer'])
        start_from_epoch = cp_dic['epoch']

    print("Start train....")
    prev_test_loss = 10000000
    prev_loss_diff = 0
    for epoch in range(start_from_epoch, options.epoch):
        train_loss = 0.0

        # делаем что то с коэффициентом обучения
        epoch_lr = adjust_learning_rate(optimizer, epoch, base_lr, prev_loss_diff)

        print ('Train epoch: ', epoch)
        net.train(True)
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # получаем входы из даталоадера
            inputs, labels = data

            # оборачиваем данные в Variable
            # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            # обнуляем градиенты
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # печатаем статистику по итерации в tensorboard
            train_loss += loss.data[0]
            #
            writer.add_scalar('loss/iter_train',loss.data[0], epoch * len(trainloader) + i )

        train_loss /= len(trainloader)

        # тестируем модель после эпохи, что бы понять что у нас еще все хорошо
        net.eval()
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        print('Test epoch: ', epoch)
        for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            inputs, labels = data

            # получаем переменные Variable
            #inputs, labels = Variable(inputs, volatile=True).cuda(), Variable(labels, volatile=True).cuda()
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            
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


        test_loss /= len(testloader)
        # расчитываем accuracy
        accuracy= {}
        avg_accuracy = 0
        for i in range(10):
            accuracy[classes[i]] = 100 * class_correct[i] / class_total[i]
            avg_accuracy += accuracy[classes[i]]

        # пишем всю статистику в tensorboard
        writer.add_scalars('loss/avg_epoch_error', {'train':train_loss, 'test':test_loss}, epoch )
        writer.add_scalars('loss/class_accuracy', accuracy , epoch)
        writer.add_scalar('loss/avg_accuracy', avg_accuracy/10, epoch)

        # выводим коэффициент обучения на эпохе
        writer.add_scalar('loss/epoch_lr', epoch_lr, epoch)

        # сохраняем модель каждые 2 итерации
        if epoch %5 ==0:
            torch.save({
                    'epoch': epoch + 1,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, options.model + '_chekpoint_%03d.pth'%epoch )

        prev_loss_diff = prev_test_loss - test_loss
        prev_test_loss = test_loss
        print("-------------------------------")
        print("test_loss: " + str(prev_test_loss))
        print("-------------------------------")
    # сохраняем финальную модель
    torch.save(net.state_dict(), options.model + '.pth')

if __name__ == '__main__':
    (options, args) = parser.parse_args()
    if not os.path.exists( options.log ):
        os.mkdir(options.log)
    train(options)
