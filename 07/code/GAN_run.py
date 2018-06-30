import os
import torchvision.transforms as transforms
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform as sk_transform
from tqdm import tqdm
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim as optim


# configs for run
#----------------
class Config:
    pass

config = Config()

config.root_directory= './img_celeba'

config.beta1 = 0.6
config.momentum = 0.99
config.D_lr = 1e-5
config.G_lr = 1e-4

config.batch_size = 64
config.num_workers = 8
config.num_epochs = 100

config.noise_size = 40
config.print_freq = 100
config.resize_shape = (64, 64)
#----------------

# class for generating data
class DatasetObject(Dataset):

    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.files = os.listdir(root_directory)

    def __getitem__(self, idx):
        return skimage_transform.resize(io.imread(os.path.join(self.root_directory, 
                                                  self.files[idx])),
                                                  config.resize_shape, mode='constant').transpose((2, 0, 1)) #RGB for torch
    def __len__(self):
        return len(self.files)

# creating dataloader from data object 
def create_dataloader(config):
    dataset = ImageDataset(config.root_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
    return dataloader

# simple generator 
class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(config.noise_size, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 3*config.resize_shape[0]*config.resize_shape[1]),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x).view(-1, 3, config.resize_shape[0], config.resize_shape[1]) # fix size

# simple dicriminator
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(3*config.resize_shape[0]*config.resize_shape[1], 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, 3*config.resize_shape[0]*config.resize_shape[1])
        return self.model(x)

# create dataloader
dataloader = create_dataloader(config)

#create generator and discriminator
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# create optimizers 
optim_Gen = optim.Adam(params=generator.parameters(), lr=config.G_lr, betas=(config.beta1, 0.99))
optim_Discr = optim.Adam(params=discriminator.parameters(), lr=config.D_lr, betas=(config.beta1, 0.99))

# create loss
criterion = nn.MSELoss()

for epoch in range(config.num_epochs):

    label_for_real = 1
    label_for_fake = 0

    for iteration, images in tqdm(enumerate(dataloader)):

        #######
        # Discriminator stage: maximize log(D(x)) + log(1 - D(G(z)))
        #######
        discriminator.zero_grad()

        # real part
        input = Variable(images.float()).cuda()
        label = Variable(torch.FloatTensor(images.shape[0],
                                           1).fill_(label_for_real)).cuda()
        output = discriminator(input)

        errD_x = criterion(output, label)
        errD_x.backward()

        # fake part
        noise = Variable(torch.FloatTensor(images.shape[0], config.noise_size).normal_(0, 1)).cuda()
        fake = (generator(noise) + 1) / 2
        img = fake.cpu().data[0].numpy().transpose(1, 2, 0)

        label.data.fill_(label_for_fake)
        output = discriminator(fake.detach())

        errD_z = criterion(output, label)
        errD_z.backward()

        optim_Discr.step()

        #######
        # Generator stage: minimize ||Ig - Ir||^2
        #######
        generator.zero_grad()
        label.data.fill_(label_for_real)

        output = discriminator(fake)

        errG = criterion(output, label)
        errG.backward()

        optim_Gen.step()

    print(img.min(), img.max())
    plt.imsave('{}_GAN.png'.format(epoch), img, format='png')
