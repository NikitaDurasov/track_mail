import torchvision.transforms as transforms
from skimage import transform as skimage_transform
from tqdm import tqdm
import pickle as pkl
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from skimage import io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# configs for run
#----------------
class Config:
    pass

config = Config()

config.root_directory = './img_celeba'

config.beta1 = 0.6
config.momentum = 0.99
config.D_lr = 1e-5
config.G_lr = 1e-4

config.batch_size = 32
config.num_workers = 7
config.num_epochs = 50

config.noise_size = 512
config.print_freq = 100
config.resize_shape = (153, 153)
#----------------

# object for data generating 
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
    dataset = DatasetObject(config.root_directory)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
    return dataloader

class LSGAN_Generator(nn.Module):
    def __init__(self):
        super(LSGAN_Generator, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 3, stride=1))

        self.fully_conn = nn.Linear(config.noise_size, 7 * 7 * 256)
        self.fc_bn = nn.BatchNorm1d(7 * 7 * 256)
        self.relu = nn.ReLU()

    def forward(self, x):
    	x = self.fc_bn(self.fully_conn(x))
        x = self.relu().view(-1, 256, 7, 7)
        x = self.deconv(x)
        return F.tanh(x)

class LSGAN_Discriminator(nn.Module):
    def __init__(self):
        super(LSGAN_Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 5, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fully_conn = nn.Linear(512 * 6 * 6, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 6 * 6)
        return self.fully_conn(x)

# create dataloader
dataloader = create_dataloader(config)

# create generator and discriminator
generator = LSGAN_Generator().cuda()
discriminator = LSGAN_Discriminator().cuda()

# create optimazers for both
optimizer_Gen = optim.Adam(params=generator.parameters(), lr=config.G_lr, betas=(config.beta1, 0.999))
optimizer_Discr = optim.Adam(params=discriminator.parameters(), lr=config.D_lr, betas=(config.beta1, 0.999))

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
        label = Variable(torch.FloatTensor(images.shape[0], 1).fill_(label_for_real)).cuda()

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

        optimizer_Discr.step()

        #######
        # Generator stage: minimize ||Ig - Ir||^2
        #######
        generator.zero_grad()
        label.data.fill_(label_for_real)

        output = discriminator(fake)

        errG = criterion(output, label)
        errG.backward()

        optimizer_Gen.step()

    plt.imsave('{}_LSGAN.png'.format(epoch), img, format='png')