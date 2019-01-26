import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('output', exist_ok=True)

img_shape = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128,512)
        self.fc3 = nn.Linear(512,1024 )
        self.fc4 = nn.Linear(1024,28*28)
        self.in1 = nn.BatchNorm1d(128)
        self.in2 = nn.BatchNorm1d(512)
        self.in3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.in2(self.fc2(x)),0.2)
        x = F.leaky_relu(self.in3(self.fc3(x)),0.2)
        x = F.tanh(self.fc4(x))
        return x.view(x.shape[0],*img_shape)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128,1)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.leaky_relu( self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = F.leaky_relu(self.fc3(x),0.2)
        x = F.sigmoid(self.fc4(x))
        return x


loss_func = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()


dataset = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),batch_size=64, shuffle=True)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss_func.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002,betas=(0.4,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,betas=(0.4,0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



for epoch in range(20):
    for i, (imgs, _) in enumerate(dataset):

        #ground truths
        val = Tensor(imgs.size(0), 1).fill_(1.0)
        fake = Tensor(imgs.size(0), 1).fill_(0.0)

        real_imgs = imgs.cuda()


        optimizer_G.zero_grad()

        gen_input = Tensor(np.random.normal(0, 1, (imgs.shape[0],100)))

        gen = generator(gen_input)

        #measure of generator's ability to fool discriminator
        g_loss = loss_func(discriminator(gen), val)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = loss_func(discriminator(real_imgs), val)
        fake_loss = loss_func(discriminator(gen.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 20, i, len(dataset),
                                                            d_loss.item(), g_loss.item()))
        
        total_batch = epoch * len(dataset) + i
        if total_batch % 400 == 0:
            save_image(gen.data[:25], 'output/%d.png' % total_batch, nrow=5, normalize=True)