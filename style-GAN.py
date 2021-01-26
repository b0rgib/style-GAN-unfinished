from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import Image, ImageFile

dataset = dset.ImageFolder(root="Cubism",
                           transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(256),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
real_batch = next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorF(nn.Module):
    def __init__(self):
        super(GeneratorF, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(256,256),
            nn.Linear(256,256),
            nn.Linear(256,256),
            nn.Linear(256,256),
            
        )

    def forward(self, input):
        return self.main(input)
mapF=GeneratorF()
print(mapF)


class CustomNoise(nn.Module):
    def __init__(self):
        super(CustomNoise, self).__init__()
    def forward(self, input):
        return input + (0.1**0.5)*torch.randn(5, 10, 20)*sqrt(0.1)


class AdaIN(nn.Module):
    def __init__(self,depth):
        super(AdaIN, self).__init__()
        self.depth=depth
        self.main = nn.Sequential(
            nn.Linear(256,depth*2)
        )

    def forward(self, input):
        return self.main(input)


class GeneratorG(nn.Module):
    def __init__(self):
        super(GeneratorG, self).__init__()
        self.main = nn.Sequential(
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 512, 256, 4, 1, 0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 32, 16, 4, 2, 1, bias=True),
            nn.ReLU(True),
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 16, 8, 4, 2, 1, bias=True),
            nn.ReLU(True),
            CustomNoise(),
            AdaIN(),
            nn.ConvTranspose2d( 8, 3, 4, 2, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


netG = GeneratorG().to(device)
netG.apply(weights_init)
print(netG)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)


criterion = nn.BCELoss()
constant = torch.ones(32, 512, 4, 4, device=device)
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(100):
    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 256, 1, 1, device=device)
        mapping=mapF(noise)
        fake = netG(constant)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label)  
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, 100, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 500 == 0) or ((epoch == 99) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

fig = plt.figure(figsize=(10,10))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True,)

HTML(ani.to_jshtml())






