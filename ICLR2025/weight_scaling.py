import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import TensorDataset
from numpy import dot
from numpy.linalg import norm
import pdb
import matplotlib.pyplot as plt
from torch import nn
import requests


def get_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Train MLP with different nonlinearity options")
    parser.add_argument('--nonlinearity', type=str, default='sine', choices=['sinc', 'sine', 'gauss', 'gabor'],
                        help='Choose the nonlinearity: sinc, sine, gauss, or gabor')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--scale', type=float, default=2.5)
    return parser.parse_args(args)


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels.view(-1, 1)

def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


#######################################################################
# SIREN and SincNet

class Sincactivation(nn.Module):
    def forward(self, x):
        return torch.sinc(8*x)

class Sineactivation(nn.Module):
    def forward(self, x):
        return torch.sin(30*x)


class SincMLP(nn.Module):
    def __init__(self, h=256, layers=3, scale=1):
        super().__init__()

        dim1 = 2
        self.fc1 = nn.Linear(dim1, h)
        mid_layers = []
        for _ in range(layers):
            linear_layer = nn.Linear(h, h)
            mid_layers.extend([linear_layer, Sincactivation()])
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(h, 1)
        
        self.Sincactivation = Sincactivation()

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        x = self.fc1(x)
        x = self.Sincactivation(x)
        activations = [x]
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x

class SineMLP(nn.Module):
    def __init__(self, h=512, layers=3, scale=1):
        super().__init__()

        dim1 = 2 
        self.fc1 = nn.Linear(dim1, h)
        with torch.no_grad():
            self.fc1.weight.uniform_(-1*scale/2, 1*scale/2)
        mid_layers = []
        for _ in range(layers):
            linear_layer = nn.Linear(h, h)
            with torch.no_grad():
                linear_layer.weight.uniform_(-np.sqrt(6/h)*scale/30,np.sqrt(6/h)*scale/30)
            mid_layers.extend([linear_layer, Sineactivation()])
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(h, 1)
        with torch.no_grad():
            self.fc2.weight.uniform_(-np.sqrt(6/h)/30,np.sqrt(6/h)/30)
        
        self.Sineactivation = Sineactivation()

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        x = self.fc1(x)
        x = self.Sineactivation(x)
        activations = [x]
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x
#######################################################################


#######################################################################
# GaussNet
class Gaussactivation(nn.Module):
    def forward(self, x):
        return torch.exp(-(20*x)**2)

class GaussMLP(nn.Module):
    def __init__(self, h=256, layers=3, scale=1):
        super().__init__()

        dim1 = 2 
        self.fc1 = nn.Linear(dim1, h)
        mid_layers = []
        for _ in range(layers):
            linear_layer = nn.Linear(h, h)
            mid_layers.extend([linear_layer, Gaussactivation()])
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(h, 1)
        
        self.Gaussactivation = Gaussactivation()


    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        x = self.fc1(x)
        x = self.Gaussactivation(x)
        activations = [x]
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x
#######################################################################

#######################################################################
# WIRE
class ComplexGaborActivation(nn.Module):
    def __init__(self, omega0=10, sigma0=10):
        super().__init__()
        self.omega0 = omega0
        self.sigma0 = sigma0

    def forward(self, x):
        omega = self.omega0 * x
        scale = self.sigma0 * x
        return torch.exp(1j * omega - scale.abs() ** 2)


class MLPWithGabor(nn.Module):
    def __init__(self, h=256, layers=3, omega0=10, sigma0=10, use_complex=True):
        super().__init__()

        self.use_complex = use_complex
        dim1 = 2
        self.fc1 = nn.Linear(dim1, h, dtype=torch.float)
        self.GaborActivation = ComplexGaborActivation(omega0, sigma0)
        mid_layers = []
        for _ in range(layers):
            linear_layer = nn.Linear(h, h, dtype=torch.cfloat)
            mid_layers.extend([linear_layer, self.GaborActivation])
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(h, 1, dtype=torch.cfloat)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        x = self.fc1(x)
        x = self.GaborActivation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return x.real
#######################################################################


def get_cameraman_tensor(sidelength):
    img = Image.open("kodim05_512_grayscale_1.png")
    
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
    ])
    img = transform(img)
    return img


def psnr(img, gt):
   maxval = np.max(gt)
   
   img = img / maxval
   gt = gt / maxval
   eps = 1e-8
   mse = np.maximum(0, np.mean((img - gt) ** 2))

   return np.mean(-10 * np.log10(mse+eps))

def initialization(seed = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#######################################################################
# main


image_url = "https://i.ibb.co/369F9T8/kodim05-512-grayscale.png"
image_path = "kodim05_512_grayscale_1.png"


response = requests.get(image_url)
if response.status_code == 200:
    with open(image_path, "wb") as f:
        f.write(response.content)


input = get_mgrid(512,2)
img = get_cameraman_tensor(512)
output = img.permute(1,2,0).view(-1,1)

total_steps = 150
cameraman = ImageFitting(512)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
args = get_args()
nonlinearity = args.nonlinearity
scale= args.scale
initialization(0)
    
if nonlinearity == 'sine':
    network = SineMLP(h=args.width, layers=3, scale=scale).to(device)
elif nonlinearity == 'gabor':
    network = MLPWithGabor(h=int(args.width/np.sqrt(2)), layers=3).to(device)
elif nonlinearity == 'gauss':
    network = GaussMLP(h=args.width, layers=3).to(device)
elif nonlinearity == 'sinc':
    network = SincMLP(h=args.width, layers=3).to(device)

        
optim = torch.optim.Adam(lr=0.0001, params=network.parameters())
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.to(device), ground_truth.to(device)
criterion = torch.nn.MSELoss()
for step in range(total_steps):
    model_output = network(model_input)
    loss = ((model_output - ground_truth)**2).mean()
    train_psnr = psnr(model_output.cpu().detach().numpy(), ground_truth.cpu().detach().numpy())
    if step % 10 == 0:
        print(f'step: {step}, PSNR: {train_psnr}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    if train_psnr >=50:
        break

