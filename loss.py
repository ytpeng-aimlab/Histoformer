import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class L2_histo(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y):
#         # input has dims: (Batch x Bins)
#         bins = x.size(1)
#         r = torch.arange(bins)
#         s, t = torch.meshgrid(r, r)
#         tt = t >= s
#         tt = tt.to(device)

#         cdf_x = torch.matmul(x, tt.float())
#         cdf_y = torch.matmul(y, tt.float())

#         return torch.sum(torch.square(cdf_x - cdf_y), dim=1)

def L2_histo(x, y):
    bins = x.size(1)
    r = torch.arange(bins)
    s, t = torch.meshgrid(r, r)
    tt = t >= s
    tt = tt.to(device)

    cdf_x = torch.matmul(x, tt.float())
    cdf_y = torch.matmul(y, tt.float())

    return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space """
    
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGG19_Content(nn.Module):
    """ Calculates content loss in vgg19 space """    
    
    def __init__(self, _pretrained_=True):
        super(VGG19_Content, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'1': 'relu1_1', '3': 'relu1_2', '6': 'relu2_1', '8': 'relu2_2'} # may add other layers    
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def forward(self, pred, true, layer):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)

        return torch.mean((true_f[layer]-pred_f[layer])**2)