import os
import cv2
import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models
from torchvision import  transforms

from datasets import *
from utils import *
from Generator import *
from Discriminator import *
from loss import *
from model_histoformer import *

# Training settings
parser = argparse.ArgumentParser(description='histogram_network')
# global settings
parser.add_argument('--epochs', type=int, default=150, help='the starting epoch count')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

# parser.add_argument('--train_dir', type=str, default ='./data/UIEB/train',  help='dir of train data')
# parser.add_argument('--gt_dir', type=str, default ='./data/UIEB/train_gt',  help='dir of gt data')
# parser.add_argument('--val_dir', type=str, default ='./data/UIEB/test',  help='dir of test data (use test data as val)')

# args for Histoformer
parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')

parser.add_argument('--save_dir', type=str, default ='./checkpoints/onlyinter/',  help='save dir')
parser.add_argument('--save_image_dir', type=str, default ='./results/',  help='save image dir')
# parser.add_argument('--weight_gan', type=float, default=0.4, help='the weight of gan')

opt = parser.parse_args()

### Data ###
trainloader= get_training_set()
valloader= get_val_set()
testloader= get_test_set()

### Model ###
model = Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF')

### Cuda/GPU ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

model = model.cuda()
net_d = Discriminator().cuda()
net_g = Generator().cuda()
net_d.apply(weights_init_normal)
net_g.apply(weights_init_normal)

#Load model
# model = torch.load('/home/roger/rong/checkpoints/onlyinter/Histoformer-PQR_200_modifyloss.pth')
# net_d = torch.load('/home/roger/rong/checkpoints/onlyinter/Histoformer-PQR_netD_200_modifyloss.pth')
# net_g = torch.load('/home/roger/rong/checkpoints/onlyinter/Histoformer-PQR_netG_200_modifyloss.pth')

### Optimizer ###
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay) #
optimizer_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_g = torch.optim.Adam(net_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))

### Loss ###
criterionGAN = GANLoss().cuda()
criterionL1 = nn.L1Loss().cuda()
perceploss = VGG19_PercepLoss().cuda()
contentloss = VGG19_Content().cuda()#.eval()

weight_gan = 0.4

### Train ###

best_loss = 1e10
best_loss_v = 1e10
Loss_avg=[]
Loss_avg_v=[]


for e in range(opt.epochs):
    total_loss_g = 0
    torch.cuda.empty_cache()
    model.train()
    for i, (input_img, label_img, ori_img, hs_img) in enumerate(trainloader): #, hs_img
        # print('i',i)
        input_img = input_img.to(device)
        label_img = label_img.to(device)

        optimizer.zero_grad() #retain_graph=True
        pred_img  = model(input_img)
        # exit()

        loss_list=[]
        percep_loss_list=[]
        hist_img_list=[]
        gt_img_list=[]

        R_out = pred_img[:,0]
        G_out = pred_img[:,1]
        B_out = pred_img[:,2]
        R_labels = label_img[:,0]
        G_labels = label_img[:,1]
        B_labels = label_img[:,2]
        R_loss = L2_histo(R_out,R_labels)
        G_loss = L2_histo(G_out,G_labels)
        B_loss = L2_histo(B_out,B_labels)

        for j in range(len(ori_img)):
            RGB_hs_img0, gt0 = hist_match(ori_img[j],hs_img[j], R_out[j], G_out[j], B_out[j]) #RGB_hs_img:numpy (460,620,3)

            RGB_hs_img = transforms.ToTensor()(RGB_hs_img0)
            RGB_hs_img = RGB_hs_img.unsqueeze(0).cuda() #torch.Size([1, 3, 460, 620])
            gt = transforms.ToTensor()(gt0)
            gt = gt.unsqueeze(0).cuda()

            hist_img_list.append(RGB_hs_img0)
            gt_img_list.append(gt0)
            
            loss_mae = criterionL1(RGB_hs_img, gt)
            loss_percep = perceploss(RGB_hs_img,gt)

            loss_list.append(loss_mae)
            percep_loss_list.append(loss_percep) ###
        
        mae_loss = sum(loss_list)
        mae_loss = mae_loss/len(ori_img)
            
        percep_loss = sum(percep_loss_list) ###
        percep_loss = percep_loss/len(ori_img) ###

        RGB_loss = (2*R_loss)+(0.5*G_loss)+(1*B_loss)
        loss = torch.mean(RGB_loss) 

        loss = loss + (0.6*mae_loss) + (0.6*percep_loss)
            
        loss.backward()
        optimizer.step()
            
        for k in range(len(hist_img_list)):


            RGB_hs_img0 = hist_img_list[k]
            gt0 = gt_img_list[k]

            RGB_hs_img1 = align_to_four(RGB_hs_img0)
            RGB_hs_img1 = npTOtensor(RGB_hs_img1)
            gt1 = align_to_four(gt0)
            gt1 = npTOtensor(gt1)

            ### forward
            real_a = RGB_hs_img1
            real_b = gt1

            # print('real_a.shape:',real_a.shape) # ([1, 3, 460, 620])
            # print('real_b.shape:',real_b.shape)
            fake_b = net_g(real_a)
            fake_b.data.clamp_(-1, 1)
                
            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()
        
            # train with fake
            pred_fake = net_d(fake_b) #,real_a
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            pred_real = net_d(real_b)
            loss_d_real = criterionGAN(pred_real, True)
        
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward(retain_graph=True) #retain_graph=True
       
            optimizer_d.step()
                
            ######################
            # (2) Update G network
            ######################
                
            content_loss1 = contentloss(fake_b,real_b,'relu1_1')
            content_loss2 = contentloss(fake_b,real_b,'relu1_2')
            content_loss3 = contentloss(fake_b,real_b,'relu2_1')
            content_loss4 = contentloss(fake_b,real_b,'relu2_2')
            loss_content = content_loss1 + content_loss2 + content_loss3 + content_loss4

                
            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            pred_fake = net_d(fake_b)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) 
            # print('fake_b',fake_b.shape,real_b.shape)
            loss_g = ((loss_g_gan + loss_g_l1)*weight_gan + loss_content*0.3) #+ mae_percep_loss*(1-weight_gan)
            total_loss_g += loss_g.item()

            loss_g.backward()
            optimizer_g.step()

        loss_gan = torch.tensor(total_loss_g)/len(ori_img)

        loss =  loss + loss_gan #

        # print('i',i)
        if i%20==0:
            print('epoch: {} , batch: {}, loss: {}'.format(e + 1+200, i + 1, loss.data))


    if (e+1)%10 == 0 or (e+1+200)>150 or e==0:
        torch.save({'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()
        }, os.path.join(opt.save_dir,"Histoformer-PQR_{}_modifyloss.pth".format(e+1))) 
                
        torch.save({'state_dict': net_g.state_dict(),
        'optimizer' : optimizer_g.state_dict()
        }, os.path.join(opt.save_dir,"Histoformer-PQR_netG_{}_modifyloss.pth".format(e+1)))   

        torch.save({'state_dict': net_d.state_dict(),
        'optimizer' : optimizer_d.state_dict()
        }, os.path.join(opt.save_dir,"Histoformer-PQR_netD_{}_modifyloss.pth".format(e+1)))   