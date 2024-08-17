import os
import cv2
import numpy as np
import skimage.io
# import skimage.viewer
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
from torchvision import models

from datasets import *
from utils import *
from Generator import *
from model_histoformer import *

import time

parser = argparse.ArgumentParser(description='histogram_network')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='the starting epoch count')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

# args for Histoformer
parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='TwoDCFF', help='TwoDCFF/ffn token mlp')

parser.add_argument('--save_dir', type=str, default ='./checkpoints_new/',  help='save dir')
parser.add_argument('--save_image_dir', type=str, default ='./results/',  help='save image dir')

opt = parser.parse_args()


#
model = Histoformer(embed_dim=opt.embed_dim,token_projection='linear',token_mlp='TwoDCFF').cuda()
net_g = Generator().cuda()
checkpoint = torch.load(os.path.join(opt.save_dir,'Histoformer-PQR_288_modifyloss.pth'))
checkpoint_net_g= torch.load(os.path.join(opt.save_dir,'Histoformer-PQR_netG_288_modifyloss.pth'))
# # 讀取模型權重
model.load_state_dict(checkpoint['state_dict'])
net_g.load_state_dict(checkpoint_net_g['state_dict'])

testloader= get_test_set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_list = []
with torch.no_grad():  #如果沒有這行，那下面在取值的時候要用.detach().numpy()
    for i,(inputs, labels, ori_img) in enumerate(testloader):
        filename = os.path.basename(ori_img[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        net_g = net_g.to(device)
        net_g.eval()
        # t0 = time.time()
        pred_img = model(inputs)

        R_out = pred_img[:,0]
        G_out = pred_img[:,1]
        B_out = pred_img[:,2]

        RGB_hs_img, _ = hist_match(ori_img[0], ori_img[0], R_out, G_out, B_out)
        
        # print('type(RGB_hs_img):', type(RGB_hs_img)) # <class 'numpy.ndarray'>
        # print('RGB_hs_img.shape:', RGB_hs_img.shape) # (300, 400, 3)
        
        RGB_hs_img = align_to_four(RGB_hs_img)
        RGB_hs_img = npTOtensor(RGB_hs_img)
        # print('RGB_hs_img',RGB_hs_img.shape)
        img_gan = net_g(RGB_hs_img)
        # print('img_gan:',img_gan.shape)
        img_gan = img_gan.cpu().data
        img_gan = img_gan.numpy().transpose((0, 2, 3, 1))
        img_gan = img_gan[0, :, :, :]*255.
        print(ori_img[0][53:],img_gan.shape)
        cv2.imwrite(os.path.join(opt.save_image_dir, filename), img_gan) #[34:]可能會要根據路徑的長度做更改
