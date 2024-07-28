import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def hist_match(image_path,image_path_hs,out_R,out_G,out_B):   
    asli = cv2.imread(image_path)
    hs = cv2.imread(image_path_hs)
    result = np.copy(asli)
    R_hist, R_bins = np.histogram(asli[:, :, 2], bins=256, range=(0, 256)) 
    G_hist, G_bins = np.histogram(asli[:, :, 1], bins=256, range=(0, 256))
    B_hist, B_bins = np.histogram(asli[:, :, 0], bins=256, range=(0, 256))
    asli_B = asli[:,:,0]
    asli_G = asli[:,:,1]
    asli_R = asli[:,:,2]
    asli_shape_B = asli_B.shape    
    asli_shape_G = asli_G.shape 
    asli_shape_R = asli_R.shape 
    asli_B = asli_B.ravel()
    asli_G = asli_G.ravel()
    asli_R = asli_R.ravel()

    values = np.array(range(0,256),dtype=np.uint8)
    try:
        o_values_B, bin_idx_B, o_counts_B = np.unique(asli_B, return_inverse=True,return_counts=True)
        o_values_G, bin_idx_G, o_counts_G = np.unique(asli_G, return_inverse=True,return_counts=True)
        o_values_R, bin_idx_R, o_counts_R = np.unique(asli_R, return_inverse=True,return_counts=True)

        o_quantiles_B = np.cumsum(o_counts_B).astype(np.float64)
        o_quantiles_B /= o_quantiles_B[-1]
        o_quantiles_G = np.cumsum(o_counts_G).astype(np.float64)
        o_quantiles_G /= o_quantiles_G[-1]
        o_quantiles_R = np.cumsum(o_counts_R).astype(np.float64)
        o_quantiles_R /= o_quantiles_R[-1]
        b_quantiles = np.cumsum((out_B.squeeze(0)*sum(B_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        b_quantiles /= b_quantiles[-1]
        g_quantiles = np.cumsum((out_G.squeeze(0)*sum(G_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        g_quantiles /= g_quantiles[-1]
        r_quantiles = np.cumsum((out_R.squeeze(0)*sum(R_hist)).cpu().detach().numpy().tolist()).astype(np.float64)
        r_quantiles /= r_quantiles[-1]
        interp_t_valuesB = np.interp(o_quantiles_B, b_quantiles, values) #, b_values
        interp_t_valuesG = np.interp(o_quantiles_G, g_quantiles, values)
        interp_t_valuesR = np.interp(o_quantiles_R, r_quantiles, values)
        result[:,:,0] = interp_t_valuesB[bin_idx_B].reshape(asli_shape_B)
        result[:,:,1] = interp_t_valuesG[bin_idx_G].reshape(asli_shape_G)
        result[:,:,2] = interp_t_valuesR[bin_idx_R].reshape(asli_shape_R)

        return result, hs
    except ValueError:
        pass

def npTOtensor(image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    
    return image

def align_to_four(img):

    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]

    return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

