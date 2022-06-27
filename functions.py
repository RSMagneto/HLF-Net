import skimage.io as skimage
import torch
import numpy as np
import pdb
from sewar.full_ref import sam
import torch.nn as nn

def matRead(data,opt):
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.type(torch.cuda.FloatTensor)
    return data

def test_matRead(data,opt):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.type(torch.cuda.FloatTensor)
    return data

def getBatch(ms,pan,gt, bs):
    N = gt.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    ms_batch = ms[batchIndex, :, :, :]
    pan_batch = pan[batchIndex, :, :, :]
    gt_batch = gt[batchIndex, :, :, :]
    return ms_batch,pan_batch,gt_batch

def getTest(ms,pan,gt):
    N = gt.shape[0]
    batchIndex = np.random.randint(0, N, size=1)
    ms_batch = ms[batchIndex, :, :, :]
    pan_batch = pan[batchIndex, :, :, :]
    gt_batch = gt[batchIndex, :, :, :]
    return ms_batch,  pan_batch,  gt_batch

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    inp = inp.to(torch.device('cpu'))
    inp = inp.numpy().transpose((1,2,0))
    inp = np.clip(inp,0,1)
    inp=inp*2047.
    return inp

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SAM(sr_img,hr_img):
    sr_img = sr_img.to(torch.device('cpu'))
    sr_img = sr_img.numpy()
    sr_img=sr_img[-1,:,:,:]
    hr_img = hr_img.to(torch.device('cpu'))
    hr_img = hr_img.numpy()
    hr_img = hr_img[-1, :, :, :]
    sam_value = sam(sr_img*1.0, hr_img*1.0)
    return sam_value
    
class GaussianBlurConv(nn.Module):
    def __init__(self, channels):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                  [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                  [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def __call__(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x
