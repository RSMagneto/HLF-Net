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
    # pdb.set_trace()
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

def getBatch_rec(ms_highfrequency, pan_highfrequency, gt_highfrequency,ms_Gaussian,pan_Gaussian,gt_Gaussian,gt_data, bs):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=bs)
    ms_low = ms_highfrequency[batchIndex, :, :, :]
    pan_low = pan_highfrequency[batchIndex, :, :, :]
    gt_low = gt_highfrequency[batchIndex, :, :, :]
    ms_high = ms_Gaussian[batchIndex, :, :, :]
    pan_high = pan_Gaussian[batchIndex, :, :, :]
    gt_high = gt_Gaussian[batchIndex, :, :, :]
    gt_train = gt_data[batchIndex, :, :, :]
    return ms_low,pan_low,gt_low,ms_high, pan_high, gt_high,gt_train
#
def getTest_rec(ms_Gaussian,pan_Gaussian,ms_highfrequency, pan_highfrequency,gt_data):
    N = gt_data.shape[0]
    batchIndex = np.random.randint(0, N, size=1)
    ms_low_test = ms_Gaussian[batchIndex, :, :, :]
    pan_low_test = pan_Gaussian[batchIndex, :, :, :]
    ms_high_test = ms_highfrequency[batchIndex, :, :, :]
    pan_high_test = pan_highfrequency[batchIndex, :, :, :]
    gt_test = gt_data[batchIndex, :, :, :]
    return ms_low_test,pan_low_test,ms_high_test, pan_high_test, gt_test

def convert_image_np(inp,opt):
    inp=inp[-1,:,:,:]
    # pdb.set_trace()
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
        # pdb.set_trace()
        x = torch.nn.functional.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


def SID(MS,PANMS,opt):
    b,d,n,m=PANMS.shape
    p=torch.zeros_like(PANMS)
    q=torch.zeros_like(PANMS)
    for i in range(d):
        p[:,i,:,:]=(MS[:,i,:,:])/torch.sum(MS,dim=1)
        q[:,i,:,:]=(PANMS[:,i,:,:])/torch.sum(PANMS,dim=1)
    S=torch.zeros([b,n,m],device=opt.device)
    N=torch.zeros([b,n,m],device=opt.device)
    for i in range(d):
        S=(p[:,i,:,:]*torch.log(p[:,i,:,:]/q[:,i,:,:]))+S
        N = (q[:, i, :, :] * torch.log(q[:, i, :, :] / p[:, i, :, :])) + N
    D=N+S
    sumD=torch.sum(D)/(n*m)
    return sumD