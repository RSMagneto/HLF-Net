import model
import torch
import functions
import numpy
import os
from skimage import io
import argparse

def test_matRead(data):
    data=data[None, :, :, :]
    data=data.transpose(0,3,1,2)/2047.
    data=torch.from_numpy(data)
    data = data.to(torch.device('cuda:0')).type(torch.cuda.FloatTensor)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mspath', help='test lrms image name', required=True)
    parser.add_argument('--panpath', help='test hrpan image name', required=True)
    parser.add_argument('--saveimgpath', help='output model dir', required=True)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--channels',default=5)
    opt = parser.parse_args()

    poolingnet = model.poolingNet(5).to(opt.device)
    unet = model.UNet(5, 4).to(opt.device)

    poolingnet.load_state_dict(torch.load('./model/lowpass/best.pth'))
    unet.load_state_dict(torch.load('/mnt/./model/highpass/best.pth'))
    gaussian_conv_4 = functions.GaussianBlurConv(4).to(opt.device)
    gaussian_conv_1 = functions.GaussianBlurConv(1).to(opt.device)
    for msfilename in os.listdir(opt.mspath):
        num = msfilename.split('m')[0]
        print(opt.mspath + msfilename)
        ms_val = io.imread(opt.mspath + msfilename)
        ms_val = test_matRead(ms_val)
        ms_val = torch.nn.functional.interpolate(ms_val, size=(256, 256), mode='bilinear')
        panname = msfilename.split('m')[0] + 'p.tif'
        pan_val = io.imread(opt.panpath + panname)
        pan_val = pan_val[:, :, None]
        pan_val = test_matRead(pan_val)
        highpass = unet(pan_val - gaussian_conv_1(pan_val), ms_val - gaussian_conv_4(ms_val))
        lowpass = poolingnet(gaussian_conv_1(pan_val), gaussian_conv_4(ms_val))
        in_s = lowpass + highpass
        outname = opt.saveimgpath + num + '.tif'
        io.imsave(outname, functions.convert_image_np(in_s.detach(), opt).astype(numpy.uint16))

if __name__ == '__main__':
    main()
