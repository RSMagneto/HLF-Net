import argparse
import model
import torch
import torch.nn as nn
import pdb
import functions
import matplotlib.pyplot as plt
import math
from time import *
import os
import skimage.io
import numpy
import scipy.io as sio
import copy
import pdb
from tqdm import tqdm
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', require=True)
    parser.add_argument('--test_dir', help='testing_data', require=True)
    parser.add_argument('--outputs_dir', help='output model dir', require=True)
    parser.add_argument('--channels',help='numble of image channel',default=5)
    parser.add_argument('--batchSize',default=6)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda'))
    parser.add_argument('--epoch', default=300)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr',type=float,default=0.0001,help='G‘s learning rate')
    parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
    parser.add_argument('--lr_decay_step', type=int, default=250)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.test_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    gaussian_conv_4 = functions.GaussianBlurConv(opt.channels - 1).to(opt.device)
    gaussian_conv_1 = functions.GaussianBlurConv(1).to(opt.device)

    net = model.poolingNet(opt.channels).to(opt.device)
#     net = model.UNet(opt.channels, opt.channels-1).to(opt.device)
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    optimizer1 = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer1,milestones=[1600],gamma=opt.gamma)

    loss = torch.nn.L1Loss()
    best_weights = copy.deepcopy(net.state_dict())
    best_epoch = 0
    best_SAM=1.0

    for i in range(opt.epoch):
        net.train()
        epoch_losses = functions.AverageMeter()
        batch_time = functions.AverageMeter()
        end = time()
        for batch_idx, (gtBatch, msBatch, panBatch) in enumerate(train_loader):
            if torch.cuda.is_available():
                msBatch, panBatch, gtBatch = msBatch.cuda(), panBatch.cuda(), gtBatch.cuda()
                msBatch = Variable(msBatch.to(torch.float32))
                panBatch = Variable(panBatch.to(torch.float32))
                gtBatch = Variable(gtBatch.to(torch.float32))
            N = len(train_loader)
            net.zero_grad()
            msBatch = torch.nn.functional.interpolate(msBatch, size=(gtBatch.shape[2], gtBatch.shape[3]),
                                                      mode='bilinear')

            lowpass=net(gaussian_conv_1(panBatch),gaussian_conv_4(msBatch))
            lowLoss = loss(lowpass,gaussian_conv_4(gtBatch))
            lowLoss.backward()
#             highpass = net(panBatch - gaussian_conv_1(panBatch), msBatch - gaussian_conv_4(msBatch))
#             highLoss=loss(highpass,gtBatch - gaussian_conv_4(gtBatch))
#             highLoss.backward()
            optimizer1.step()
            epoch_losses.update(lowLoss.item(), msBatch.shape[0])  # highLoss
            batch_time.update(time() - end)
            end = time()
            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'Loss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, lowLoss
                )   # highLoss
                print(training_state)
        print('%d epoch: loss is %.6f, epoch time is %.4f' % (i, epoch_losses.avg, batch_time.avg))
        torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'lowpass/epoch_{}.pth'.format(i)))   # highpass

        net.eval()
        epoch_SAM=functions.AverageMeter()
        with torch.no_grad():
            for j, (gtTest, msTest, panTest) in enumerate(val_loader):
                if torch.cuda.is_available():
                    msTest, panTest, gtTest = msTest.cuda(), panTest.cuda(), gtTest.cuda()
                    msTest = Variable(msTest.to(torch.float32))
                    panTest = Variable(panTest.to(torch.float32))
                    gtTest = Variable(gtTest.to(torch.float32))
                msTest = torch.nn.functional.interpolate(msTest, size=(256, 256), mode='bilinear')
                lowpass=net(gaussian_conv_1(panTest),gaussian_conv_4(msTest))
                test_SAM=functions.SAM(lowpass, gaussian_conv_4(gtTest))
#                 highpass = net(panTest - gaussian_conv_1(panTest), msTest - gaussian_conv_4(msTest))
#                 test_SAM = functions.SAM(highpass, gtTest - gaussian_conv_4(gtTest))
                if test_SAM==test_SAM:
                    epoch_SAM.update(test_SAM,lowpass.shape[0])
            print('net eval SAM: {:.6f}'.format(epoch_SAM.avg))
        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(net.state_dict())
        print('best epoch:{:.0f}'.format(best_epoch))
        scheduler1.step()
    print('net best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(opt.outputs_dir, 'lowpass/best.pth'))   # highpass
