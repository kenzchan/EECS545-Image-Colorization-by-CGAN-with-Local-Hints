import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse

from gan_model import *
from utils import *



parser = argparse.ArgumentParser(description='LOCAL HINTS NETWORK CGAN')
parser.add_argument('path', type=str,
                    help='Root path for dataset')
parser.add_argument('--dataset', type=str,
                    help='which dataset?', choices=['imgnet','bob'])
parser.add_argument('--batch_size', default=50, type=int,
                    help='Batch size: default 50')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate for optimizer')
parser.add_argument('--num_epoch', default=20, type=int,
                    help='Number of epochs')
parser.add_argument('--alpha', default=100, type=int,
                    help='Alpha for L1 Loss')
parser.add_argument('-s','--save', action="store_true",
                    help='Save model?')
parser.add_argument('--gpu', default=0, type=int,
                    help='Which GPU to use?')

def main():
    global args, date
    args = parser.parse_args()
    date = '040421'

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    model_G = ConvGen()
    model_D = ConvDis()

    start_epoch_G = start_epoch_D = 0
    start_epoch = 0

    model_G.cuda()
    model_D.cuda()

    # loss function
    global criterion
    criterion = nn.BCELoss()
    global criterion2
    criterion2 = nn.MSELoss()
    global L1
    L1 = nn.L1Loss()

    #############################################################################
    # dataset: Change you dataset and path                                      #
    #############################################################################
    data_root = args.path
    dataset = args.dataset
    if dataset == 'imgnet':
        from load_data import Imgnet_Dataset as myDataset
    else:
        raise ValueError('dataset type not supported')

    data_train = myDataset(data_root, mode='train',
                           shuffle=True
                           )

    train_loader = data.DataLoader(data_train,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=4)

    data_val = myDataset(data_root, mode='test',
                         shuffle=True
                         )

    val_loader = data.DataLoader(data_val,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    global val_bs
    val_bs = val_loader.batch_size

    # set up plotter, path, etc.
    global iteration, print_interval, plotter, plotter_basic
    iteration = 0
    print_interval = 5
    plotter = Plotter_GAN_TV()
    plotter_basic = Plotter_GAN()

    global img_path
    size = '224'
    img_path = 'img/%s/GAN_%s%s_%dL1_bs%d_%s_lr%s/' \
               % (date, args.dataset, size, args.alpha, args.batch_size, 'Adam', str(args.lr))
    model_path = 'model/%s/GAN_%s%s_%dL1_bs%d_%s_lr%s/' \
               % (date, args.dataset, size, args.alpha, args.batch_size, 'Adam', str(args.lr))
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # start loop
    start_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)
        
        #########################################################################
        #Design Learning Rate here or use weight decay                          #
        #########################################################################
        if epoch<12:
            lr_p=args.lr
        elif epoch<25:
            lr_p=args.lr/5
        elif epoch<40:
            lr_p=args.lr/10
        else:
            lr_p=args.lr/20

        optimizer_G = optim.Adam(model_G.parameters(),
                                 lr=lr_p, betas=(0.5, 0.999),
                                 eps=1e-8)
        optimizer_D = optim.Adam(model_D.parameters(),
                                 lr=lr_p, betas=(0.5, 0.999),
                                 eps=1e-8)

        
        if epoch == 0:
            val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch=-1)
        # train
        train_errG, train_errD = train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration)
        # validate
        val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch)

        plotter.train_update(train_errG, train_errD)
        plotter.val_update(val_lerrG, val_errD)
        plotter.draw(img_path + 'train_val.png')

        if args.save:
            print('Saving check point')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_G.state_dict(),
                             'optimizer': optimizer_G.state_dict(),
                             },
                             filename=model_path+'G_epoch%d.pth.tar' \
                             % epoch)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_D.state_dict(),
                             'optimizer': optimizer_D.state_dict(),
                             },
                             filename=model_path+'D_epoch%d.pth.tar' \
                             % epoch)


def train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration):
    errorG = AverageMeter() # will be reset after each epoch
    errorD = AverageMeter() # will be reset after each epoch
    errorG_basic = AverageMeter() # basic will be reset after each print
    errorD_basic = AverageMeter() # basic will be reset after each print
    errorD_real = AverageMeter()
    errorD_fake = AverageMeter()
    errorG_GAN = AverageMeter()
    errorG_R = AverageMeter()

    model_G.train()
    model_D.train()

    real_label = 1
    fake_label = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())

        ########################
        # update D network     #
        ########################
        # train with real
        model_D.zero_grad()
        output = model_D(target)
        label = torch.FloatTensor(target.size(0)).fill_(real_label).cuda()
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        fake, interaction_p =  model_G(data)

        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)
        errD_fake.backward()
        D_G_x1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizer_D.step()

        ########################
        # update G network     #
        ########################
        model_G.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        errG_GAN = criterion(torch.squeeze(output), labelv)
        errG_L1 = L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))
        errG_interaction = criterion2(interaction_p, data[:, 1:4, :, :])

        errG = errG_GAN + args.alpha * errG_L1 + 800 * args.alpha * errG_interaction
        errG.backward()
        D_G_x2 = output.data.mean()
        optimizer_G.step()

        # store error values
        errorG.update(errG.data.item(), target.size(0), history=1)
        errorD.update(errD.data.item(), target.size(0), history=1)
        errorG_basic.update(errG.data.item(), target.size(0), history=1)
        errorD_basic.update(errD.data.item(), target.size(0), history=1)
        errorD_real.update(errD_real.data.item(), target.size(0), history=1)
        errorD_fake.update(errD_fake.data.item(), target.size(0), history=1)

        errorD_real.update(errD_real.data.item(), target.size(0), history=1)
        errorD_fake.update(errD_fake.data.item(), target.size(0), history=1)
        errorG_GAN.update(errG_GAN.data.item(), target.size(0), history=1)
        errorG_R.update(errG_L1.data.item(), target.size(0), history=1)


        if iteration % print_interval == 0:
            print('Epoch%d[%d/%d]: Loss_D: %.4f(R%0.4f+F%0.4f) Loss_G: %0.4f(GAN%.4f+R%0.4f) D(x): %.4f D(G(z)): %.4f / %.4f' \
                % (epoch, i, len(train_loader),
                errorD_basic.avg, errorD_real.avg, errorD_fake.avg,
                errorG_basic.avg, errorG_GAN.avg, errorG_R.avg,
                D_x, D_G_x1, D_G_x2
                ))
            # plot image
            plotter_basic.g_update(errorG_basic.avg)
            plotter_basic.d_update(errorD_basic.avg)
            plotter_basic.draw(img_path + 'train_basic.png')
            # reset AverageMeter
            errorG_basic.reset()
            errorD_basic.reset()
            errorD_real.reset()
            errorD_fake.reset()
            errorG_GAN.reset()
            errorG_R.reset()

        iteration += 1

    return errorG.avg, errorD.avg


def validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch):
    errorG = AverageMeter()
    errorD = AverageMeter()

    model_G.eval()
    model_D.eval()

    real_label = 1
    fake_label = 0

    for i, (data, target) in enumerate(val_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        ########################
        # D network            #
        ########################
        # validate with real
        output = model_D(target)
        label = torch.FloatTensor(target.size(0)).fill_(real_label).cuda()
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)

        # validate with fake

        fake, interaction_p =  model_G(data)
        
        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)

        errD = errD_real + errD_fake

        ########################
        # G network            #
        ########################
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        errG_GAN = criterion(torch.squeeze(output), labelv)
        errG_L1 = L1(fake.view(fake.size(0),-1), target.view(target.size(0),-1))
        errG_interaction = criterion2(interaction_p, data[:, 1:4, :, :])
        
        errG = errG_GAN + args.alpha * errG_L1 + 800 * args.alpha * errG_interaction

        errorG.update(errG.data.item(), target.size(0), history=1)
        errorD.update(errD.data.item(), target.size(0), history=1)

        if i == 0:
            vis_result(data.data[:, 0, :, :], data.data[:, 1:4, :, :], target.data, fake.data, epoch)

        if i % 50 == 0:
            print('Validating Epoch %d: [%d/%d]' \
                % (epoch, i, len(val_loader)))

    print('Validation: Loss_D: %.4f Loss_G: %.4f '\
        % (errorD.avg, errorG.avg))

    return errorG.avg, errorD.avg

def vis_result(data, mask0, target, output, epoch):
    img_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data[i]), 0).cpu().numpy()
        mask = mask0[i].cpu().numpy()
        raw = target[i].cpu().numpy()
        pred = output[i].cpu().numpy()

        mask_rgb = (np.transpose(mask, (1,2,0)).astype(np.float64) + 1) / 2.
        raw_rgb = (np.transpose(raw, (1,2,0)).astype(np.float64) + 1) / 2.
        pred_rgb = (np.transpose(pred, (1,2,0)).astype(np.float64) + 1) / 2.

        grey = np.transpose(l, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, mask_rgb, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(36,27))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()

if __name__ == '__main__':
    main()
