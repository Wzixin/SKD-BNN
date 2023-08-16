import argparse
import os
import time
import logging

import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import resnet

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from SKD_BNN import KL_SoftLabelloss
from SKD_BNN import Spatial_Channel_loss
from SKD_BNN import Line_CosSimloss
from SKD_BNN import Save_Label_bank
from SKD_BNN import Up_Label_bank
from SKD_BNN import setup_my_seed
from SKD_BNN import tSNE_Show
from SKD_BNN import tSNE_data_Maker
from SKD_BNN import Clustering
from mydataset import imagenet_dataloaders

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for Image1000 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data', default='/home/wzx/myData/ImageNet2012/', help='path to dataset')

parser.add_argument('--epochs', default=128, type=int, metavar='N',
                    help='number of total epochs to run (default: 128')
parser.add_argument('--warm_up_epochs', default=0, type=int, metavar='N',
                    help='warm_up_epochs to run (default: 8 - 0')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1 - 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum ！！！ (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--num_classes', default=1000, type=int, metavar='Num_Classes')
parser.add_argument('--seed', default=2023, type=int, metavar='Seed', help='setup my seed (default: 2023)')

parser.add_argument('--Feature_alpha', default=1.0, type=float, metavar='Feature_alpha',
                    help='(default: 1.0)')
parser.add_argument('--Label_beta', default=0.05, type=float, metavar='label_beta',
                    help='(default: 0.05)')
parser.add_argument('--temperature', default=4.0, type=float, metavar='temperature',
                    help='(default: 4.0) Do not need change')
parser.add_argument('--Cluster_gamma', default=0.01, type=float, metavar='cluster_gamma',
                    help='(default: 0.01)')

parser.add_argument('--resume', default='/home/wzx/myCode/Image-ResNet18/saveModel-2/128-checkpoint.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--Use_tSNE', dest='Use_tSNE', action='store_true',
                    help='use tSNE show the penultimate Feature')

parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='saveModel-4', type=str)
parser.add_argument('--print_freq', '-p', type=int, default=200,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_every', dest='save_every', type=int, default=8,
                    help='Saves checkpoints at every specified number of epochs')

best_prec1 = 0.


def main():
    global args, best_prec1, KD_Label_bank
    args = parser.parse_args()
    print(args)

    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    setup_my_seed(seed=args.seed)

    # record_log
    if not os.path.exists('record_log'):
        os.mkdir('record_log')
    logging.basicConfig(level=logging.INFO, filename='record_log/' + ''.join(args.arch) + '-4.log', format='%(message)s')
    logging.info(args)
    logging.info('Epoch\t''train_loss\t''val_loss\t''train_acc\t''val_acc\t')
    # Tensorboard
    SumWriter = SummaryWriter(log_dir='log-4')

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] * 0
            best_prec1 = checkpoint['best_prec1'] * 0
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    train_loader, val_loader = imagenet_dataloaders(args.batch_size, args.data, args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad is not False, model.parameters()),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warm_up_with_cosine_lr = lambda epoch: (epoch+1) / (args.warm_up_epochs+1) if epoch <= args.warm_up_epochs \
        else 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.evaluate:
        prec1, val_loss = validate(val_loader, model, criterion)
        print(' *** Prec@1 {:.3f}\t'.format(prec1))
        return

    print(model.module)

    T_min, T_max = 1e-3, 1e1
    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()

    KD_Label_bank = torch.zeros(args.num_classes, args.num_classes).cuda()

    print("\n************************ Start ************************\n")
    for epoch in range(args.start_epoch, args.epochs):
        t = Log_UP(T_min, T_max, epoch)
        if (t < 1):
            k = 1. / t
        else:
            k = torch.tensor([1]).float().cuda()

        for i in range(2):
            model.module.layer1[i].conv1.k = k
            model.module.layer1[i].conv2.k = k
            model.module.layer1[i].conv1.t = t
            model.module.layer1[i].conv2.t = t

            model.module.layer2[i].conv1.k = k
            model.module.layer2[i].conv2.k = k
            model.module.layer2[i].conv1.t = t
            model.module.layer2[i].conv2.t = t

            model.module.layer3[i].conv1.k = k
            model.module.layer3[i].conv2.k = k
            model.module.layer3[i].conv1.t = t
            model.module.layer3[i].conv2.t = t

            model.module.layer4[i].conv1.k = k
            model.module.layer4[i].conv2.k = k
            model.module.layer4[i].conv1.t = t
            model.module.layer4[i].conv2.t = t

        # train for one epoch
        print('current lr: {:.6e}'.format(optimizer.param_groups[0]['lr']))
        StartEpochTime = time.time()
        train_acc, train_loss, Fmap_loss, Cluster_loss, Label_loss, KD_Label_bank = train(train_loader,
                                                                                          model,
                                                                                          criterion,
                                                                                          optimizer,
                                                                                          epoch,
                                                                                          SumWriter,
                                                                                          args,
                                                                                          KD_Label_bank)
        lr_scheduler.step()
        print('train_acc {:.3f}\t'.format(train_acc))
        print("--- Train One Epoch Time(/s) :", (time.time() - StartEpochTime))

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, model, criterion)
        print(' *** Prec@1 {:.3f}\t'.format(prec1))

        SumWriter.add_scalar("train_loss", train_loss, epoch)
        SumWriter.add_scalar("Fmap_loss", Fmap_loss, epoch)
        SumWriter.add_scalar("Label_loss", Label_loss, epoch)
        SumWriter.add_scalar("Cluster_loss", Cluster_loss, epoch)
        SumWriter.add_scalar("train_acc", train_acc, epoch)
        SumWriter.add_scalar("test_loss", val_loss, epoch)
        SumWriter.add_scalar("test_acc", prec1, epoch)
        SumWriter.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        Save_Label_bank(epoch, args.save_every, KD_Label_bank, args.save_dir)
        save_checkpoint(epoch, args.save_every, {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.save_dir)

        log_message = str(epoch) + '\t' + '{:.3f}'.format(train_loss) + '\t' + '{:.3f}'.format(
            val_loss) + '\t' + '{:.3f}'.format(train_acc) + '\t' + '{:.3f}'.format(prec1)
        logging.info(log_message)

    print('best_Prec@1 {:.3f}'.format(best_prec1))


def train(train_loader, model, criterion, optimizer, epoch, SumWriter, args, KD_Label_bank):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Fmap_losses = AverageMeter()
    Label_losses = AverageMeter()
    Cluster_losses = AverageMeter()
    top1 = AverageMeter()

    Feature_loss = Spatial_Channel_loss().cuda()
    kl_criterion = KL_SoftLabelloss().cuda()
    # Penultimate_loss = nn.SmoothL1Loss(reduction='mean').cuda()
    Penultimate_loss = Line_CosSimloss().cuda()

    tSNEdata_Maker = tSNE_data_Maker()

    Son_Label_bank = torch.zeros(args.num_classes, args.num_classes).cuda()
    True_Num = torch.zeros(args.num_classes, dtype=torch.int64).cuda()

    # switch to train mode
    model.train()
    scaler = GradScaler()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # with autocast():
        # compute output
        output, penul_Binaryout, \
        L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout, \
        L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout, \
        L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout, \
        L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout, \
        L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout, \
        L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout, \
        L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout, \
        L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout = model(input)

        L1_1_loss = Feature_loss(L1_1_Binaryout, L1_1_Realout.detach())
        L1_2_loss = Feature_loss(L1_2_Binaryout, L1_2_Realout.detach())
        L1_3_loss = Feature_loss(L1_3_Binaryout, L1_3_Realout.detach())
        L1_4_loss = Feature_loss(L1_4_Binaryout, L1_4_Realout.detach())

        L2_1_loss = Feature_loss(L2_1_Binaryout, L2_1_Realout.detach())
        L2_2_loss = Feature_loss(L2_2_Binaryout, L2_2_Realout.detach())
        L2_3_loss = Feature_loss(L2_3_Binaryout, L2_3_Realout.detach())
        L2_4_loss = Feature_loss(L2_4_Binaryout, L2_4_Realout.detach())

        L3_1_loss = Feature_loss(L3_1_Binaryout, L3_1_Realout.detach())
        L3_2_loss = Feature_loss(L3_2_Binaryout, L3_2_Realout.detach())
        L3_3_loss = Feature_loss(L3_3_Binaryout, L3_3_Realout.detach())
        L3_4_loss = Feature_loss(L3_4_Binaryout, L3_4_Realout.detach())

        L4_1_loss = Feature_loss(L4_1_Binaryout, L4_1_Realout.detach())
        L4_2_loss = Feature_loss(L4_2_Binaryout, L4_2_Realout.detach())
        L4_3_loss = Feature_loss(L4_3_Binaryout, L4_3_Realout.detach())
        L4_4_loss = Feature_loss(L4_4_Binaryout, L4_4_Realout.detach())

        Fmap_loss = L1_1_loss + L1_2_loss + L1_3_loss + L1_4_loss + \
                    L2_1_loss + L2_2_loss + L2_3_loss + L2_4_loss + \
                    L3_1_loss + L3_2_loss + L3_3_loss + L3_4_loss + \
                    L4_1_loss + L4_2_loss + L4_3_loss + L4_4_loss
        Fmap_loss = args.Feature_alpha * Fmap_loss / 16.0

        soft_target = KD_Label_bank[target].cuda()
        if epoch == 0:
            Label_loss = 0 * kl_criterion(output, soft_target.detach(), temperature=args.temperature)
        else:
            Label_loss = args.Label_beta * kl_criterion(output, soft_target.detach(), temperature=args.temperature)

        penul_Clustering = Clustering(penul_Binaryout, target, args.num_classes)
        Cluster_loss = args.Cluster_gamma * Penultimate_loss(penul_Binaryout, penul_Clustering.detach())

        loss = criterion(output, target.detach()) + Fmap_loss + Label_loss + Cluster_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        output = output.float()
        loss = loss.float()

        # ######################################### update Son label bank ##############################################
        Son_Label, Num = Up_Label_bank(output, target, args.num_classes, mode='top1')
        Son_Label_bank += Son_Label
        True_Num += Num
        # ##############################################################################################################

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        Fmap_losses.update(Fmap_loss.item(), input.size(0))
        Label_losses.update(Label_loss.item(), input.size(0))
        Cluster_losses.update(Cluster_loss.item(), input.size(0))
        prec1 = accuracy(output.data, target)
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.Use_tSNE:
            tSNE_Binarydata, tSNE_Binarylabel = tSNEdata_Maker(penul_Binaryout, target, i, args.batch_size, ceiling=10000)

        if i % args.print_freq == 0:
            print('Epoch：[{0}][{1}/{2}]'
                  'Time：{batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data：{data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss：{loss.val:.4f}({loss.avg:.4f}) '
                  'F_loss：{Fmap_loss.val:.4f}({Fmap_loss.avg:.4f}) '
                  'L_loss：{Label_loss.val:.4f}({Label_loss.avg:.4f}) '
                  'C_loss：{Cluster_loss.val:.4f}({Cluster_loss.avg:.4f}) '
                  'Prec@1：{top1.val:.3f}({top1.avg:.3f}) '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, Fmap_loss=Fmap_losses, Label_loss=Label_losses,
                        Cluster_loss=Cluster_losses, top1=top1))

    # ######################################### update KD label bank ##############################################
    Son_Label_bank = Son_Label_bank / torch.unsqueeze(True_Num, dim=1)
    KD_Label_bank = Son_Label_bank
    # ##############################################################################################################

    if args.Use_tSNE and (epoch+1) == args.epochs:
        tSNE_Show(tSNE_Binarydata, tSNE_Binarylabel, name='Penultimate Feature of Binary')

    if (epoch + 1) % 100 == 0:
        image = vutils.make_grid(input[2].float(), normalize=True)
        SumWriter.add_image("image", image, epoch)

        L1_1_Binaryout = vutils.make_grid(L1_1_Binaryout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L1_1_Binaryout", L1_1_Binaryout, epoch)
        L1_1_Realout = vutils.make_grid(L1_1_Realout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L1_1_Realout", L1_1_Realout, epoch)

        L1_6_Binaryout = vutils.make_grid(L1_4_Binaryout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L1_6_Binaryout", L1_6_Binaryout, epoch)
        L1_6_Realout = vutils.make_grid(L1_4_Realout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L1_6_Realout", L1_6_Realout, epoch)

        L2_1_Binaryout = vutils.make_grid(L2_1_Binaryout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L2_1_Binaryout", L2_1_Binaryout, epoch)
        L2_1_Realout = vutils.make_grid(L2_1_Realout[2].float().detach().cpu().unsqueeze(dim=1), nrow=32, normalize=True)
        SumWriter.add_image("L2_1_Realout", L2_1_Realout, epoch)

    return top1.avg, losses.avg, Fmap_losses.avg, Cluster_losses.avg, Label_losses.avg, KD_Label_bank


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input = input.cuda()

            # compute output
            output, penul_Binaryout, \
            L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout, \
            L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout, \
            L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout, \
            L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout, \
            L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout, \
            L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout, \
            L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout, \
            L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout = model(input)

            loss = criterion(output, target.detach())

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1 = accuracy(output.data, target)
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test：[{0}/{1}] '
                      'Time：{batch_time.val:.3f}({batch_time.avg:.3f}) '
                      'Loss：{loss.val:.4f}({loss.avg:.4f}) '
                      'Prec@1：{top1.val:.3f}({top1.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


def save_checkpoint(epoch, save_every, state, is_best, save):
    """
    Save the training model
    """
    filename = os.path.join(save, str(epoch + 1) + '-checkpoint.th')
    best_filename = os.path.join(save, 'best-checkpoint.th')
    if (epoch + 1) % save_every == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
