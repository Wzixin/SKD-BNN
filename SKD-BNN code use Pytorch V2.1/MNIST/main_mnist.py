
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import copy
from SKD_BNN import KL_SoftLabelloss
from SKD_BNN import Spatial_Channel_loss
from SKD_BNN import Line_CosSimloss
from SKD_BNN import Save_Label_bank
from SKD_BNN import Up_Label_bank
from SKD_BNN import setup_my_seed
from SKD_BNN import tSNE_Show
from SKD_BNN import tSNE_data_Maker
from SKD_BNN import Clustering

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', default=0,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--num_classes', default=10, type=int, metavar='Num_Classes')
parser.add_argument('--Feature_alpha', default=1.0, type=float, metavar='Feature_alpha',
                    help='(default: 1.0)')
parser.add_argument('--Label_beta', default=0.05, type=float, metavar='label_beta',
                    help='(default: 0.05)')
parser.add_argument('--temperature', default=4.0, type=float, metavar='temperature',
                    help='(default: 4.0) Do not need change')
parser.add_argument('--Cluster_gamma', default=0.01, type=float, metavar='cluster_gamma',
                    help='(default: 0.01)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/root/autodl-tmp/myData', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/root/autodl-tmp/myData', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)

        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.5)


    def forward(self, x):
        x = x.view(-1, 28 * 28)

        # First Layer
        Binaryout1, Realout1 = self.fc1(x, x.detach())
        Binaryout1 = self.bn1(Binaryout1)
        Binaryout1 = self.htanh1(Binaryout1)
        # Copy reused layers and freeze them
        realbn1 = copy.deepcopy(self.bn1)
        with torch.no_grad():
            Realout1 = realbn1(Realout1)
            Realout1 = F.hardtanh(Realout1)

        # Next Layer
        Binaryout2, Realout2 = self.fc2(Binaryout1, Realout1)
        Binaryout2 = self.bn2(Binaryout2)
        Binaryout2 = self.htanh2(Binaryout2)
        # Copy reused layers and freeze them
        realbn2 = copy.deepcopy(self.bn2)
        with torch.no_grad():
            Realout2 = realbn2(Realout2)
            Realout2 = F.hardtanh(Realout2)

        # Next Layer
        Binaryout3, Realout3 = self.fc3(Binaryout2, Realout2)
        Binaryout3 = self.drop(Binaryout3)
        Binaryout3 = self.bn3(Binaryout3)
        Binaryout3 = self.htanh3(Binaryout3)
        # Copy reused layers and freeze them
        realdrop = copy.deepcopy(self.drop)
        realbn3 = copy.deepcopy(self.bn3)
        with torch.no_grad():
            Realout3 = realdrop(Realout3)
            Realout3 = realbn3(Realout3)
            Realout3 = F.hardtanh(Realout3)

        # Next Layer
        out = self.fc4(Binaryout3)

        penul_Binaryout = out

        return self.logsoftmax(out), penul_Binaryout, Binaryout1, Realout1, Binaryout2, Realout2, Binaryout3, Realout3
        # return self.softmax(x)

model = Net()

if args.cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5, last_epoch=-1)


def train(epoch, KD_Label_bank):
    model.train()

    # Feature Map Loss
    Feature_loss = Spatial_Channel_loss().cuda()
    # Soft Label Loss
    kl_criterion = KL_SoftLabelloss().cuda()
    # clustering loss
    # Penultimate_loss = nn.SmoothL1Loss(reduction='mean').cuda()
    Penultimate_loss = Line_CosSimloss().cuda()

    # Create Son_Label_bank to collect the sum of labels generated by each epoch, and each epoch iteration is initialized to 0
    Son_Label_bank = torch.zeros(args.num_classes, args.num_classes).cuda()
    True_Num = torch.zeros(args.num_classes, dtype=torch.int64).cuda()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output, penul_Binaryout, L1_Binaryout, L1_Realout, L2_Binaryout, L2_Realout, L3_Binaryout, L3_Realout = model(data)

        L1_loss = Feature_loss(L1_Binaryout, L1_Realout.detach())
        L2_loss = Feature_loss(L2_Binaryout, L2_Realout.detach())
        L3_loss = Feature_loss(L3_Binaryout, L3_Realout.detach())

        Fmap_loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        Fmap_loss = Fmap_loss + L1_loss + L2_loss + L3_loss
        Fmap_loss = args.Feature_alpha * Fmap_loss / 3.0

        # Soft Label Loss
        Label_loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        soft_target = KD_Label_bank[target].cuda()
        if epoch == 0:
            Label_loss = Label_loss + 0 * kl_criterion(output, soft_target.detach(), temperature=args.temperature)
        else:
            Label_loss = Label_loss + args.Label_beta * kl_criterion(output, soft_target.detach(), temperature=args.temperature)

        # Penultimate layer Clustering Loss
        Cluster_loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        penul_Clustering = Clustering(penul_Binaryout, target, args.num_classes)
        Cluster_loss = Cluster_loss + args.Cluster_gamma * Penultimate_loss(penul_Binaryout, penul_Clustering.detach())

        # Total Loss = Task loss + Other Loss
        loss = criterion(output, target.detach()) + Fmap_loss + Label_loss + Cluster_loss
        # loss = criterion(output, target)

        # if epoch%20==0:
        #     optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Fmap_loss:{:.6f} Label_loss:{:.6f} Cluster_loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), Fmap_loss.item(), Label_loss.item(), Cluster_loss.item()))

        # ######################################### update Son label bank ##############################################
        # Calculate the label sum of the current mini-batch
        Son_Label, Num = Up_Label_bank(output, target, args.num_classes, mode='top1')
        # Update the Son_Label_bank inside the statistical epoch
        Son_Label_bank += Son_Label
        True_Num += Num
        # ##############################################################################################################

    # ######################################### update KD label bank ##############################################
    # Computes the aggregated average label for the current epoch
    Son_Label_bank = Son_Label_bank / torch.unsqueeze(True_Num, dim=1)
    # Update KD label bank
    KD_Label_bank = Son_Label_bank
    # ##############################################################################################################


    # print('lr =', optimizer.param_groups[0]['lr'])
    scheduler.step()

    return KD_Label_bank


def test():
    model.eval()
    test_loss = 0
    correct = 0
    best = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output, penul_Binaryout, L1_Binaryout, L1_Realout, L2_Binaryout, L2_Realout, L3_Binaryout, L3_Realout = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return (100. * correct / len(test_loader.dataset))


best_acc = 0
for epoch in range(0, args.epochs):
    # Initialize the Soft Label Bank
    KD_Label_bank = torch.zeros(args.num_classes, args.num_classes).cuda()

    KD_Label_bank = train(epoch, KD_Label_bank)

    acc = test()

    # if epoch%20==0:
    #     optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5

    best_acc = max(best_acc, acc)
    print('--- Best Test ACC =', best_acc.item(), '%\n')
