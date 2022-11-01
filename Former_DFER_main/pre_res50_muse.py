import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,1,3"
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import six
import sys
from os.path import join as pjoin
import torch.utils.data
import torch.utils.data.distributed
from models.pre_ST50 import GenerateModel
from models.resnet50_ferplus_dag import Resnet50_ferplus_dag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_muse import train_data_loader, test_data_loader
from loss_muse import *
from eval_muse import mean_pearsons
import random

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--gpu', type=str, default='5,2,3,4,6,7')
parser.add_argument('--seed',type=float,default=1001)

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = './'
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'

np.random.seed(10)
random.seed(10)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def main():


    best_score = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    # model = GenerateModel()
    model = GenerateModel()
    # model_res50 = Resnet50_ferplus_dag()
    model = torch.nn.DataParallel(model)
    state_dict = torch.load('/data6/wangkexin/muse2022/Former_DFER_main/Former-DFER-Pretrained-on-DFEW.pth')
    state_dict_res50 = torch.load('/data2/heyu/pretrain_model/ferplus/resnet50_ferplus_dag.pth')
    for k in list(state_dict_res50):
        state_dict_res50['s_former.'+k]=state_dict_res50.pop(k)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.module.load_state_dict(state_dict_res50,strict=False)
    model = model.cuda()
    # model = model.module
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, 7),
        torch.nn.Sigmoid()
    )

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = WrappedMSELoss(reduction='mean')
    # criterion = WrappedPRSLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            recorder = checkpoint['recorder']
            best_score = best_score.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        _, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_score, val_los = validate(val_loader, model, criterion, args)

        print(f'Epoch:{epoch:>3} / {args.epochs} | [Train] | Loss: {train_los:>.4f}')
        print(f'Epoch:{epoch:>3} / {args.epochs} | [Val] | Loss: {val_los:>.4f} | [mean pearson]: {val_score:>7.4f}')
        print('-' * 50)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_score > best_score
        best_score = max(val_score, best_score)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_score': best_score,
                         'seed':args.seed,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # if val_score > best_val_score:
        #     early_stop = 0
        #     best_val_score = val_score
        #     best_val_loss = val_los
        #     best_model_file = save_model(model, model_path, current_seed)


        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, 0, val_los, val_score)
        recorder.plot_curve(log_curve_path)

        # print('The best accuracy: {:.3f}'.format(best_acc.item()))
        # print('An epoch time: {:.1f}s'.format(epoch_time))
        # with open(log_txt_path, 'a') as f:
        #     f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
        #     f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    # top1 = AverageMeter('Accuracy', ':6.3f')
    # progress = ProgressMeter(len(train_loader),
    #                          [losses, top1],
    #                          prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        # if i % args.print_freq == 0:
        #     progress.display(i)

    return 0, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    # top1 = AverageMeter('Accuracy', ':6.3f')
    # progress = ProgressMeter(len(val_loader),
    #                          [losses, top1],
    #                          prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    full_labels=[]
    full_preds = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            full_labels.append(target.cpu().detach().squeeze().numpy().tolist()[:args.batch_size])
            full_preds.append(output.cpu().detach().squeeze().numpy().tolist()[:args.batch_size])

            # if i % args.print_freq == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        # with open(log_txt_path, 'a') as f:
        #     f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
        score = mean_pearsons(full_preds, full_labels)
    return score, losses.avg

def write_reaction_predictions(full_metas, full_preds, csv_dir, filename):
    meta_arr = np.row_stack(full_metas).squeeze()
    preds_arr = np.row_stack(full_preds)
    pred_df = pd.DataFrame(columns=['File_ID']+REACTION_LABELS)
    pred_df['File_ID'] = meta_arr
    pred_df[REACTION_LABELS] = preds_arr
    pred_df.to_csv(os.path.join(csv_dir, filename), index=False)
    return None


def write_predictions(task, full_metas, full_preds, prediction_path, filename):
    assert prediction_path != ''


    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if task == 'reaction':
        return write_reaction_predictions(full_metas, full_preds, prediction_path, filename)

    metas_flat = []
    for meta in full_metas:
        metas_flat.extend(meta)
    preds_flat = []
    for pred in full_preds:
        preds_flat.extend(pred if isinstance(pred, list) else (pred.squeeze() if pred.ndim>1 else pred))

    if isinstance(metas_flat[0],list):
        num_meta_cols = len(metas_flat[0])
    else:
        # np array
        num_meta_cols = metas_flat[0].shape[0]
    prediction_df = pd.DataFrame(columns = [f'meta_col_{i}' for i in range(num_meta_cols)])
    for i in range(num_meta_cols):
        prediction_df[f'meta_col_{i}'] = [m[i] for m in metas_flat]
    prediction_df['prediction'] = preds_flat
    #prediction_df['label'] = labels_flat
    prediction_df.to_csv(os.path.join(prediction_path, filename), index=False)



def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


if __name__ == '__main__':
    main()
