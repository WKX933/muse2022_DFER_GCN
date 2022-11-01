# *_*coding:utf-8 *_*
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import config
from visual_pretrain_feature.manet.model.manet import manet
from visual_pretrain_feature.dataset import FaceDataset
from visual_pretrain_feature.util import write_feature_to_csv, get_vids


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
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
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
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, ids in tqdm(data_loader):
            images = images.cuda()
            embedding = model(images, return_embedding=True)
            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(ids)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


def main(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    print(f'==> Extracting manet embedding...')
    # in: face dir
    face_dir = config.PATH_TO_RAW_FACE[params.task]
    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.task], 'manet1')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # load model
    model = manet(num_classes=12666).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS[params.task], 'manet/Pretrained_on_MSCeleb.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    pre_trained_dict = {k.replace('module.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(pre_trained_dict)

    # transform
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    # extract embedding video by video
    vids = get_vids(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    face_rates = []
    feature_dim = 0
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # forward
        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            features, timestamps = np.empty((0, 0)), np.empty((0,))
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
            features, timestamps = extract(data_loader, model)
            feature_dim = features.shape[1]
        # write
        face_rate = write_feature_to_csv(features, timestamps, save_dir, vid, feature_dim=feature_dim)
        print(f'face rate: {face_rate}')
        face_rates.append(face_rate)
    print(f'Mean face rate: {np.mean(face_rates)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--task', type=str, default='reaction', help='task id')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    params = parser.parse_args()

    main(params)
