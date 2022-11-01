import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from Former_DFER_main.dataloader.video_transform import *
import numpy as np
import torch
import torchvision

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        x = self._data[2].replace(","," ")
        label = x.strip().split()
        label = list(map(float,label))
        label = torch.tensor(label, dtype=torch.float)
        return label


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]  把class_idx变为label
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 16] #try 12 and 14
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_name = record.path.split('/')[-1]
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path.sort()

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        return images, record.label

    def __len__(self):
        return len(self.video_list)


def train_data_loader(data_set):
    image_size = 112
    # image_size = 224
    train_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])

    train_data = VideoDataset(list_file="./dataloader/muse_train.txt",
                              num_segments=8,
                              duration=2,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(data_set):
    image_size = 112
    # image_size = 224
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = VideoDataset(list_file="./dataloader/muse_val.txt",
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data

def pred_data_loader(data_set):
    image_size = 112
    # image_size = 224
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = VideoDataset(list_file="./dataloader/muse_test.txt",
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data
