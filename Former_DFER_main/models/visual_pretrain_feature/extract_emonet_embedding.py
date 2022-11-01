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

import config
from visual_pretrain_feature.emonet.models.emonet import EmoNet
from visual_pretrain_feature.dataset import FaceDatasetForEmoNet
from visual_pretrain_feature.util import write_feature_to_csv, get_vids
from visual_pretrain_feature.emonet.data_augmentation import DataAugmentor


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

    print(f'==> Extracting emonet embedding...')
    # in: face dir
    face_dir = config.PATH_TO_RAW_FACE[params.task]
    # out: feature csv dir
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.task], 'emonet')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif params.overwrite:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')

    # load model
    model = EmoNet().cuda()
    # model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS[params.task], 'emonet_8.pth')
    checkpoint = torch.load(checkpoint_file)
    pre_trained_dict = {k.replace('module.', ''): v for k,v in checkpoint.items()}
    model.load_state_dict(pre_trained_dict)

    # transform
    augmentor = DataAugmentor(256, 256)
    transform = transforms.Compose([transforms.ToTensor()])

    # extract embedding video by video
    vids = get_vids(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    face_rates = []
    feature_dim = 0
    for i, vid in enumerate(vids, 1):
        print(f"Processing video '{vid}' ({i}/{len(vids)})...")
        # forward
        dataset = FaceDatasetForEmoNet(vid, face_dir, transform=transform, augmentor=augmentor)
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
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    params = parser.parse_args()

    main(params)
