# *_*coding:utf-8 *_*
import os
import re
import numpy as np
import pandas as pd


def write_feature_to_csv(features, timestamps, save_dir, vid, feature_dim=None):
    feature_dim = features.shape[1] if feature_dim is None else feature_dim
    assert feature_dim != 0, f"Error: feature dim must be non-zero!"

    # get corresponding labeled/feature file to align
    task_id = int(re.search('c(\d)_muse_', save_dir).group(1))  # infer the task id from save_dir (naive/unelegant approach)
    if task_id == 2:  # for task "c2"
        rel_path = '../FAUs'  # use csv file in "au" feature as reference beacause of there is no timestamp in the label file
    elif task_id == 4:  # for task "c4"
        rel_path = '../../label_segments/anno12_EDA'  # no arousal label for this task
    else:
        rel_path = '../FAUs'
    ref_dir = os.path.abspath(os.path.join(save_dir, rel_path))
    assert os.path.exists(ref_dir), f'Error:  label dir "{ref_dir}" does not exist!'
    ref_file = os.path.join(ref_dir, f'{vid}.csv')
    df_ref = pd.read_csv(ref_file)

    meta_columns = ['timestamp', 'File_ID']
    timestamp_column = meta_columns[0]
    metas = df_ref[meta_columns].values
    timestamps_ref = df_ref[timestamp_column].values

    # pad
    pad_features = []
    face_count = 0
    # for ts in timestamps_ref:
    for ts in timestamps:
        if ts in timestamps:
            feature = features[timestamps == ts]
            face_count += 1
        else:
            feature = np.zeros((feature_dim,))
        pad_features.append(feature)
    pad_features = np.row_stack(pad_features)
    face_rate = 100.0 * face_count / len(timestamps_ref)
    # assert np.all(timestamps == timestamps_ref), 'Invalid timestamps!'

    # combine
    data = np.column_stack([metas, pad_features])
    columns = meta_columns + [str(i) for i in range(feature_dim)]
    df = pd.DataFrame(data, columns=columns)
   # df[meta_columns] = df[meta_columns].astype(np.int64)
    csv_file = os.path.join(save_dir, f'{vid}.csv')
    df.to_csv(csv_file, index=False)

    return face_rate


def get_vids(data_path):
    vids = []
    for dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir)):
            try:
                vid = int(dir)
            except:
                print(f'Warning: invalid dir "{dir}"!')
                continue
            vids.append(dir)
    vids = sorted(vids, key=lambda x: int(x))
    return vids
