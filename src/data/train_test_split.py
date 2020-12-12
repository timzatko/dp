import os
import time
import shutil
import numpy as np

from tqdm import tqdm


def train_test_split(src, dst, **kwargs):
    split = kwargs.get('split', (0.8, 0.1, 0.1))
    dirname = kwargs.get('dirname', str(int(time.time())))

    if len(split) != 3:
        raise Exception("split mus be length of three!")

    if sum(split) != 1:
        raise Exception("sum of split must be 1!")

    dst_dir = os.path.join(dst, f'{dirname}')
    train_dir = os.path.join(dst_dir, 'train')
    test_dir = os.path.join(dst_dir, 'test')
    val_dir = os.path.join(dst_dir, 'val')

    if os.path.exists(dst_dir):
        print("not copying files since the destination directory already exists")

        return train_dir, test_dir, val_dir

    os.mkdir(dst_dir)
    print(f"copying to {dst_dir}...\n")

    # list of directories to copy
    src_dirs = os.listdir(src)
    print('shuffling an array...')
    np.random.shuffle(src_dirs)

    print('copying files...')
    src_dirs_count = len(src_dirs)
    for idx, dir in tqdm(enumerate(src_dirs), total=src_dirs_count):
        dst_dir = train_dir

        if idx > split[0] * src_dirs_count:
            dst_dir = test_dir
        if idx > (split[0] + split[1]) * src_dirs_count:
            dst_dir = val_dir

        shutil.copytree(os.path.join(src, dir), os.path.join(dst_dir, dir))

    return train_dir, test_dir, val_dir
