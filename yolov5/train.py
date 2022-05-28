import os
from absl import app
from sklearn.model_selection import GroupKFold
import yaml

from yolov5.prepare_dataset import get_df

FOLD = 1  # which fold to train
DIM = 3000
MODEL = 'yolov5s6'
BATCH = 4
EPOCHS = 7
OPTMIZER = 'Adam'

PROJECT = 'great-barrier-reef-public'  # w&b in yolov5
NAME = f'{MODEL}-dim{DIM}-fold{FOLD}'  # w&b for yolov5

REMOVE_NOBBOX = True  # remove images with no bbox
# ROOT_DIR  = '/kaggle/input/tensorflow-great-barrier-reef/'
# IMAGE_DIR = '/kaggle/images' # directory to save images
# LABEL_DIR = '/kaggle/labels' # directory to save labels


def config(train_df, valid_df):

    cwd = './working/'

    print('os.path', os.path.join(cwd, 'train.txt'))
    with open(os.path.join(cwd, 'train.txt'), 'w') as f:
        for path in train_df.image_path.tolist():
            f.write(path + '\n')

    with open(os.path.join(cwd, 'val.txt'), 'w') as f:
        for path in valid_df.image_path.tolist():
            f.write(path + '\n')

    data = dict(
        path='./working',
        train=os.path.join(cwd, 'train.txt'),
        val=os.path.join(cwd, 'val.txt'),
        nc=1,
        names=['cots'],
    )

    with open(os.path.join(cwd, 'gbr.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    f = open(os.path.join(cwd, 'gbr.yaml'), 'r')
    print('\nyaml:')
    print(f.read())


def main(argv):
    del argv

    # get dataframe from prepare_dataset
    df = get_df()

    # Create folds
    kf = GroupKFold(n_splits=3)
    df = df.reset_index(drop=True)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, groups=df.video_id.tolist())):
        df.loc[val_idx, 'fold'] = fold

    # Dataset
    train_files = []
    val_files   = []
    train_df = df.query("fold!=@FOLD")
    valid_df = df.query("fold==@FOLD")
    train_files += list(train_df.image_path.unique())
    val_files += list(valid_df.image_path.unique())
    # len(train_files), len(val_files)

    # Configuration
    config(train_df, valid_df)

    # # yolov5
    # %cd /working
    # !rm -r /working/yolov5
    # # !git clone https://github.com/ultralytics/yolov5  # clone
    # !cp -r /input/yolov5-lib-ds /working/yolov5
    # %cd yolov5
    # %pip install -qr requirements.txt  # install

    # # Training
    # !python train.py --img {DIM}\
    # --batch {BATCH}\
    # --epochs {EPOCHS}\
    # --optimizer {OPTMIZER}\
    # --data /working/gbr.yaml\
    # --hyp /working/hyp.yaml\
    # --weights {MODEL}.pt\
    # --project {PROJECT} --name {NAME}\
    # --exist-ok

    # # Output files
    # OUTPUT_DIR = '{}/{}'.format(PROJECT, NAME)
    # # !ls {OUTPUT_DIR}

if __name__ == "__main__":
    app.run(main)