#!/usr/bin/env python3

from absl import app
from absl import flags
import pandas as pd
import shutil
from bbox.utils import coco2yolo, coco2voc, voc2yolo
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str
from tqdm import tqdm
import numpy as np

flags.DEFINE_string("data_dir", None, "Directory to read data from")
flags.DEFINE_string("image_dir", "images", "Directory to save images")
flags.DEFINE_string("label_dir", "labels", "Directory to save labels")
flags.DEFINE_bool("remove_boxless_images", True, "Whether or not to remove boxless images.")

FLAGS = flags.FLAGS

def create_annotation_string(row):
    boxes_coco  = np.array(row.boxes).astype(np.float32).copy()
    num_boxes = len(boxes)
    class_names = ['cots'] * num_boxes
    labels = np.array([0] * num_boxes)[..., None].astype(str)
    bboxes_voc  = coco2voc(bboxes_coco, image_height, image_width)
    bboxes_voc  = clip_bbox(bboxes_voc, image_height, image_width)
    bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
    all_bboxes.extend(bboxes_yolo.astype(float))
    bboxes_info.extend([[row.image_id, row.video_id, row.sequence]]*len(bboxes_yolo))
    annots = np.concatenate([labels, bboxes_yolo], axis=1)
    return annot2str(annots)

def copy_files(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        shutil.copyfile(row.old_image_path, row.image_path)

def write_annotations(df):
    for _, row in tqdm(df.iterrows(), total=len(df)):
        string = create_annotation_string(row)
        with open(row.label_path, 'w') as f:
            if num_boxes == 0:
                f.write('')
                continue
            f.write(string)

def main(argv):
    del argv

    df = pd.read_csv(f'{FLAGS.data_dir}/train.csv')
    df['old_image_path'] = f'{FLAGS.data_dir}/train_images/video_'+df.video_id.astype(str)+'/'+df.video_frame.astype(str)+'.jpg'
    df['image_path'] = f'{FLAGS.image_dir}/' +df.image_id + '.jpg'
    df['label_path'] = f'{FLAGS.label_dir}/' + df.image_id + '.txt'
    # Python eval the strings, creating an annotation list
    print("Parsing annotations")
    df['annotations'] = df['annotations'].apply(eval)
    print("Showing first two rows in the dataframe")
    print(df.head())

    df['num_boxes'] = df['annotations'].apply(lambda x: len(x))
    if FLAGS.remove_boxless_images:
        df = df.query("num_boxes>0")

    print(f"Copying all images to {FLAGS.image_dir}")
    copy_files(df)
    print("Done copying images")

    print("Parsing annotations")
    df['boxes'] = df.annotations.apply(lambda x: [list(a.values()) for a in x])
    df['width']  = 1280
    df['height'] = 720

    write_annotations(df)

app.run(main)
