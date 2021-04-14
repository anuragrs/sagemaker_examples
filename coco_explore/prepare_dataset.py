from pycocotools.coco import COCO
from collections import defaultdict, Counter
from pprint import pprint
import random
import copy
import os
import shutil
import re

ONLY_GEN_ANNS = True # False

# same sequence
random.seed(17)

# paths
dataDir = '/shared/data/coco/'
targetDataDir = '/shared/data/coco_balanced/'
annFile='/shared/data/coco/nv_tfrecords/annotations/instances_train2017.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)
images = coco.dataset['images']
image_id_to_detail_map = {}
for image in images:
    image_id_to_detail_map[image['id']] = image

categories = coco.dataset['categories']

cats = coco.loadCats(coco.getCatIds())

class_to_images = defaultdict(list)

# get image ids for each class
for cat in cats:
    obj_id = cat['id']
    img_ids = coco.getImgIds(catIds=[obj_id])
    if img_ids:
        class_to_images[obj_id].extend(img_ids)

target_images_per_class = 1500 # roughly calculated as 120000/80

balanced_class_to_images = defaultdict(list)

# sampling
for k, v in class_to_images.items():
    if len(v) > target_images_per_class: # undersample
        balanced_class_to_images[k].extend(random.sample(v, target_images_per_class))
    else: # sampling with replacement
        balanced_class_to_images[k].extend(random.choices(v, k=target_images_per_class))

coco_balanced = {}

# mkdirs
if not ONLY_GEN_ANNS:
    try:
        os.makedirs(os.path.join(targetDataDir, 'train2017'))
    except:
        shutil.rmtree(targetDataDir)
        os.makedirs(os.path.join(targetDataDir, 'train2017'))

global_img_counts = Counter()
for k, v in balanced_class_to_images.items():
    global_img_counts += Counter(v)

img_id_counter = 1 # these indices need to be integers (for cocotools index creator)
id_counter = 1 # assigned to annotations
new_annotations = []
new_images = []
new_categories = copy.deepcopy(categories)

dbg = open('debug.txt', 'w')

for k, v in balanced_class_to_images.items():
    for img_id in set(v):
        local_img_count = Counter(v)[img_id]
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        start_idx = global_img_counts[img_id]
        end_idx = start_idx - local_img_count
        assert end_idx >= 0, '{} has invalid counts start {} end {}'.format(img_id, start_idx, end_idx)
        for img_dup_index in range(start_idx, end_idx, -1):
            # dup_img_id = '{}_{}'.format(img_id, img_dup_index)
            dup_img_id = img_id_counter
            img_id_counter += 1
            # add to new_images
            dup_image_details = copy.deepcopy(image_id_to_detail_map[img_id])
            dup_image_details['id'] = dup_img_id
            orig_filename = dup_image_details['file_name']
            dup_image_details['file_name'] = re.sub('(?P<zeros>00*)[^0][0-9]+', r'\g<zeros>{}'.format(dup_img_id), orig_filename)
            print(dup_image_details)
            if not ONLY_GEN_ANNS:
                # make a copy of original image
                src = os.path.join(dataDir, 'train2017', orig_filename)
                dest = os.path.join(targetDataDir, 'train2017', dup_image_details['file_name'])
                shutil.copy(src, dest)
            new_images.append(dup_image_details)
            dup_img_anns = copy.deepcopy(anns)
            #print(dup_img_id)
            for d in dup_img_anns: # list of dicts
                d['image_id'] = dup_img_id
                d['id'] = id_counter
                id_counter += 1
            # add to new_annotations
            new_annotations.extend(dup_img_anns)
            print(img_id, dup_img_id, file=dbg)
            pprint(dup_img_anns, dbg)
        global_img_counts[img_id] -= local_img_count


dbg.close()

balanced_coco_dict = {}
balanced_coco_dict['images'] = new_images
assert len(balanced_coco_dict['images']) == target_images_per_class * 80
balanced_coco_dict['categories'] = new_categories
balanced_coco_dict['annotations'] = new_annotations

import json
json.dump(balanced_coco_dict, open(os.path.join(targetDataDir, 'annotations.json'), 'w'), indent=4)

