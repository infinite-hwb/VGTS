import os
import sys
import random
from collections import OrderedDict
import math
import copy
import logging
import pickle
import glob
import numpy as np
import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ElementTree

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from vgts.structures.bounding_box import BoxList
from vgts.engine.augmentation import DataAugmentation
from vgts.utils import get_image_size_after_resize_preserving_aspect_ratio, mkdir, read_image
from vgts.structures.feature_map import FeatureMapSize


def read_annotation_file(path):
    dataframe = pd.read_csv(path)
    if not "imagefilename" in dataframe.columns:
        imagefilename = []
        for row in dataframe["imageid"]:
            imagefilename.append(str(row)+".jpg")
        dataframe["imagefilename"] = imagefilename

    if not "classfilename" in dataframe.columns:
        classfilename = []
        for row in dataframe["classid"]:
            classfilename.append(str(row)+".jpg")
        dataframe["classfilename"] = classfilename

    required_columns = {"imageid", "imagefilename", "classid", "classfilename", "gtbboxid", "difficult", "lx", "ty", "rx", "by"}
    assert required_columns.issubset(dataframe.columns), "Missing columns in gtboxframe: {}".format(required_columns - set(dataframe.columns))

    return dataframe

def build_dbtest_dataset(data_path, name, eval_scale, cache_images=False, no_image_reading=False, logger_prefix="VGTS"):
    logger = logging.getLogger(f"{logger_prefix}.dataset")
    logger.info("Preparing the DBTEST dataset: version {0}, eval scale {1}, image caching {2}".format(name, eval_scale, cache_images))

    annotation_folder="classes"
    image_size = 1656
    classdatafile = os.path.join(data_path, "dbtest", annotation_folder,"dbtest.csv")
    gt_path = os.path.join(data_path, "dbtest", annotation_folder, "images")
    image_path = os.path.join(data_path, "dbtest", "src", str(image_size))
    gtboxframe = read_annotation_file(classdatafile)

    subset_name = name.lower()
    assert subset_name.startswith("dbtest"), ""
    subset_name = subset_name[len("dbtest"):]
    subsets = ["train", "val-old-cl", "val-new-cl", "val-all"]
    found_subset = False
    for subset in subsets:
        if subset_name == "-"+subset:
            found_subset = subset
            break
    assert found_subset, "Could not identify subset {}".format(subset_name)

    def get_unique_images(gtboxframe):
        unique_images = gtboxframe[["imageid", "imagefilename"]].drop_duplicates()
        image_ids = list(unique_images["imageid"])
        image_file_names = list(unique_images["imagefilename"])
        return image_ids, image_file_names

    if subset in ["val-new-cl"]:
        gtboxframe = gtboxframe[gtboxframe["split"] == "val-new-cl"]
        image_ids, image_file_names = get_unique_images(gtboxframe)
    elif subset in ["val-old-cl", "val-new-cl", "val-all"]:
        gtboxframe = gtboxframe[gtboxframe["split"].isin(["val-old-cl", "val-new-cl"])]
        image_ids, image_file_names = get_unique_images(gtboxframe)
        if subset != "val-all":
            gtboxframe = gtboxframe[gtboxframe["split"] == subset]
    else:
        raise RuntimeError("Unknown subset {0}".format(subset))

    dataset = DatasetOneShotDetection(gtboxframe, gt_path, image_path, name, image_size, eval_scale,
                                      image_ids=image_ids, image_file_names=image_file_names,
                                      cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    return dataset
    
def build_db_dataset(data_path, name, eval_scale, cache_images=False, no_image_reading=False, logger_prefix="VGTS"):
    logger = logging.getLogger(f"{logger_prefix}.dataset")
    logger.info("Preparing the DB dataset: version {0}, eval scale {1}, image caching {2}".format(name, eval_scale, cache_images))

    annotation_folder="classes"
    image_size = 1656 
    classdatafile = os.path.join(data_path, "db", annotation_folder,"db.csv")
    gt_path = os.path.join(data_path, "db", annotation_folder, "images")
    image_path = os.path.join(data_path, "db", "src", str(image_size))
    gtboxframe = read_annotation_file(classdatafile)
    subset_name = name.lower()
    assert subset_name.startswith("db"), ""
    subset_name = subset_name[len("db"):]
    subsets = ["train", "val-new-cl", "val-all"]
    found_subset = False
    for subset in subsets:
        if subset_name == "-"+subset:
            found_subset = subset
            break
    assert found_subset, "Could not identify subset {}".format(subset_name)

    def get_unique_images(gtboxframe):
        unique_images = gtboxframe[["imageid", "imagefilename"]].drop_duplicates()
        image_ids = list(unique_images["imageid"])
        image_file_names = list(unique_images["imagefilename"])
        return image_ids, image_file_names

    if subset in ["train"]:
        gtboxframe = gtboxframe[gtboxframe["split"] == "train"]
        image_ids, image_file_names = get_unique_images(gtboxframe)
    elif subset in ["val-new-cl", "val-all"]:
        gtboxframe = gtboxframe[gtboxframe["split"].isin(["val-new-cl"])]
        image_ids, image_file_names = get_unique_images(gtboxframe)
        if subset != "val-all":
            gtboxframe = gtboxframe[gtboxframe["split"] == subset]
    else:
        raise RuntimeError("Unknown subset {0}".format(subset))

    dataset = DatasetOneShotDetection(gtboxframe, gt_path, image_path, name, image_size, eval_scale,
                                      image_ids=image_ids, image_file_names=image_file_names,
                                      cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    return dataset


def build_tkh_dataset(data_path, name, eval_scale, cache_images=False, no_image_reading=False, logger_prefix="OS2D"):
    logger = logging.getLogger(f"{logger_prefix}.dataset")
    logger.info("Preparing the TKH dataset: version {0}, eval scale {1}, image caching {2}".format(name, eval_scale, cache_images))

    annotation_folder="classes"
    image_size = 2400           # max w/h pixes
    classdatafile = os.path.join(data_path, "tkh", annotation_folder,"tkh.csv")
    gt_path = os.path.join(data_path, "tkh", annotation_folder, "images")
    image_path = os.path.join(data_path, "tkh", "src", str(image_size))
    gtboxframe = read_annotation_file(classdatafile)

    # define a subset split (using closure)
    subset_name = name.lower()
    assert subset_name.startswith("tkh"), ""
    subset_name = subset_name[len("tkh"):]
    subsets = ["train", "val-old-cl", "val-new-cl", "val-all", "train-mini"]
    found_subset = False
    for subset in subsets:
        if subset_name == "-"+subset:
            found_subset = subset
            break
    assert found_subset, "Could not identify subset {}".format(subset_name)

    def get_unique_images(gtboxframe):
        unique_images = gtboxframe[["imageid", "imagefilename"]].drop_duplicates()
        image_ids = list(unique_images["imageid"])
        image_file_names = list(unique_images["imagefilename"])
        return image_ids, image_file_names

    if subset in ["train", "train-mini"]:
        gtboxframe = gtboxframe[gtboxframe["split"] == "train"]
        image_ids, image_file_names = get_unique_images(gtboxframe)
        if subset == "train-mini":
            image_ids = image_ids[:2]
            image_file_names = image_file_names[:2]
            gtboxframe = gtboxframe[gtboxframe["imageid"].isin(image_ids)]
    elif subset in ["val-old-cl", "val-new-cl", "val-all"]:
        gtboxframe = gtboxframe[gtboxframe["split"].isin(["val-old-cl", "val-new-cl"])]
        image_ids, image_file_names = get_unique_images(gtboxframe)
        if subset != "val-all":
            gtboxframe = gtboxframe[gtboxframe["split"] == subset]
    else:
        raise RuntimeError("Unknown subset {0}".format(subset))

    dataset = DatasetOneShotDetection(gtboxframe, gt_path, image_path, name, image_size, eval_scale,
                                      image_ids=image_ids, image_file_names=image_file_names,
                                      cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    return dataset

def build_tkhtest_dataset(data_path, name, eval_scale, cache_images=False, no_image_reading=False, logger_prefix="OS2D"):
    logger = logging.getLogger(f"{logger_prefix}.dataset")
    logger.info("Preparing the TKH TEST dataset: version {0}, eval scale {1}, image caching {2}".format(name, eval_scale, cache_images))

    annotation_folder="classes"
    image_size = 2400           # max w/h pixes
    classdatafile = os.path.join(data_path, "tkhtest", annotation_folder,"tkhtest.csv")
    gt_path = os.path.join(data_path, "tkhtest", annotation_folder, "images")
    image_path = os.path.join(data_path, "tkhtest", "src", str(image_size))
    gtboxframe = read_annotation_file(classdatafile)

    # define a subset split (using closure)
    subset_name = name.lower()
    assert subset_name.startswith("tkhtest"), ""
    subset_name = subset_name[len("tkhtest"):]
    subsets = ["val-old-cl", "val-new-cl"]
    found_subset = False
    for subset in subsets:
        if subset_name == "-"+subset:
            found_subset = subset
            break
    assert found_subset, "Could not identify subset {}".format(subset_name)

    def get_unique_images(gtboxframe):
        unique_images = gtboxframe[["imageid", "imagefilename"]].drop_duplicates()
        image_ids = list(unique_images["imageid"])
        image_file_names = list(unique_images["imagefilename"])
        return image_ids, image_file_names

    if subset in ["val-old-cl", "val-new-cl", "val-all"]:
        gtboxframe = gtboxframe[gtboxframe["split"].isin(["val-old-cl", "val-new-cl"])]
        image_ids, image_file_names = get_unique_images(gtboxframe)
        if subset != "val-all":
            gtboxframe = gtboxframe[gtboxframe["split"] == subset]
    else:
        raise RuntimeError("Unknown subset {0}".format(subset))

    dataset = DatasetOneShotDetection(gtboxframe, gt_path, image_path, name, image_size, eval_scale,
                                      image_ids=image_ids, image_file_names=image_file_names,
                                      cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    return dataset
    
def build_dataset_by_name(data_path, name, eval_scale, cache_images=False, no_image_reading=False, logger_prefix="VGTS"):
    if name.lower().startswith("dbtest"):
        return build_dbtest_dataset(data_path, name, eval_scale, cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    elif name.lower().startswith("db"):
        return build_db_dataset(data_path, name, eval_scale, cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    elif name.lower().startswith("tkhtest"):
        return build_tkhtest_dataset(data_path, name, eval_scale, cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)
    elif name.lower().startswith("tkh"):
        return build_tkh_dataset(data_path, name, eval_scale, cache_images=cache_images, no_image_reading=no_image_reading, logger_prefix=logger_prefix)

class DatasetOneShotDetection(data.Dataset):
    def __init__(self, gtboxframe, gt_path, image_path, name, image_size, eval_scale,
                       cache_images=False, no_image_reading=False,
                       image_ids=None, image_file_names=None, logger_prefix="VGTS"):
        self.logger = logging.getLogger(f"{logger_prefix}.dataset")
        self.name = name
        self.image_size = image_size
        self.eval_scale = eval_scale
        self.cache_images = cache_images

        self.gtboxframe = gtboxframe
        required_columns = {"imageid", "imagefilename", "classid", "classfilename", "gtbboxid", "difficult", "lx", "ty", "rx", "by"}
        assert required_columns.issubset(self.gtboxframe.columns), "Missing columns in gtboxframe: {}".format(required_columns - set(self.gtboxframe.columns))

        self.gt_path = gt_path
        self.image_path = image_path
        self.have_images_read = False

        if image_ids is not None and image_file_names is not None:
            self.image_ids = image_ids
            self.image_file_names = image_file_names
        else:
            unique_images = gtboxframe[["imageid", "imagefilename"]].drop_duplicates()
            self.image_ids = list(unique_images["imageid"])
            self.image_file_names = list(unique_images["imagefilename"])

        if not no_image_reading:
            self._read_dataset_gt_images()
            self._read_dataset_images()
            self.have_images_read=True

        self.num_images = len(self.image_ids)
        self.num_boxes = len(self.gtboxframe)
        self.num_classes = len(self.gtboxframe["classfilename"].unique())

        self.logger.info("Loaded dataset {0} with {1} images, {2} boxes, {3} classes".format(
            self.name, self.num_images, self.num_boxes, self.num_classes
        ))

    def get_name(self):
        return self.name

    def get_eval_scale(self):
        return self.eval_scale

    def get_class_ids(self):
        return self.gtboxframe["classid"].unique()

    def get_class_ids_for_image_ids(self, image_ids):
        dataframe = self.get_dataframe_for_image_ids(image_ids)
        return dataframe["classid"].unique()

    def get_dataframe_for_image_ids(self, image_ids):
        return self.gtboxframe[self.gtboxframe["imageid"].isin(image_ids)]

    def get_image_size_for_image_id(self, image_id):
        return self.image_size_per_image_id[image_id]

    def _read_dataset_images(self):
        self.image_path_per_image_id = OrderedDict()
        self.image_size_per_image_id = OrderedDict()
        self.image_per_image_id = OrderedDict()
        for image_id, image_file in zip(self.image_ids, self.image_file_names):
            if image_id not in self.image_path_per_image_id:
                img_path = os.path.join(self.image_path, image_file)
                self.image_path_per_image_id[image_id] = img_path
                img = self._get_dataset_image_by_id(image_id)
                self.image_size_per_image_id[image_id] = FeatureMapSize(img=img)

        self.logger.info("{1} {0} data images".format(len(self.image_path_per_image_id), "Read" if self.cache_images else "Found"))

    def _read_dataset_gt_images(self):
        self.gt_images_per_classid = OrderedDict()
        if self.gt_path is not None:
            for index, row in self.gtboxframe.iterrows():
                gt_file = row["classfilename"]
                class_id = row["classid"]
                if class_id not in self.gt_images_per_classid:
                    self.gt_images_per_classid[class_id] = read_image(os.path.join(self.gt_path, gt_file))
            self.logger.info("Read {0} GT images".format(len(self.gt_images_per_classid)))
        else:
            self.logger.info("GT images are not provided")

    def split_images_into_buckets_by_size(self):
        buckets = []
        bucket_image_size = []
        for image_id, s in self.image_size_per_image_id.items():
            if s not in bucket_image_size:
                bucket_image_size.append(s)
                buckets.append([])
            i_bucket = bucket_image_size.index(s)
            buckets[i_bucket].append(image_id)
        return buckets

    def _get_dataset_image_by_id(self, image_id):
        assert image_id in self.image_path_per_image_id, "Can work only with checked images"

        if image_id not in self.image_per_image_id :
            img_path = self.image_path_per_image_id[image_id]
            img = read_image(img_path)
            img_size = FeatureMapSize(img=img)
            if max(img_size.w, img_size.h) != self.image_size:
                h, w = get_image_size_after_resize_preserving_aspect_ratio(img_size.h, img_size.w, self.image_size)
                img = img.resize((w, h), resample=Image.ANTIALIAS)
            if self.cache_images:
                self.image_per_image_id[image_id] = img
        else:
            img = self.image_per_image_id[image_id]

        return img

    @staticmethod
    def get_boxes_from_image_dataframe(image_data, image_size):
        if not image_data.empty:
            label_ids_global = torch.tensor(list(image_data["classid"]), dtype=torch.long)
            difficult_flag = torch.tensor(list(image_data["difficult"] == 1), dtype=torch.bool)
            boxes = image_data[["lx", "ty", "rx", "by"]].to_numpy()
            boxes[:, 0] *= image_size.w
            boxes[:, 2] *= image_size.w
            boxes[:, 1] *= image_size.h
            boxes[:, 3] *= image_size.h
            boxes = torch.FloatTensor(boxes)
            boxes = BoxList(boxes, image_size=image_size, mode="xyxy")
        else:
            boxes = BoxList.create_empty(image_size)
            label_ids_global = torch.tensor([], dtype=torch.long)
            difficult_flag = torch.tensor([], dtype=torch.bool)

        boxes.add_field("labels", label_ids_global)
        boxes.add_field("difficult", difficult_flag)
        boxes.add_field("labels_original", label_ids_global)
        boxes.add_field("difficult_original", difficult_flag)
        return boxes

    def get_image_annotation_for_imageid(self, image_id):
        image_data = self.gtboxframe[self.gtboxframe["imageid"] == image_id]
        img_size = self.image_size_per_image_id[image_id]
        boxes = self.get_boxes_from_image_dataframe(image_data, img_size)
        return boxes

    def copy_subset(self, subset_size=None, set_eval_mode=True):
        dataset_subset = copy.copy(self) 

        if subset_size is not None:
            dataset_subset.num_images = min(subset_size, dataset_subset.num_images)
            dataset_subset.image_ids = self.image_ids[:dataset_subset.num_images]
            dataset_subset.image_file_names = self.image_file_names[:dataset_subset.num_images]
            image_mask = dataset_subset.gtboxframe["imageid"].isin(dataset_subset.image_ids)
            dataset_subset.gtboxframe = dataset_subset.gtboxframe[image_mask]

            dataset_subset.name = self.name + "-subset{}".format(subset_size)

            dataset_subset._read_dataset_gt_images()
            dataset_subset._read_dataset_images()

        if set_eval_mode:
            dataset_subset.data_augmentation = None

        return dataset_subset
