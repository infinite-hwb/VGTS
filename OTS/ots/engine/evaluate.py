import os
import random
import time
import math
from collections import OrderedDict
import numpy as np
import logging

import torch

from ots.loaddata.voc_eval import do_voc_evaluation
from ots.utils import add_to_meters_in_dict, print_meters, time_since, time_for_printing
from ots.modeling.model import get_feature_map_size_for_network
from ots.structures.feature_map import FeatureMapSize
import pickle

@torch.no_grad()
def evaluate(dataloader, net, cfg, criterion=None, print_per_class_results=False):
    # Get logger
    logger = logging.getLogger("OTS.evaluate")
    
    # Get dataloader name and scale
    dataset_name = dataloader.get_name()
    dataset_scale = dataloader.get_eval_scale()
    
    # Log the start of evaluation
    logger.info("Starting to eval on {0}, scale {1}".format(dataset_name, dataset_scale))
    t_start_eval = time.time()
    
    # Set the network to evaluation mode
    net.eval()

    # Create an iterator for the dataset
    iterator = make_iterator_extract_scores_from_images_batched(dataloader, net, logger,
                                                                image_batch_size=cfg.eval.batch_size,
                                                                is_cuda=cfg.is_cuda,
                                                                class_image_augmentation=cfg.eval.class_image_augmentation)
    # Initialize empty lists for boxes, gt_boxes, losses, and image_ids
    boxes = []
    gt_boxes = []
    losses = OrderedDict()
    image_ids = []
    num_evaluted_images = 0
    
    # Iterate over the data
    for data in iterator:
        # Unpack data
        image_id, image_loc_scores_pyramid, image_class_scores_pyramid,\
                    image_pyramid, query_img_sizes, class_ids,\
                    box_reverse_transform, image_fm_sizes_p, transform_corners_pyramid\
                    = data
        
        # Append image_id to list
        image_ids.append(image_id)

        # Increase the number of evaluated images by 1
        num_evaluted_images += 1
        img_size_pyramid = [FeatureMapSize(img=img) for img in image_pyramid]

        # Get the number of labels
        num_labels = len(class_ids)
        gt_boxes_one_image = dataloader.get_image_annotation_for_imageid(image_id)
        gt_boxes.append(gt_boxes_one_image)

        # If there are any ground truth boxes, process them
        if len(gt_boxes_one_image) > 0:
            # Get labels and update them
            gt_labels_one_image = gt_boxes_one_image.get_field("labels")
            dataloader.update_box_labels_to_local(gt_boxes_one_image, class_ids)

            # Create targets for localization
            loc_targets_pyramid, class_targets_pyramid = \
                    dataloader.box_coder.encode_pyramid(gt_boxes_one_image,
                                                        img_size_pyramid, num_labels,
                                                        default_box_transform_pyramid=box_reverse_transform)

            gt_boxes_one_image.add_field("labels", gt_labels_one_image)

            # If using cuda, move tensors to GPU
            if cfg.is_cuda:
                loc_targets_pyramid = [loc_targets.cuda() for loc_targets in loc_targets_pyramid]
                class_targets_pyramid = [class_targets.cuda() for class_targets in class_targets_pyramid]
                transform_corners_pyramid = [transform_corners.cuda() for transform_corners in transform_corners_pyramid]

            add_batch_dim = lambda list_of_tensors: [t.unsqueeze(0) for t in list_of_tensors]
            
            # Calculate losses if criterion is provided
            if criterion is not None:
                losses_iter = criterion(add_batch_dim(image_loc_scores_pyramid) if image_loc_scores_pyramid[0] is not None else None,
                                        add_batch_dim(loc_targets_pyramid),
                                        add_batch_dim(image_class_scores_pyramid),
                                        add_batch_dim(class_targets_pyramid)
                                        )
            
                for l in losses_iter:
                    losses_iter[l] = losses_iter[l].mean().item()
                print_meters(losses_iter, logger)
                add_to_meters_in_dict(losses_iter, losses)
        
        boxes_one_image = \
            # Decode the predicted boxes
            dataloader.box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                                img_size_pyramid, class_ids,
                                                nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                                nms_score_threshold=cfg.eval.nms_score_threshold,
                                                inverse_box_transforms=box_reverse_transform,
                                                transform_corners_pyramid=transform_corners_pyramid)

        boxes.append(boxes_one_image.cpu())
        
        if cfg.is_cuda:
            torch.cuda.empty_cache()

    for k in losses:
        losses[k] /= num_evaluted_images

    path_to_save_detections = ""
    if path_to_save_detections:
        data = {"image_ids" : image_ids,
                "boxes_xyxy" : [bb.bbox_xyxy for bb in boxes], 
                "labels" : [bb.get_field("labels") for bb in boxes],
                "scores" : [bb.get_field("scores") for bb in boxes],
                "gt_boxes_xyxy" : [bb.bbox_xyxy for bb in gt_boxes],
                "gt_labels" : [bb.get_field("labels") for bb in gt_boxes],
                "gt_difficults" : [bb.get_field("difficult") for bb in gt_boxes]
        }
        dataset_name = dataloader.get_name()
        os.makedirs(path_to_save_detections, exist_ok=True)
        save_path = os.path.join(path_to_save_detections, dataset_name + "_detections.pth")
        torch.save(data, save_path) 
    
    for mAP_iou_threshold in cfg.eval.mAP_iou_thresholds:
        logger.info("Evaluating at IoU th {:0.2f}".format(mAP_iou_threshold))
        ap_data = do_voc_evaluation(boxes, gt_boxes, iou_thresh=mAP_iou_threshold, use_07_metric=False)
        losses["mAP@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map"]
        losses["mAPw@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map_weighted"]
        losses["recall@{:0.2f}".format(mAP_iou_threshold)] = ap_data["recall"]
        losses["AP_joint_classes@{:0.2f}".format(mAP_iou_threshold)] = ap_data["ap_joint_classes"]

        if print_per_class_results:
            for i_class, (ap, recall, n_pos) in enumerate(zip(ap_data["ap_per_class"], ap_data["recall_per_class"], ap_data["n_pos"])):
                if not np.isnan(ap):
                    assert i_class in class_ids, "Could not find class_id in the list of ids"
                    logger.info("Class {0} (local {3}), AP {1:0.4f}, #obj {2}, recall {4:0.4f}".format(i_class,
                                                                                                       ap,
                                                                                                       n_pos,
                                                                                                       class_ids.index(i_class),
                                                                                                       recall))
    # Add evaluation time to losses and log the completion of evaluation
    losses["eval_time"] = (time.time() - t_start_eval)
    logger.info("Evaluated on {0}, scale {1}".format(dataset_name, dataset_scale))
    
    print_meters(losses, logger)
    
    return losses


def make_iterator_extract_scores_from_images_batched(dataloader, net, logger, image_batch_size, is_cuda,
                                                     num_random_pyramid_scales=0, num_random_negative_labels=-1,
                                                     class_image_augmentation=""):

    # Log the start of the score extraction process
    logger.info("Extracting scores from all images")
    
    # Get all class images, aspect ratios, and class ids from the dataloader
    class_images, class_aspect_ratios, class_ids = dataloader.get_all_class_images()
    num_classes = len(class_images)
    
    # Assert that the number of classes is equal to the lengths of class aspect ratios and class ids
    assert len(class_aspect_ratios) == num_classes
    assert len(class_ids) == num_classes
    
    # Calculate the size of the query images
    query_img_sizes = [FeatureMapSize(img=img) for img in class_images]
    class_batch_size = 1

    # Initialize list to store class convolutional layers
    class_conv_layer_batched = []
    
    # Log the start of the weight extraction process
    logger.info("Extracting weights from {0} classes{1}".format(num_classes,
        f" with {class_image_augmentation} augmentation" if class_image_augmentation else ""))
    
    # Loop over classes and extract weights
    for i in range(0, num_classes, class_batch_size):
        # Get batch of class ids
        batch_class_ids = class_ids[i : i + class_batch_size]

        # Prepare batch of class images
        batch_class_images = []
        for i_label in range(len(batch_class_ids)):
            # Get class image and move to cuda if needed
            im = class_images[i + i_label].squeeze(0)
            if is_cuda:
                im = im.cuda()
            batch_class_images.append(im)
            # Apply image augmentation if required
            if not class_image_augmentation:
                num_class_views = 1
            elif class_image_augmentation == "rotation90":
                im90 = im.rot90(1, [1, 2])
                im180 = im90.rot90(1, [1, 2])
                im270 = im180.rot90(1, [1, 2])
                batch_class_images.append(im90)
                batch_class_images.append(im180)
                batch_class_images.append(im270)
                num_class_views = 4
            elif class_image_augmentation == "horflip":
                im_flipped = im.flip(2)
                batch_class_images.append(im_flipped)
                num_class_views = 2
            elif class_image_augmentation == "horflip_rotation90":
                im90 = im.rot90(1, [1, 2])
                im180 = im90.rot90(1, [1, 2])
                im270 = im180.rot90(1, [1, 2])
                im_flipped = im.flip(2)
                im90_flipped = im90.flip(2)
                im180_flipped = im180.flip(2)
                im270_flipped = im270.flip(2)

                for new_im in [im90, im180, im270, im_flipped, im90_flipped, im180_flipped, im270_flipped]:
                    batch_class_images.append(new_im)

                num_class_views = len(batch_class_images)
            else:
                raise RuntimeError(f"Unknown value of class_image_augmentation: {class_image_augmentation}")

        # Extract feature maps from batch images and create corresponding OTS head
        for b_im in batch_class_images:
            class_feature_maps = net.net_label_features([b_im])
            class_conv_layer = net.OTS_head_creator.create_OTS_head(class_feature_maps)
            class_conv_layer_batched.append(class_conv_layer)
    
    # Prepare iterator for all images
    iterator_batches = dataloader.make_iterator_for_all_images(image_batch_size, num_random_pyramid_scales=num_random_pyramid_scales)
    
    # Loop over batches
    for batch_ids, pyramids_batch, box_transforms_batch, initial_img_size_batch in iterator_batches:
        t_start_batch = time.time()
        if num_random_negative_labels >= 0 :
            neg_labels = torch.randperm(len(class_conv_layer_batched))
            neg_labels = neg_labels[:num_random_negative_labels]
            pos_labels = dataloader.get_class_ids_for_image_ids(batch_ids)
            pos_labels = dataloader.convert_label_ids_global_to_local(pos_labels, class_ids)
            batch_labels_local = torch.cat([neg_labels, pos_labels], 0).unique()
        else:
            batch_labels_local = torch.arange(len(class_conv_layer_batched))
        
        batch_class_ids = [class_ids[l // num_class_views] for l in batch_labels_local]
        batch_query_img_sizes = [query_img_sizes[l // num_class_views] for l in batch_labels_local]

        # Initialize lists to store batch images pyramid, location scores, class scores, feature map sizes and transform corners
        batch_images_pyramid = []
        loc_scores = []
        class_scores = []
        fm_sizes = []
        transform_corners = []
        num_pyramid_levels = len(pyramids_batch)
        
        # Time measurements
        t_cum_features = 0.0
        t_cum_labels = 0.0
        for batch_images in pyramids_batch:
            # Extract feature maps from batch images
            if is_cuda:
                batch_images = batch_images.cuda()
            
            t_start_features = time.time()
            feature_maps = net.net_feature_maps(batch_images)
            torch.cuda.synchronize()
            t_cum_features += time.time() - t_start_features

            loc_scores.append([])
            class_scores.append([])
            fm_sizes.append([])
            transform_corners.append([])
            t_start_labels = time.time()
            assert class_batch_size == 1, "the iterator on images works only with labels batches of size 1"

            # Extract location and class scores, feature map sizes and transform corners for each class in the batch
            for i_class_batch in batch_labels_local:
                loc_s_p, class_s_p, _, fm_sizes_p, transform_corners_p = \
                     net(class_head=class_conv_layer_batched[i_class_batch],
                         feature_maps=feature_maps)
                loc_scores[-1].append(loc_s_p)
                class_scores[-1].append(class_s_p)
                fm_sizes[-1].append(fm_sizes_p)
                transform_corners[-1].append(transform_corners_p)
            torch.cuda.synchronize()
            t_cum_labels += time.time() - t_start_labels

            if not feature_maps.requires_grad:
                del feature_maps

            batch_images_pyramid.append(batch_images)

        timing_str = "Feature time: {0}, Label time: {1}, ".format(time_for_printing(t_cum_features, mode="s"),
                                                          time_for_printing(t_cum_labels, mode="s"))
        # Loop over images in the batch
        for i_image_in_batch, image_id in enumerate(batch_ids):
            # Initialize lists to store image location scores, class scores, feature map sizes and transform corners
            image_loc_scores_p, image_class_scores_p, image_fm_sizes_p = [], [], []
            transform_corners_p = []
            
            # Loop over pyramid levels
            for i_p in range(num_pyramid_levels):
                # Append location scores, class scores, feature map sizes and transform corners for each pyramid level
                if loc_scores is not None and loc_scores[0] is not None and loc_scores[0][0] is not None:
                    image_loc_scores_p.append(torch.cat([s[i_image_in_batch] for s in loc_scores[i_p]], 0))
                else:
                    image_loc_scores_p.append(None)
                image_class_scores_p.append(torch.cat([s[i_image_in_batch] for s in class_scores[i_p]], 0))

                if transform_corners is not None and transform_corners[0] is not None and transform_corners[0][0] is not None:
                    transform_corners_p.append(torch.cat([s[i_image_in_batch] for s in transform_corners[i_p]], 0))
                else:
                    transform_corners_p.append(None)

                image_fm_sizes_p.append(fm_sizes[i_p][0])

            # Get the images for each pyramid level
            one_image_pyramid = [p[i_image_in_batch] for p in batch_images_pyramid]

            # Get the box reverse transforms for the current image
            box_reverse_transforms = box_transforms_batch[i_image_in_batch]

            # Log timing information
            logger.info(timing_str + "Net time: {0}".format(time_since(t_start_batch)))
            
            yield image_id, image_loc_scores_p, image_class_scores_p, one_image_pyramid,\
                  batch_query_img_sizes, batch_class_ids, box_reverse_transforms, image_fm_sizes_p, transform_corners_p
