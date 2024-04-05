import os
import random
import time, datetime
import math
import copy
import logging
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from vgts.utils import add_to_meters_in_dict, log_meters, print_meters, time_since, get_trainable_parameters, checkpoint_model, init_log
from .evaluate import evaluate, make_iterator_extract_scores_from_images_batched
from .optimization import setup_lr, get_learning_rate, set_learning_rate
from vgts.structures.bounding_box import nms, cat_boxlist
from vgts.structures.feature_map import FeatureMapSize

# Function to prepare batch data, possibly moving it to GPU and logging the size of the batch
def prepare_batch_data(batch_data, is_cuda, logger):
    # Unpack batch data into variables
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, batch_box_inverse_transform, batch_boxes, batch_img_size = batch_data
    
    # If CUDA is available and specified, move the data to GPU
    if is_cuda:
        # Move images, loc_targets, and class_targets to GPU
        images, loc_targets, class_targets = images.cuda(), loc_targets.cuda(), class_targets.cuda()
        # Move each image in class_images to GPU
        class_images = [im.cuda() for im in class_images]

    # Log the size of the images and the number of classes
    logger.info(f"{images.size(0)} imgs, {len(class_images)} classes")

    # Return the prepared data
    return images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, batch_box_inverse_transform, batch_boxes, batch_img_size

    
# Function to train the model on one batch of data
def train_one_batch(batch_data, net, cfg, criterion, optimizer, dataloader, logger):
    # Record the start time of the batch
    t_start_batch = time.time()
    
    # Set the model to training mode, possibly freezing certain parts of the model
    net.train(freeze_bn_in_extractor=cfg.train.model.freeze_bn, 
              freeze_transform_params=cfg.train.model.freeze_transform, 
              freeze_bn_transform=cfg.train.model.freeze_bn_transform)
    
    # Zero the gradients of the optimizer
    optimizer.zero_grad()

    # Prepare the batch data, possibly moving it to GPU
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, batch_box_inverse_transform, batch_boxes, batch_img_size = prepare_batch_data(batch_data, cfg.is_cuda, logger)

    # Forward pass of the model, get the scores and other outputs
    loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = net(images, class_images, train_mode=True, fine_tune_features=cfg.train.model.train_features)

    # Remap anchor targets for computing loss
    cls_targets_remapped, ious_anchor, ious_anchor_corrected = dataloader.box_coder.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)

    # Compute losses
    losses = criterion(loc_scores, loc_targets, class_scores, class_targets, 
                       cls_targets_remapped=cls_targets_remapped, 
                       cls_preds_for_neg=class_scores_transform_detached if not cfg.train.model.train_transform_on_negs else None)

    # Backpropagate the main loss
    main_loss = losses["loss"]
    main_loss.backward()

    # Copy the gradients from GPU to CPU and compute the gradient norm
    grad = OrderedDict((name, param.grad.clone().cpu()) for name, param in net.named_parameters() if param.requires_grad and param.grad is not None)
    grad_norm = torch.nn.utils.clip_grad_norm_(get_trainable_parameters(net), cfg.train.optim.max_grad_norm, norm_type=2)
    
    # If the gradient norm is not NaN, perform an optimization step
    if not math.isnan(grad_norm):
        optimizer.step()
    else:
        # If the gradient norm is NaN, save a dump of the current state and log an error
        batch_data[6] = None
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        dump_file = f"error_nan_appeared-{time_stamp}.pth"
        if cfg.output.path:
            dump_file = os.path.join(cfg.output.path, dump_file)
        logger.error(f"gradient is NaN. Saving dump to {dump_file}")
        torch.save({"batch_data": batch_data, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(), "cfg": cfg, "grad": grad}, dump_file)

    # Record the mean loss for each type of loss, the gradient norm, and the time it took to process this batch
    meters = {l: losses[l].mean().item() for l in losses}
    meters["grad_norm"] = grad_norm
    meters["batch_time"] = time.time() - t_start_batch
    
    # Return the metrics for this batch
    return meters



@torch.no_grad()
def mine_hard_patches(dataloader, net, cfg, criterion):
    # Get logger instance
    logger = logging.getLogger("VGTS.mining_hard_patches")
    logger.info("Starting to mine hard patches")
    t_start_mining = time.time()

    # Set model to evaluation mode
    net.eval()

    # Get the number of batches in the dataloader
    num_batches = len(dataloader)

    # Initialize a dictionary to store hard negative examples per image ID
    hardnegdata_per_imageid = OrderedDict()

    # Create an iterator that extracts scores from images in batches
    iterator = make_iterator_extract_scores_from_images_batched(dataloader, net, logger,
                                                                image_batch_size=cfg.eval.batch_size,
                                                                is_cuda=cfg.is_cuda,
                                                                num_random_pyramid_scales=cfg.train.mining.num_random_pyramid_scales,
                                                                num_random_negative_labels=cfg.train.mining.num_random_negative_classes)

    # Initialize lists to store boxes and losses
    boxes = []
    gt_boxes = []
    losses = OrderedDict()

    # Iterate over data in the iterator
    for data in iterator:
        t_item_start = time.time()

        # Unpack the data
        image_id, image_loc_scores_pyramid, image_class_scores_pyramid, image_pyramid, query_img_sizes, batch_class_ids, box_reverse_transform_pyramid, image_fm_sizes_p, transform_corners_pyramid = data

        # Calculate the image size pyramid
        img_size_pyramid = [FeatureMapSize(img=image) for image in image_pyramid]

        # Get the ground truth boxes for the current image
        gt_boxes_one_image = dataloader.get_image_annotation_for_imageid(image_id)
        gt_boxes.append(gt_boxes_one_image)

        # Update the box labels to local
        dataloader.update_box_labels_to_local(gt_boxes_one_image, batch_class_ids)

        # Get the number of labels
        num_labels = len(batch_class_ids)

        # Encode the pyramid of ground truth boxes
        loc_targets_pyramid, class_targets_pyramid = dataloader.box_coder.encode_pyramid(gt_boxes_one_image, img_size_pyramid, num_labels, default_box_transform_pyramid=box_reverse_transform_pyramid)

        # If using CUDA, move data to GPU
        if cfg.is_cuda:
            loc_targets_pyramid = [loc_targets.cuda() for loc_targets in loc_targets_pyramid]
            class_targets_pyramid = [class_targets.cuda() for class_targets in class_targets_pyramid]

        # Add a batch dimension to the location scores pyramid
        add_batch_dim = lambda list_of_tensors: [t.unsqueeze(0) for t in list_of_tensors]
        loc_scores_pyramid = add_batch_dim(image_loc_scores_pyramid)

        # Remap the class targets for each pyramid level
        cls_targets_remapped_pyramid = []
        for loc_scores, img_size, box_reverse_transform in zip(loc_scores_pyramid, img_size_pyramid, box_reverse_transform_pyramid):
            cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
                dataloader.box_coder.remap_anchor_targets(loc_scores, [img_size], query_img_sizes, [gt_boxes_one_image],
                                                          box_reverse_transform=[box_reverse_transform])

            cls_targets_remapped_pyramid.append(cls_targets_remapped)

        # Compute losses for the current iteration and per anchor
        losses_iter, losses_per_anchor = criterion(loc_scores_pyramid,
                                                    add_batch_dim(loc_targets_pyramid),
                                                    add_batch_dim(image_class_scores_pyramid),
                                                    add_batch_dim(class_targets_pyramid),
                                                    cls_targets_remapped=cls_targets_remapped_pyramid,
                                                    patch_mining_mode=True)
        # Assert that data augmentation is set, as hard patches can only be mined through data augmentation
        assert dataloader.data_augmentation is not None, "Can mine hard patches only through data augmentation"
        
        # Get the crop size from the data augmentation configuration
        crop_size = dataloader.data_augmentation.random_crop_size
        
        # Calculate the mean loss for each loss in the losses of the current iteration
        for l in losses_iter:
            losses_iter[l] = losses_iter[l].mean().item()
            
        # Print the losses of the current iteration
        print_meters(losses_iter, logger)
        
        # Add the losses of the current iteration to the total losses
        add_to_meters_in_dict(losses_iter, losses)
        
        # Get the feature map sizes for each image size in the current batch
        query_fm_sizes = [dataloader.box_coder._get_feature_map_size_per_image_size(sz) for sz in query_img_sizes]
        
        # Initialize lists to store crops, anchors, labels, pyramid levels, losses, corners, masks and indices
        crops = []
        achors = []
        labels_of_anchors = []
        pyramid_level_of_anchors = []
        losses_of_anchors = []
        corners_of_anchors = []
        losses_loc_of_anchors = []
        pos_mask_of_anchors = []
        pos_loc_mask_of_anchors = []
        neg_mask_of_anchors = []
        anchor_indices = []
        
        # Set image in batch index to 0
        i_image_in_batch = 0
        
        # For each pyramid level and image size, get the crop and anchor positions, and update the relevant lists
        for i_p, img_size in enumerate(img_size_pyramid):
            for i_label, query_fm_size in enumerate(query_fm_sizes):
                crop_position, anchor_position, anchor_index = \
                    dataloader.box_coder.output_box_grid_generator.get_box_to_cut_anchor(img_size,
                                                                                         crop_size,
                                                                                         image_fm_sizes_p[i_p],
                                                                                         box_reverse_transform_pyramid[i_p])
                # Compute corners of the anchor
                cur_corners = transform_corners_pyramid[i_p][i_label].transpose(0,1)
                cur_corners = dataloader.box_coder.apply_transform_to_corners(cur_corners, box_reverse_transform_pyramid[i_p], img_size)
                # If using CUDA, move crop and anchor positions to GPU
                if cfg.is_cuda:
                    crop_position, anchor_position = crop_position.cuda(), anchor_position.cuda()
                
                # Update the relevant lists
                crops.append(crop_position)
                achors.append(anchor_position)
                device = crop_position.bbox_xyxy.device
                losses_of_anchors.append(losses_per_anchor["cls_loss"][i_p][i_image_in_batch, i_label].to(crop_position.bbox_xyxy))
                pos_mask_of_anchors.append(losses_per_anchor["pos_mask"][i_p][i_image_in_batch, i_label].to(device=device))
                neg_mask_of_anchors.append(losses_per_anchor["neg_mask"][i_p][i_image_in_batch, i_label].to(device=device))
                losses_loc_of_anchors.append(losses_per_anchor["loc_loss"][i_p][i_image_in_batch, i_label].to(crop_position.bbox_xyxy))
                pos_loc_mask_of_anchors.append(losses_per_anchor["pos_for_regression"][i_p][i_image_in_batch, i_label].to(device=device))
                corners_of_anchors.append(cur_corners.to(crop_position.bbox_xyxy))
                
                # Get the number of anchors
                num_anchors = len(crop_position)
                
                # Update labels, pyramid levels and indices
                labels_of_anchors.append(torch.full([num_anchors], i_label, dtype=torch.long))
                pyramid_level_of_anchors.append(torch.full([num_anchors], i_p, dtype=torch.long))
                anchor_indices.append(anchor_index)
                
        # Concatenate all of the generated lists
        crops = cat_boxlist(crops)
        achors = cat_boxlist(achors)
        labels_of_anchors  = torch.cat(labels_of_anchors, 0)
        pyramid_level_of_anchors = torch.cat(pyramid_level_of_anchors, 0)
        losses_of_anchors = torch.cat(losses_of_anchors, 0)
        losses_loc_of_anchors = torch.cat(losses_loc_of_anchors, 0)
        pos_mask_of_anchors = torch.cat(pos_mask_of_anchors, 0)
        pos_loc_mask_of_anchors = torch.cat(pos_loc_mask_of_anchors, 0)
        neg_mask_of_anchors = torch.cat(neg_mask_of_anchors, 0)
        anchor_indices = torch.cat(anchor_indices, 0)
        corners_of_anchors = torch.cat(corners_of_anchors, 0)
        
        # Function for performing NMS (Non-Maximum Suppression) and collecting the remaining data
        def nms_masked_and_collect_data(mask, crops_xyxy, scores, nms_iou_threshold_in_mining, max_etries=None):
            mask_ids = torch.nonzero(mask).squeeze(1)
            boxes_selected = copy.deepcopy(crops_xyxy[mask])
            boxes_selected.add_field("scores", scores[mask])
            remaining_boxes = nms(boxes_selected, nms_iou_threshold_in_mining)
            remaining_boxes = mask_ids[remaining_boxes]
            ids = torch.argsort(scores[remaining_boxes], descending=True)
            if max_etries is not None:
                ids = ids[:max_etries]
            remaining_boxes = remaining_boxes[ids]

            return remaining_boxes
        
        # Get NMS IOU threshold and number of hard patches per image from the configuration
        nms_iou_threshold_in_mining = cfg.train.mining.nms_iou_threshold_in_mining
        num_hard_patches_per_image = cfg.train.mining.num_hard_patches_per_image
        
        # Apply NMS and collect hard negative and positive data
        hard_negs = nms_masked_and_collect_data(neg_mask_of_anchors, crops, losses_of_anchors,
                                                nms_iou_threshold_in_mining,
                                                num_hard_patches_per_image)

        hard_pos  = nms_masked_and_collect_data(pos_mask_of_anchors, crops, losses_of_anchors,
                                                nms_iou_threshold_in_mining,
                                                num_hard_patches_per_image)

        hard_pos_loc  = nms_masked_and_collect_data(pos_loc_mask_of_anchors, crops, losses_loc_of_anchors,
                                                    nms_iou_threshold_in_mining,
                                                    num_hard_patches_per_image)
        
        # Function for standardizing a value (converting a tensor to a Python number)
        def standardize(v):
            return v.item() if type(v) == torch.Tensor else v
        
        # Function for adding an item to the data
        def add_item(data, role, pyramid_level, label_local, anchor_index, crop_position_xyxy, anchor_position_xyxy, transform_corners):
            new_item = OrderedDict()
            new_item["pyramid_level"] = standardize(pyramid_level)
            new_item["label_local"] = standardize(label_local)
            new_item["anchor_index"] = standardize(anchor_index)
            new_item["role"] = role
            new_item["crop_position_xyxy"] = crop_position_xyxy
            new_item["anchor_position_xyxy"] = anchor_position_xyxy
            new_item["transform_corners"] = transform_corners
            data.append(new_item)

        hardnegdata = []
        # Add the hard negative, hard positive, and hard positive localization items to the hardnegdata list
        for i in hard_negs:
            add_item(hardnegdata, "neg", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())
        for i in hard_pos:
            add_item(hardnegdata, "pos", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())
        for i in hard_pos_loc:
            add_item(hardnegdata, "pos_loc", pyramid_level_of_anchors[i],
                        labels_of_anchors[i], anchor_indices[i],
                        crops[i].cpu(), achors[i].cpu(), corners_of_anchors[i].cpu())
        
        # Add additional information (global label, loss, loss location, score, and image id) to the items in the hardnegdata list
        for a in hardnegdata:
            a["label_global"] = standardize(batch_class_ids[ a["label_local"] ])
            a["loss"] = standardize(losses_per_anchor["cls_loss"][a["pyramid_level"]][i_image_in_batch, a["label_local"], a["anchor_index"]])
            a["loss_loc"] = standardize(losses_per_anchor["loc_loss"][a["pyramid_level"]][i_image_in_batch, a["label_local"], a["anchor_index"]])
            a["score"] = standardize(image_class_scores_pyramid[a["pyramid_level"]][a["label_local"], a["anchor_index"]])
            a["image_id"] = standardize(image_id)

        # Add the hard negative data for this image id to the overall list
        hardnegdata_per_imageid[image_id] = hardnegdata
        
        # Log the time taken to process this item and the total time taken since the start of mining
        logger.info("Item time: {0}, since mining start: {1}".format(time_since(t_item_start), time_since(t_start_mining)))
    
    # Log the total time taken for hard negative mining
    logger.info("Hard negative mining finished in {0}".format(time_since(t_start_mining)))
    
    # Return the hard negative data per image id
    return hardnegdata_per_imageid


def evaluate_model(dataloaders, net, cfg, criterion=None, print_per_class_results=False):
    # Initialize an ordered dictionary to store the results
    meters_all = OrderedDict()
    for dataloader in dataloaders:
        if dataloader is not None:
            # Evaluate the model on the data provided by the current dataloader
            meters_val = evaluate(dataloader, net, cfg, criterion=criterion, print_per_class_results=print_per_class_results)
            # Store the results in the meters_all dictionary with the dataloader name as the key
            meters_all[dataloader.get_name()] = meters_val
        else:
            meters_val = None
        
    return meters_all


def trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=[]):
    # Setting up the logger
    logger = logging.getLogger("VGTS.train")
    t_start = time.time()
    
    # Initialize counters and logs
    num_steps_for_logging, meters_running = 0, {}
    full_log = init_log()
    
    # Start the training if the number of iterations is greater than zero and training flag is set
    if cfg.train.optim.max_iter > 0 and cfg.train.do_training:
        logger.info("Start training")
        
        # Setup learning rate and evaluation of the model
        _, anneal_lr_func = setup_lr(optimizer, full_log, cfg.train.optim.anneal_lr, cfg.eval.iter)

        meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion)
        
        # Check and handle best model saving configuration
        if cfg.output.best_model.do_get_best_model:
            assert (cfg.output.best_model.dataset and cfg.output.best_model.dataset in meters_eval) \
                or (len(cfg.eval.dataset_names) > 0 and cfg.eval.dataset_names[0] in meters_eval), \
                "Cannot determine which dataset to use for the best model"
            best_model_dataset_name = cfg.output.best_model.dataset if cfg.output.best_model.dataset else cfg.eval.dataset_names[0]
            best_model_metric = meters_eval[best_model_dataset_name][cfg.output.best_model.metric]

            logger.info(f"Init model is the current best on {best_model_dataset_name} by {cfg.output.best_model.metric}, value {best_model_metric:.4f}")
            if cfg.output.path:
                checkpoint_best_model_name = f"best_model_{best_model_dataset_name}_{cfg.output.best_model.metric}"
                checkpoint_best_model_path = \
                    checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, model_name=checkpoint_best_model_name,
                                          extra_fields={"criterion_dataset": best_model_dataset_name,
                                                        "criterion_metric": cfg.output.best_model.metric,
                                                        "criterion_mode": cfg.output.best_model.mode,
                                                        "criterion_value": best_model_metric,
                                                        "criterion_value_old": None})
            else:
                raise RuntimeError("cfg.output.best_model.do_get_best_model i set to True, but cfg.output.path is not provided, so cannot save best models")
                
        # Check if we need to reload the best model after learning rate annealing
        if cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr and\
           cfg.train.optim.anneal_lr.type != "none":
                assert cfg.output.best_model.do_get_best_model, "cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr was set to True, but cfg.output.best_model.do_get_best_model is False, so there is no best model to reload from"
        
        # Log the initial evaluation metrics
        log_meters(full_log, t_start, -1, cfg.output.path,
                meters_eval=meters_eval,
                anneal_lr=anneal_lr_func)

        if cfg.output.path:
            checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=0)

        i_epoch = 0
        i_batch = len(dataloader_train) 
        
        # Main training loop
        for i_iter in range(cfg.train.optim.max_iter):
            # Shuffle dataloader and increment epoch counter when all data is used
            if i_batch >= len(dataloader_train):
                i_epoch += 1
                i_batch = 0
                dataloader_train.shuffle()
            
            # If hard example mining is enabled and we are at the correct iteration, mine hard examples
            if cfg.train.mining.do_mining and i_iter % cfg.train.mining.mine_hard_patches_iter == 0:
                hardnegdata_per_imageid = mine_hard_patches(dataloader_train, net, cfg, criterion)
                dataloader_train.set_hard_negative_data(hardnegdata_per_imageid)
            
            # Log the current iteration, epoch, and elapsed time
            logger.info(f"Iter {i_iter} ({cfg.train.optim.max_iter}), epoch {i_epoch}, time {time_since(t_start)}")
            
            # Record start time of data loading, load a batch, and calculate loading time
            t_start_loading = time.time()
            batch_data = dataloader_train.get_batch(i_batch)
            t_data_loading = time.time() - t_start_loading
            
            # Increment the batch index and logging step counter
            i_batch += 1
            num_steps_for_logging += 1
            
            # Train the model on the current batch and record the loading time
            meters = train_one_batch(batch_data, net, cfg, criterion, optimizer, dataloader_train, logger)
            meters["loading_time"] = t_data_loading
            
            # Print the current meters if we are at the correct iteration
            if i_iter % cfg.output.print_iter == 0:
                print_meters(meters, logger)
            
            # Add the current meters to the running meters
            add_to_meters_in_dict(meters, meters_running)
            
            # If we are at the correct iteration, evaluate the model and check if the current model is the best model so far
            if (i_iter + 1) % cfg.eval.iter == 0:
                meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion)
                
                # Check if the current model is the best model so far and if so, save it
                if cfg.output.best_model.do_get_best_model:
                    # Calculate current metric and assert that mode is either "max" or "min"
                    cur_metric = meters_eval[best_model_dataset_name][cfg.output.best_model.metric]
                    assert cfg.output.best_model.mode in ["max", "min"], f"cfg.output.best_model.mode should be 'max' or 'min', but have {cfg.output.best_model.mode}"
                    # If current model is the best, log this and possibly save the model
                    if (cfg.output.best_model.mode=="max" and cur_metric > best_model_metric) or \
                       (cfg.output.best_model.mode=="min" and cur_metric < best_model_metric):
                        logger.info(f"New best model on {best_model_dataset_name} by {cfg.output.best_model.metric}, value {cur_metric:.4f}")

                        if cfg.output.path:
                            checkpoint_best_model_path = \
                                checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, model_name=checkpoint_best_model_name,
                                                 extra_fields={"criterion_dataset": best_model_dataset_name,
                                                               "criterion_metric": cfg.output.best_model.metric,
                                                               "criterion_mode": cfg.output.best_model.mode,
                                                               "criterion_value": cur_metric,
                                                               "criterion_value_old": best_model_metric})
                        best_model_metric = cur_metric

                for k in meters_running:
                    meters_running[k] /= num_steps_for_logging
                meters_running["lr"] = get_learning_rate(optimizer)
                if anneal_lr_func:
                    lr = anneal_lr_func(i_iter + 1, anneal_now=i_iter > cfg.train.optim.anneal_lr.initial_patience)
                    flag_changed_lr = lr != meters_running["lr"]
                else:
                    lr = meters_running["lr"]
                    flag_changed_lr = False
                # Possibly adjust the learning rate and reload the best model after learning rate annealing
                if cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr and flag_changed_lr:
                    if cfg.output.best_model.do_get_best_model: 
                        optimizer_state = net.init_model_from_file(checkpoint_best_model_path)
                        if optimizer_state is not None:
                            optimizer.load_state_dict(optimizer_state)
                        set_learning_rate(optimizer, lr)
                        
                # Log the current meters and reset counters
                log_meters(full_log, t_start, i_iter, cfg.output.path,
                        meters_running=meters_running,
                        meters_eval=meters_eval)
                num_steps_for_logging, meters_running = 0, {}
            # If we are at the correct iteration, save the model
            if cfg.output.path and cfg.output.save_iter and i_iter % cfg.output.save_iter == 0:
                checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=i_iter)

    logger.info("Final evaluation")
    meters_eval = evaluate_model(dataloaders_eval, net, cfg, criterion, print_per_class_results=True)
    if cfg.train.optim.max_iter > 0 and cfg.train.do_training:
        log_meters(full_log, t_start, cfg.train.optim.max_iter, cfg.output.path,
                   meters_eval=meters_eval)
        if cfg.output.path:
            checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=cfg.train.optim.max_iter)
