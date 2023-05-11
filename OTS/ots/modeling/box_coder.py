import math
import itertools
from functools import lru_cache

import torch

from torchvision.models.detection._utils import Matcher, BoxCoder, encode_boxes

from ots.structures.bounding_box import BoxList, cat_boxlist, boxlist_iou, nms


BOX_ENCODING_WEIGHTS = torch.tensor([10, 10, 5, 5])


@lru_cache()
def create_strided_boxes_columnfirst(grid_size, box_size, box_stride):
    h = torch.arange(0, grid_size.h, dtype=torch.float)
    cy = (h + 0.5) * box_stride.h
    w = torch.arange(0, grid_size.w, dtype=torch.float)
    cx = (w + 0.5) * box_stride.w

    cx = cx.unsqueeze(0).expand(cy.size(0), -1).contiguous()
    cy = cy.unsqueeze(1).expand(-1, cx.size(1)).contiguous()
    cx = cx.view(-1)
    cy = cy.view(-1)

    sx = torch.FloatTensor( [box_size.w] ).expand_as(cx)
    sy = torch.FloatTensor( [box_size.h] ).expand_as(cy)

    boxes_cXcYWH = torch.stack( [cx, cy, sx, sy], dim=1 )

    boxes_xyxy = BoxList.convert_bbox_format(boxes_cXcYWH, "cx_cy_w_h", "xyxy")
    return boxes_xyxy

def masked_select_or_fill_constant(a, mask, constant=0):
    constant_tensor = torch.tensor([constant], dtype=a.dtype, device=a.device)
    return torch.where(mask, a, constant_tensor)

class BoxGridGenerator:
    def __init__(self, box_size, box_stride):
        self.box_size = box_size
        self.box_stride = box_stride

    def create_strided_boxes_columnfirst(self, fm_size):
        return create_strided_boxes_columnfirst(fm_size, self.box_size, self.box_stride)

    def  get_box_to_cut_anchor(self, img_size, crop_size, fm_size, default_box_transform=None):
        anchor_index = torch.arange( fm_size.h * fm_size.w )
        anchor_y_index = anchor_index // fm_size.w
        anchor_x_index = anchor_index % fm_size.w

        cx = (anchor_x_index.float() + 0.5) * self.box_stride.w
        cy = (anchor_y_index.float() + 0.5) * self.box_stride.h

        box_left = cx - crop_size.w / 2
        box_top = cy - crop_size.h / 2

        anchor_box = torch.stack([cx, cy, torch.full_like(cx, self.box_size.w), torch.full_like(cx, self.box_size.h)], 1)
        anchor_box = BoxList.convert_bbox_format(anchor_box, "cx_cy_w_h", "xyxy")

        def floor_to_stride(pos, stride):
            return (torch.floor(pos) // stride) * stride

        def ceil_to_stride(pos, stride):
            return torch.floor(torch.ceil(torch.floor(pos) / stride)) * stride

        box_left = masked_select_or_fill_constant(floor_to_stride(box_left, self.box_stride.w), box_left > 0, 0)
        box_top = masked_select_or_fill_constant(floor_to_stride(box_top, self.box_stride.h), box_top > 0, 0)

        box_right = box_left + crop_size.w
        box_bottom = box_top + crop_size.h

        mask_have_to_move_right = box_left < 0
        box_right[mask_have_to_move_right] -= box_left[mask_have_to_move_right]
        box_left[mask_have_to_move_right] = 0

        mask = box_right > img_size.w
        shift_left = ceil_to_stride(box_right - img_size.w, self.box_stride.w)
        mask_good_fit = (box_left - shift_left >= 0)

        box_left[mask & mask_good_fit] -= shift_left[mask & mask_good_fit]
        box_right[mask & mask_good_fit] -= shift_left[mask & mask_good_fit]

        box_left[ mask & ~mask_good_fit ] = 0
        box_right[ mask & ~mask_good_fit ] = crop_size.w

        
        mask_have_to_move_down = box_top < 0
        box_bottom[mask_have_to_move_down] -= box_top[mask_have_to_move_down]
        box_top[mask_have_to_move_down] = 0

        mask = box_bottom > img_size.h
        shift_up = ceil_to_stride(box_bottom - img_size.h, self.box_stride.h)
        mask_good_fit = (box_top - shift_up >= 0)

        box_top[mask & mask_good_fit] -= shift_up[mask & mask_good_fit]
        box_bottom[mask & mask_good_fit] -= shift_up[mask & mask_good_fit]

        box_top[ mask & ~mask_good_fit ] = 0
        box_bottom[ mask & ~mask_good_fit ] = crop_size.h

        crop_boxes_xyxy = torch.stack([box_left, box_top, box_right, box_bottom], 1) 
        crop_boxes_xyxy = BoxList(crop_boxes_xyxy, img_size, mode="xyxy")
        anchor_box = BoxList(anchor_box, img_size, mode="xyxy")
        if default_box_transform is not None:
            crop_boxes_xyxy = default_box_transform(crop_boxes_xyxy)
            anchor_box = default_box_transform(anchor_box)

        return crop_boxes_xyxy, anchor_box, anchor_index


class OTSBoxCoder:
    def __init__(self, positive_iou_threshold, negative_iou_threshold,
                       remap_classification_targets_iou_pos, remap_classification_targets_iou_neg,
                       output_box_grid_generator, function_get_feature_map_size,
                       do_nms_across_classes=False):
        self.get_feature_map_size = function_get_feature_map_size
        self.output_box_grid_generator = output_box_grid_generator
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.remap_classification_targets_iou_pos = remap_classification_targets_iou_pos
        self.remap_classification_targets_iou_neg = remap_classification_targets_iou_neg
        self.do_nms_across_classes = do_nms_across_classes

        self.weights = BOX_ENCODING_WEIGHTS
        self.box_coder = BoxCoder(self.weights)
        self.matcher = Matcher(self.positive_iou_threshold,
                               self.negative_iou_threshold)
        self.matcher_remap = Matcher(self.remap_classification_targets_iou_pos,
                                     self.remap_classification_targets_iou_neg)

    def _get_default_boxes(self, img_size):
        feature_map_size = self._get_feature_map_size_per_image_size(img_size)
        boxes_xyxy = self.output_box_grid_generator.create_strided_boxes_columnfirst(feature_map_size)
        boxes_xyxy = BoxList(boxes_xyxy, image_size=img_size, mode="xyxy")
        return boxes_xyxy

    @lru_cache()
    def _get_feature_map_size_per_image_size(self, img_size):
        return self.get_feature_map_size(img_size)

    @staticmethod
    def assign_anchors_to_boxes_threshold(detection_boxes, annotation_boxes, matcher):
        ious = boxlist_iou(annotation_boxes, detection_boxes)
        index = matcher(ious)
        class_difficult_flags = annotation_boxes.get_field("difficult")
        good_index_mask = index >= 0
        if good_index_mask.any():
            good_index = good_index_mask.nonzero()
            difficult_mask = class_difficult_flags[index[good_index]]
            difficult_index = good_index[difficult_mask]
            index[difficult_index] = -2

        return index, ious

    def remap_anchor_targets(self, loc_scores, batch_img_size, class_image_sizes, batch_boxes,
                             box_reverse_transform=None):
        cls_targets_remapped = []
        ious_anchor_corrected = []
        ious_anchor = []
        for i_image in range(loc_scores.size(0)):
            default_boxes_xyxy = self._get_default_boxes(batch_img_size[i_image]) 
            image_cls_targets_remapped = []
            image_ious_anchor_corrected = []
            image_ious_anchor = []
            for i_label in range(loc_scores.size(1)):
                cur_loc_scores = loc_scores[i_image, i_label].transpose(0,1)  
                cur_default_boxes_xyxy = default_boxes_xyxy.to(cur_loc_scores) 
                box_predictions = self.build_boxes_from_loc_scores(cur_loc_scores, cur_default_boxes_xyxy) 

                if box_reverse_transform is not None:
                    box_predictions = box_reverse_transform[i_image](box_predictions)
                    cur_default_boxes_xyxy = box_reverse_transform[i_image](cur_default_boxes_xyxy)
                
                cur_labels = batch_boxes[i_image].get_field("labels")
                label_mask = cur_labels == i_label
                ids = torch.nonzero(label_mask).view(-1)
                device = box_predictions.bbox_xyxy.device
                if ids.numel() > 0:
                    class_boxes = batch_boxes[i_image][ids].to(device=device)

                    _, ious = self.assign_anchors_to_boxes_threshold(cur_default_boxes_xyxy,
                                                                     class_boxes,
                                                                     self.matcher_remap)
                    ious_anchors_max_gt = ious.max(0)[0] 

                    index, ious = self.assign_anchors_to_boxes_threshold(box_predictions,
                                                                         class_boxes,
                                                                         self.matcher_remap)
                    ious_corrected_max_gt = ious.max(0)[0] 
                    image_class_cls_targets_remapped = 1 + index.clamp(min=-2, max=0)
                else:
                    image_class_cls_targets_remapped = torch.LongTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                    ious_anchors_max_gt = torch.FloatTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                    ious_corrected_max_gt = torch.FloatTensor(len(cur_default_boxes_xyxy)).zero_().to(device=device)
                image_cls_targets_remapped.append(image_class_cls_targets_remapped)
                image_ious_anchor_corrected.append(ious_corrected_max_gt)
                image_ious_anchor.append(ious_anchors_max_gt)

            image_cls_targets_remapped = torch.stack(image_cls_targets_remapped, 0)  
            cls_targets_remapped.append(image_cls_targets_remapped)

            image_ious_anchor_corrected = torch.stack(image_ious_anchor_corrected, 0)  
            ious_anchor_corrected.append(image_ious_anchor_corrected)
            image_ious_anchor = torch.stack(image_ious_anchor, 0)  
            ious_anchor.append(image_ious_anchor)
        
        cls_targets_remapped = torch.stack(cls_targets_remapped, 0) 
        
        ious_anchor_corrected = torch.stack(ious_anchor_corrected, 0) 
        ious_anchor = torch.stack(ious_anchor, 0) 

        return cls_targets_remapped, ious_anchor, ious_anchor_corrected
        
    @staticmethod
    def build_loc_targets(class_boxes, default_boxes):
        class_boxes.clip_to_min_size(min_size=1)
        default_boxes.clip_to_min_size(min_size=1)
        class_loc_targets = encode_boxes(class_boxes.bbox_xyxy, default_boxes.bbox_xyxy, BOX_ENCODING_WEIGHTS)
        return class_loc_targets

    def build_boxes_from_loc_scores(self, loc_scores, default_boxes):
        box_preds = self.box_coder.decode_single(loc_scores, default_boxes.bbox_xyxy)
        return BoxList(box_preds, image_size=default_boxes.image_size, mode="xyxy")

    def encode(self, boxes, img_size, num_labels, default_box_transform=None):
        difficult_flags = boxes.get_field("difficult")
        labels = boxes.get_field("labels")

        default_boxes = self._get_default_boxes(img_size)
        if default_box_transform is not None:
            default_boxes = default_box_transform(default_boxes)

        loc_targets = []
        cls_targets = []
        for i_label in range(num_labels):

            label_mask = labels == i_label
            ids = torch.nonzero(label_mask).view(-1)

            if ids.numel() > 0:
                class_boxes = boxes[ids]

                index, ious = self.assign_anchors_to_boxes_threshold(default_boxes, class_boxes, self.matcher)
                ious_max_gt = ious.max(0)[0] 
                class_boxes = class_boxes[index.clamp(min=0)] 

                class_loc_targets = self.build_loc_targets(class_boxes, default_boxes)
                class_cls_targets = 1 + index.clamp(min=-2, max=0)
            else:
                class_loc_targets = torch.zeros(len(default_boxes), 4, dtype=torch.float)
                class_cls_targets = torch.zeros(len(default_boxes), dtype=torch.long)

            loc_targets.append(class_loc_targets.transpose(0, 1).contiguous()) 
            cls_targets.append(class_cls_targets)

        loc_targets = torch.stack(loc_targets, 0)
        cls_targets = torch.stack(cls_targets, 0)

        return loc_targets, cls_targets

    def encode_pyramid(self, boxes, img_size_pyramid, num_labels,
                       default_box_transform_pyramid=None):
        num_pyramid_levels = len(img_size_pyramid)
        
        loc_targets_pyramid = []
        cls_targets_pyramid = []
        for i_p in range(num_pyramid_levels):
            loc_targets_this_level, cls_targets_this_level = \
                    self.encode(boxes, img_size_pyramid[i_p], num_labels,
                                default_box_transform=default_box_transform_pyramid[i_p])
            loc_targets_pyramid.append(loc_targets_this_level)
            cls_targets_pyramid.append(cls_targets_this_level)

        return loc_targets_pyramid, cls_targets_pyramid

    @staticmethod
    def _nms_box_lists(boxlists, nms_iou_threshold):
        boxes = cat_boxlist(boxlists)
        scores = boxes.get_field("scores")
    
        ids_boxes_keep = nms(boxes, nms_iou_threshold)
        
        scores = scores[ids_boxes_keep]
        _, score_sorting_index = torch.sort(scores, dim=0, descending=True)

        ids_boxes_keep = ids_boxes_keep[score_sorting_index]

        return boxes[ids_boxes_keep]

    @staticmethod
    def apply_transform_to_corners(masked_transform_corners, transform, img_size):
        masked_transform_corners = masked_transform_corners.contiguous().view(-1, 4)
        corners_as_boxes = BoxList(masked_transform_corners, img_size, mode="xyxy")
        corners_as_boxes = transform(corners_as_boxes)
        masked_transform_corners = corners_as_boxes.bbox_xyxy.contiguous().view(-1, 8)
        return masked_transform_corners

    def decode_pyramid(self, loc_scores_pyramid, cls_scores_pyramid, img_size_pyramid, class_ids,
               nms_score_threshold=0.0, nms_iou_threshold=0.3,
               inverse_box_transforms=None, transform_corners_pyramid=None):
        num_classes = len(class_ids)
        num_pyramid_levels = len(img_size_pyramid)
        default_boxes_per_level = [self._get_default_boxes(img_size) for img_size in img_size_pyramid]

        device = cls_scores_pyramid[0].device
        for cl, loc in zip (cls_scores_pyramid, loc_scores_pyramid):
            assert cl.device == device, "scores and boxes should be on the same device"
            assert loc.device == device, "scores and boxes should be on the same device"

        boxes_per_label = []
        transform_corners_per_label = []

        for real_label in set(class_ids):
            masked_boxes_pyramid, masked_score_pyramid, masked_default_boxes_pyramid, masked_labels_pyramid = [], [], [], []
            masked_transform_corners_pyramid = []
            for i_label in range(num_classes):
                if class_ids[i_label] != real_label:
                    continue
                for i_p, (loc_scores, cls_scores) in enumerate(zip(loc_scores_pyramid, cls_scores_pyramid)):
                    default_boxes = default_boxes_per_level[i_p]
                    default_boxes = default_boxes.to(device=device)

                    box_preds = self.build_boxes_from_loc_scores(loc_scores[i_label].transpose(0,1), default_boxes)
                    box_preds.add_field("scores", cls_scores[i_label, :].float())
                    box_preds.add_field("default_boxes", default_boxes)
                    box_preds.add_field("labels", torch.zeros(len(box_preds), dtype=torch.long, device=device).fill_(int(real_label)))

                    if transform_corners_pyramid is not None:
                        box_preds.add_field("transform_corners", transform_corners_pyramid[i_p][i_label].transpose(0,1))

                    assert img_size_pyramid[i_p] == box_preds.image_size
                    box_preds.clip_to_image(remove_empty=False)
                    bad_boxes = box_preds.get_mask_empty_boxes()
                    mask = (box_preds.get_field("scores").float() > nms_score_threshold) & ~bad_boxes
                    if mask.any():
                        masked_boxes = box_preds[mask]
                        if inverse_box_transforms is not None:
                            img_size = masked_boxes.image_size
                            masked_boxes = inverse_box_transforms[i_p](masked_boxes)
                            masked_boxes.add_field("default_boxes",
                                                   inverse_box_transforms[i_p](masked_boxes.get_field("default_boxes")))
                            if masked_boxes.has_field("transform_corners"):
                                masked_transform_corners = masked_boxes.get_field("transform_corners")
                                masked_transform_corners = self.apply_transform_to_corners(masked_transform_corners, inverse_box_transforms[i_p], img_size)
                                masked_boxes.add_field("transform_corners", masked_transform_corners)

                        masked_boxes_pyramid.append(masked_boxes)

            if len(masked_boxes_pyramid) > 0:
                boxes_after_nms = self._nms_box_lists(masked_boxes_pyramid, nms_iou_threshold)
                boxes_per_label.append(boxes_after_nms)
            
        if self.do_nms_across_classes:
            boxes_stacked = \
                self._nms_box_lists(boxes_per_label, nms_iou_threshold)
        else:
            boxes_stacked = cat_boxlist(boxes_per_label)

        return boxes_stacked
