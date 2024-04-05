import random

from vgts.structures.transforms import random_distort, crop


class DataAugmentation():
    def __init__(self, random_flip_batches, random_crop_size, random_crop_scale, jitter_aspect_ratio, scale_jitter, random_color_distortion, random_crop_label_images, min_box_coverage):
        # Initialize the DataAugmentation class with the given parameters
        self.batch_random_hflip = self.batch_random_vflip = random_flip_batches
        self.do_random_color = random_color_distortion
        self.brightness_delta, self.contrast_delta, self.saturation_delta, self.hue_delta = 32/255., 0.5, 0.5, 0.1
        self.scale_jitter, self.jitter_aspect_ratio = scale_jitter, jitter_aspect_ratio
        self.do_random_crop = True if random_crop_size else False
        if self.do_random_crop:
            self.random_crop_size, self.random_crop_scale = random_crop_size, random_crop_scale
            self.random_interpolation, self.coverage_keep_threshold, self.coverage_remove_threshold, self.max_trial, self.min_box_coverage = True, 0.7, 0.3, 100, min_box_coverage
        self.do_random_crop_label_images = random_crop_label_images

    def random_distort(self, img):
        # If color distortion is enabled, apply random distortions to the image
        if self.do_random_color:
            img = random_distort(img, brightness_delta=self.brightness_delta, contrast_delta=self.contrast_delta, saturation_delta=self.saturation_delta, hue_delta=self.hue_delta)
        return img

    def random_crop(self, img, boxes=None, transform_list=None):
        # If random cropping is enabled, apply random cropping to the image
        if not self.do_random_crop: raise(RuntimeError("Random crop data augmentation is not initialized"))
        return self.crop_image(img, crop_position=None, boxes=boxes, transform_list=transform_list, random_crop_size=self.random_crop_size)

    def crop_image(self, img, crop_position, boxes=None, transform_list=None, random_crop_size=None):
        # Crop the image and update the bounding boxes, mask cutoff boxes and mask difficult boxes
        img, boxes, mask_cutoff_boxes, mask_difficult_boxes = crop(img, crop_position=crop_position, random_crop_size=random_crop_size, random_crop_scale=self.random_crop_scale, crop_size=self.random_crop_size, scale_jitter=self.scale_jitter, jitter_aspect_ratio=self.jitter_aspect_ratio, coverage_keep_threshold=self.coverage_keep_threshold, coverage_remove_threshold=self.coverage_remove_threshold, max_trial=self.max_trial, min_box_coverage=self.min_box_coverage, boxes=boxes, transform_list=transform_list)
        return img, boxes, mask_cutoff_boxes, mask_difficult_boxes

    def random_crop_label_image(self, img):
        # If random cropping for label images is enabled, apply random cropping to the label image
        if self.do_random_crop_label_images:
            new_ar = random.uniform(img.size[0] / img.size[1] * self.jitter_aspect_ratio, img.size[0] / img.size[1] / self.jitter_aspect_ratio)
            random_crop_size = (int(min(img.size[0], img.size[1] * new_ar)), int(min(img.size[0] / new_ar, img.size[1])))
            img = self.crop_image(img, None, random_crop_size=random_crop_size)[0]
        return img
