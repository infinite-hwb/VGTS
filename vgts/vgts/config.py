from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

cfg = CN()


cfg.is_cuda = True
cfg.random_seed = 42

# model
cfg.model = CN()
cfg.model.backbone_arch = "ResNet50"
cfg.model.merge_branch_parameters = True
cfg.model.use_inverse_geom_model = True
cfg.model.use_simplified_affine_model = False
cfg.model.class_image_size = 240 
cfg.model.use_group_norm = False
cfg.model.normalization_mean = [0.485, 0.456, 0.406]
cfg.model.normalization_std = [0.229, 0.224, 0.225]

# init
cfg.init = CN()
cfg.init.model = ""
cfg.init.transform = ""

# Training settings
cfg.train = CN()
cfg.train.do_training = True
cfg.train.batch_size = 4 
cfg.train.class_batch_size = 15
cfg.train.dataset_name = "db-train"
cfg.train.dataset_scale = 2500      
cfg.train.cache_images = True

cfg.train.objective = CN()
cfg.train.objective.class_objective = "Torus"
cfg.train.objective.neg_margin = 0.5
cfg.train.objective.pos_margin = 0.6
cfg.train.objective.loc_weight = 0.2
cfg.train.objective.positive_iou_threshold = 0.5
cfg.train.objective.negative_iou_threshold = 0.1
cfg.train.objective.neg_to_pos_ratio = 3
cfg.train.objective.class_neg_weight = 1.0
cfg.train.objective.rll_neg_weight_ratio = 0.001
cfg.train.objective.remap_classification_targets = True
cfg.train.objective.remap_classification_targets_iou_pos = 0.8
cfg.train.objective.remap_classification_targets_iou_neg = 0.4

# Choose which parts of the model to train
cfg.train.model = CN()
cfg.train.model.train_features = True
cfg.train.model.freeze_bn = True
cfg.train.model.freeze_bn_transform = True
cfg.train.model.freeze_transform = False
cfg.train.model.num_frozen_extractor_blocks = 0
cfg.train.model.train_transform_on_negs = False

# data augmentation
cfg.train.augment = CN()
cfg.train.augment.train_patch_width = 800     
cfg.train.augment.train_patch_height = 800    
cfg.train.augment.scale_jitter = 0.7
cfg.train.augment.jitter_aspect_ratio = 0.9
cfg.train.augment.random_flip_batches = False  
cfg.train.augment.random_color_distortion = False
cfg.train.augment.random_crop_class_images = False
cfg.train.augment.min_box_coverage = 0.7
cfg.train.augment.mine_extra_class_images = False

# hard example mining
cfg.train.mining = CN()
# do it or not
cfg.train.mining.do_mining = False
cfg.train.mining.mine_hard_patches_iter = 5000
cfg.train.mining.num_hard_patches_per_image = 10
cfg.train.mining.num_random_pyramid_scales = 2
cfg.train.mining.num_random_negative_classes = 200
cfg.train.mining.nms_iou_threshold_in_mining = 0.5

# optimization
cfg.train.optim = CN()
cfg.train.optim.lr = 1e-4
cfg.train.optim.max_iter = 20000

# Optimizer
cfg.train.optim.optim_method = "sgd"
cfg.train.optim.weight_decay = 1e-4
cfg.train.optim.sgd_momentum = 0.9
cfg.train.optim.max_grad_norm = 1e+2
cfg.train.optim.anneal_lr = CN()
cfg.train.optim.anneal_lr.type = "none"
cfg.train.optim.anneal_lr.milestones = []
cfg.train.optim.anneal_lr.gamma = 0.1
cfg.train.optim.anneal_lr.quantity_to_monitor = "mAP@0.50_db-val-new-cl"
cfg.train.optim.anneal_lr.quantity_mode = "max"
cfg.train.optim.anneal_lr.quantity_epsilon = 1e-2
cfg.train.optim.anneal_lr.reduce_factor = 0.5
cfg.train.optim.anneal_lr.min_value = 1e-5
cfg.train.optim.anneal_lr.patience = 1000
cfg.train.optim.anneal_lr.initial_patience = 0
cfg.train.optim.anneal_lr.cooldown = 10000
cfg.train.optim.anneal_lr.quantity_smoothness = 2000
cfg.train.optim.anneal_lr.reload_best_model_after_anneal_lr = True

# Evaluation setting
cfg.eval = CN()
cfg.eval.iter = 5000
cfg.eval.dataset_names = ["db-val-new-cl", "db-val-old-cl"]
cfg.eval.dataset_scales = [2500]                                                                                              
cfg.eval.cache_images = False

# Its multiscale evaluation
cfg.eval.scales_of_image_pyramid = [1]
cfg.eval.train_subset_for_eval_size = 0

cfg.eval.nms_iou_threshold = 0.3
cfg.eval.nms_score_threshold = float("-inf")
cfg.eval.nms_across_classes = False
cfg.eval.mAP_iou_thresholds = [0.5]
cfg.eval.batch_size = 1
cfg.eval.class_image_augmentation = ""

# Logging parameters
cfg.output = CN()
cfg.output.path = ""
cfg.output.save_log_to_file = False
cfg.output.print_iter = 1
cfg.output.save_iter = 50000

# checkpoint the best model
cfg.output.best_model = CN()
cfg.output.best_model.do_get_best_model = False
cfg.output.best_model.dataset = ""
cfg.output.best_model.metric = "mAP@0.50"
cfg.output.best_model.mode = "max"