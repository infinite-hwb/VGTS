# import necessary libraries and modules
import os
import argparse

import torch

from vgts.modeling.model import build_VGTS_from_config

from vgts.loaddata.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from vgts.engine.train import trainval_loop
from vgts.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path
from vgts.engine.optimization import create_optimizer
from vgts.config import cfg

# Function for parsing the command line options
def parse_opts():
    parser = argparse.ArgumentParser(description="Training and evaluation of the model")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Merge the config file if provided
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args.config_file

# Function to initialize the logger
def init_logger(cfg, config_file):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("VGTS", output_dir if cfg.output.save_log_to_file else None)
    # If config file is provided, log its contents
    if config_file:
        logger.info(f"Loaded configuration file {config_file}")
        with open(config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    else:
        logger.info("Config file was not provided")
    logger.info(f"Running with config:\n{cfg}")
    # Save the config file if training is to be done
    if output_dir and cfg.train.do_training:
        output_config_path = os.path.join(output_dir, "config.yml")
        logger.info(f"Saving config into: {output_config_path}")
        save_config(cfg, output_config_path)

# Main function for training and evaluation
def main():
    cfg, config_file = parse_opts()
    init_logger(cfg, config_file)
    # Check if cuda is available if it's specified in the config
    if cfg.is_cuda:
        assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
        torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.random_seed, cfg.is_cuda)
    # Build the model, the loss function, and the optimizer from the config
    net, box_coder, criterion, img_normalization, optimizer_state = build_VGTS_from_config(cfg)
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)
    # Get data path and build data loaders for training and evaluation
    data_path = get_data_path()
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(cfg, box_coder, img_normalization, data_path=data_path)
    dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization, datasets_for_eval=datasets_train_for_eval, data_path=data_path)
    # Start the training and evaluation loop
    trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=dataloaders_eval)

# Run the main function if the script is run as a standalone program
if __name__ == "__main__":
    main()
