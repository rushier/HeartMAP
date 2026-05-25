import argparse
import os
import yaml
import shutil
from src.engine_joint import Engine
# from src.engine_ecg import Engine  ########################
import logging

from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from hubert_ecg import HuBERTECG, HuBERTECGConfig


def run(test_mode):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="./logs/temp",
        help="Directory to save config and model checkpoint",
    )
    parser.add_argument(
        "--config_path",
        default="./configs/default.yml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="indicates whether we are only testing",
    )
    parser.add_argument(
        "--sweep",
        default=False,
        action="store_true",
        help="indicates whether this is a sweep run",
    )
    args = parser.parse_args()

    # Create the save directory
    for patch_str in ["patch_level", "frame_level", "vid_level", "test_vis"]:
        os.makedirs(
            os.path.join(args.save_dir, "visualizations", patch_str), exist_ok=True
        )

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Copy the provided config file into save_dir
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    # Create the logger
    logging.basicConfig(
        filename=os.path.join(args.save_dir, "log.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("heart-transformer")

    if args.test:
        # Create the engine taking care of building different components and starting training/inference
        config['model']['checkpoint_path'] = args.save_dir + '/best/checkpoint_best_loss.pth'
        #config['model']['checkpoint_path'] = args.save_dir + '/best/checkpoint_last_246.pth'
        #config['model']['checkpoint_path'] = args.save_dir + '/best/checkpoint_last_241.pth'
        # config['model']['checkpoint_path'] = args.save_dir + '/checkpoint_best.pth'
        engine = Engine(
        config=config,
        save_dir=args.save_dir+'/'+config['model']['checkpoint_path'].split('_')[-1].split('.')[0]+'/',
        logger=logger,
        train=not args.test,
        sweep=args.sweep)

        # engine.evaluate()
        engine.test(test_mode)
        # engine.test('test')
    else:
        if test_mode == 'test':
            engine = Engine(
            config=config,
            save_dir=args.save_dir+'/'+config['model']['checkpoint_path'].split('_')[-1].split('.')[0]+'/',
            logger=logger,
            train=not args.test,
            sweep=args.sweep)
            engine.train_model()


if __name__ == "__main__":

    # run('Fudan')
    # run('tumor')
    # run('tumor_202512') # ./csv_files/new_data/test_tumor_2511_cycles.csv
    # run('healthy_202512') # ./csv_files/new_data/test_tumor_2511_cycles.csv
    # run('special_202512') # ./csv_files/new_data/test_tumor_2511_cycles.csv
    # run('CTRCD_ehj') #./csv_files/external/CTRCD_supp_cycles.csv
    # run('tumor_0901_ehj') #./csv_files/external/test_tumor_0901_cycles.csv
    # run('tumor_0822_ehj') #./csv_files/external/test_tumor_0822_quality_cycles.csv
    # run('wqh_0822_ehj') #./csv_files/external/test_wqh_0822_cycles.csv
    # run('CLBBB_0822_ehj') #./csv_files/external/test_CLBBB_quality_cycles.csv
    # run('tumor_0525_ehj') # ./csv_files/external/test_tumor_0525_cycles.csv
    # run('healthy_0525_ehj') # ./csv_files/external/test_normal_0525_cycles.csv
    # run('DMD_0603_ehj') # ./csv_files/external/test_DMD_0603_cycles.csv
    # run('tumor_0603_ehj') # ./csv_files/external/test_tumor_0603_cycle_quality.csv
    # run('BMD_0603_ehj') # ./csv_files/external/test_BMD_0603_quality_cycles.csv
    # run('zll_train')
    # run('zhaoli_train')

    run('val')
    run('test')
    run('test_KD')
    run('test_myo')
    run('test_DMD')
    run('test_train')
    
