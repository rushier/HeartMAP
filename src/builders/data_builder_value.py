import sys
from src.core.data import (
    EchoNetAp4Dataset,
    LVBiplaneEFDataset,
    AorticStenosisDataset,
    KineticsDataset,
    HuaxiLGEDataset
)
from torch.utils.data import DataLoader, ConcatDataset
from torch import distributed as dist
import os
import pandas as pd
import numpy as np

DATASETS = {
    "echonet": EchoNetAp4Dataset,
    "biplane": LVBiplaneEFDataset,
    "as": AorticStenosisDataset,
    "kinetics": KineticsDataset,
    'huaxi':HuaxiLGEDataset
}


def get_dataloaders(config, dataset_train, dataset_val, dataset_test, dataset_test_train, train=True):
    dataloaders = dict()

    if train:
        if config.mode == "as":
            dataloaders.update(
                {
                    "train": DataLoader(
                        dataset_train,
                        batch_size=config["batch_size"],
                        sampler=dataset_train.class_samplers(),
                        num_workers=min(8, os.cpu_count()),
                        pin_memory=True,
                        drop_last=True,
                    )
                }
            )
        else:
            dataloaders.update(
                {
                    "train": DataLoader(
                        dataset_train,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        num_workers=min(8, os.cpu_count()),
                        pin_memory=True,
                        drop_last=True,
                    )
                }
            )

    if config.mode != "pretrain":
        dataloaders.update(
            {
                "val": DataLoader(
                    dataset_val,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )
            }
        )

        dataloaders.update(
            {
                "test": DataLoader(
                    dataset_test,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )
            }
        )
        dataloaders.update(
            {
                "test_train": DataLoader(
                    dataset_test_train,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )
            }
        )

    return dataloaders

def split_csv_PID(df, column, seed, subset):
    df_complete = df.dropna(subset=subset)

    # 确保PID是字符串类型，以避免排序时出现错误
    df[column] = df[column].astype(str)
    # 根据PID分组，并获取所有唯一的PID列表
    unique_pids_all = df[column].unique()

    # 确保PID是字符串类型，以避免排序时出现错误
    df_complete[column] = df_complete[column].astype(str)
    # 根据PID分组，并获取所有唯一的PID列表
    unique_pids = df_complete[column].unique()

    # 将PID列表随机打乱
    np.random.seed(seed)
    np.random.shuffle(unique_pids)
    # 按照6：2：2的比例划分数据集
    train_size = int(0.8 * len(unique_pids_all))
    val_size = int(0.1 * len(unique_pids_all))
    test_size = len(unique_pids_all) - train_size - val_size

    test_pids = unique_pids[:test_size]
    val_pids = unique_pids[test_size:test_size + val_size]
    train_pids = [item for item in unique_pids_all if item not in test_pids]
    train_pids = [item for item in train_pids if item not in val_pids]

    # 根据PID将原始数据划分为训练集、验证集和测试集
    train_set = df[df[column].isin(train_pids)]
    val_set = df[df[column].isin(val_pids)]
    test_set = df[df[column].isin(test_pids)]
    return train_set, val_set, test_set


def build(config, train, transform, aug_transform, logger):
    dataset_name = config.name
    csv_file = pd.read_csv(config.dataset_path, encoding='gbk')
    csv_file = csv_file.dropna(subset=['LGE'])
    if not os.path.exists('./linux_code/csv_files/train_LGE_'+str(config.seed)+'_relabel.csv'):
        if config.chamber_list == 'mri':
            csv_file = csv_file.dropna(subset=['mri'])
            train_csv, val_csv, test_csv = split_csv_PID(csv_file, 'PID', config.seed, subset=['mri'])
        else:
            train_csv, val_csv, test_csv = split_csv_PID(csv_file, 'PID', config.seed, subset=['a2c', 'a3c', 'a4c', 'mid', 'mv', 'ecg'])
        train_csv.to_csv('./csv_files/train_LGE_'+str(config.seed)+'_relabel.csv')
        val_csv.to_csv('./csv_files/val_LGE_'+str(config.seed)+'_relabel.csv')
        test_csv.to_csv('./csv_files/test_LGE_'+str(config.seed)+'_relabel.csv')
    else:
        print('loaded stored dataset splits!')
        train_csv = pd.read_csv('./linux_code/csv_files/train_LGE_'+str(config.seed)+'_relabel.csv')
        val_csv = pd.read_csv('./linux_code/csv_files/val_LGE_'+str(config.seed)+'_relabel.csv')
        test_csv = pd.read_csv('./linux_code/csv_files/test_LGE_'+str(config.seed)+'_relabel.csv')
    if len(config.chamber_list.split('+')) == 1:
        print(len(train_csv), len(val_csv), len(test_csv))
        train_csv = train_csv.dropna(subset=[config.chamber_list])
        val_csv = val_csv.dropna(subset=[config.chamber_list])
        test_csv = test_csv.dropna(subset=[config.chamber_list])
        print('after dropna', len(train_csv), len(val_csv), len(test_csv))
    # print(val_csv.head())
    # if np.sum(val_csv['灌注异常'].values.tolist())/len(val_csv['灌注异常'].values.tolist()) < 0.2 or np.sum(test_csv['灌注异常'].values.tolist())/len(test_csv['灌注异常'].values.tolist()) < 0.2:
    #     return None
    train_csv = train_csv[train_csv['LGE1']!='not exists'] ##############################
    # train_csv = train_csv[train_csv['LGE']!=0] #################################
    val_csv = val_csv[val_csv['LGE1']!='not exists'] ##############################
    # val_csv = val_csv[val_csv['LGE']!=0] #################################
    test_csv = test_csv[test_csv['LGE1']!='not exists'] ##############################
    # test_csv = test_csv[test_csv['LGE']!=0] #################################
    dataset_train = (
        DATASETS[dataset_name](
            args=config,
            info_csv=train_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=aug_transform,
            split=config.split if config.mode == "pretrain" else "train",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
        )
        if train
        else None
    )

    if config.mode == "pretrain":
        dataset_val = None
        dataset_test = None
    else:
        dataset_val = DATASETS[dataset_name](
            args=config,
            info_csv=val_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="val" if train else "test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
        )
        dataset_test = DATASETS[dataset_name](
            args=config,
            info_csv=test_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
        )
        dataset_test_train = DATASETS[dataset_name](
            args=config,
            info_csv=train_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
        )

    dataloaders = get_dataloaders(config, dataset_train, dataset_val, dataset_test, dataset_test_train, train)

    if train:
        logger.info("Len of training dataset: {}".format(len(dataset_train)))
        logger.info(
            "Len of validation dataset: {}".format(
                len(dataset_val) if dataset_val is not None else 0
            )
        )

        print("Len of training dataset: {}".format(len(dataset_train)))
        print(
            "Len of validation dataset: {}".format(
                len(dataset_val) if dataset_val is not None else 0
            )
        )
    else:
        # logger.info("Len of test dataset: {}".format(len(dataset_val)))
        print("Len of test dataset: {}".format(len(dataset_val)))

    return (
        dataloaders,
        None
        if dataset_name in ["huaxi", "as", "prostate_single_patch", "kinetics"]
        else dataset_val.patient_data_dirs,
    )
