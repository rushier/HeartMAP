import sys
# sys.path.append('../gemtrans-main-ori-final-0407-ecgsupervised-value/')
from src.core.data import (
    EchoNetAp4Dataset,
    LVBiplaneEFDataset,
    AorticStenosisDataset,
    KineticsDataset,
    HuaxiLGEDataset,
    HuaxiLGEDataset_external_test,
    Fudan_test,
    tumor_test
)
from torch.utils.data import DataLoader, ConcatDataset
from torch import distributed as dist
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS = {
    "echonet": EchoNetAp4Dataset,
    "biplane": LVBiplaneEFDataset,
    "as": AorticStenosisDataset,
    "kinetics": KineticsDataset,
    'huaxi':HuaxiLGEDataset,
    'huaxi_external':HuaxiLGEDataset_external_test,
    'Fudan':Fudan_test,
    'tumor':tumor_test
}


def get_dataloaders(config, dataset_train, dataset_val, dataset_test, dataset_test_train, dataset_test_KD, dataset_test_DMD, dataset_test_myo, train=True):
    dataloaders = dict()
    # num worker 8 is ok
    if train:
        if config.mode == "as":
            dataloaders.update(
                {
                    "train": DataLoader(
                        dataset_train,
                        batch_size=config["batch_size"],
                        sampler=dataset_train.class_samplers(),
                        num_workers=min(4, os.cpu_count()),
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
                        num_workers=min(4, os.cpu_count()),
                        pin_memory=True,
                        drop_last=True,
                    )
                }
            )

    if config.mode != "pretrain":
        if dataset_val is not None:
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
        if dataset_test is not None:
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
        if dataset_test_train is not None:
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
        if dataset_test_KD is not None:
            dataloaders.update(
                {
                    "test_KD": DataLoader(
                        dataset_test_KD,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                    )
                }
            )
        if dataset_test_KD is not None:
            dataloaders.update(
                {
                    "test_DMD": DataLoader(
                        dataset_test_DMD,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                    )
                }
            )
        if dataset_test_KD is not None:    
            dataloaders.update(
                {
                    "test_myo": DataLoader(
                        dataset_test_myo,
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

def testing_set(csv_files=['./csv_files/exam_subset_three_cycles_0.csv',
'./csv_files/exam_subset_three_cycles_1.csv',
'./csv_files/exam_subset_three_cycles_2.csv',
'./csv_files/exam_subset_three_cycles_3.csv',
'./csv_files/exam_subset_three_cycles_4.csv']):
    all_test_df = []
    all_test_df_external = []
    for csv_file in csv_files[:3]:
        df = pd.read_csv(csv_file)
        print('数据集长度', len(df))
        all_test_df.append(df)
    for csv_file in csv_files[3:5]:
        df = pd.read_csv(csv_file)
        print('数据集长度', len(df))
        all_test_df_external.append(df)
    # all_test_df = pd.concat(all_test_df)
    # all_test_df = all_test_df[all_test_df['selected']==True]
    # all_test_df_external = pd.concat(all_test_df_external)
    # all_test_df_external = all_test_df_external[all_test_df_external['selected']==True]
    # print('xiba', len(all_test_df_external), len(all_test_df))
    all_test_df_external = pd.concat(all_test_df_external)
    return all_test_df, all_test_df_external

# def find_match_PID(PID_echodates, testing_csv_1, testing_csv_2):
#     new_testing_df = []
#     for PID,date in PID_echodates:
#         try:
#             PID = int(PID)
#         except:
#             PID = PID
#         match_PID = testing_csv_1[testing_csv_1['PID'] == PID]
#         match_date = match_PID[match_PID['Echodate'] == date]
#         if len(match_date) >0:
#             for i in range(len(match_date)):
#                 new_testing_df.append(match_date.iloc[i])
#         else:
#             match_PID = testing_csv_2[testing_csv_2['PID'] == PID]
#             match_date = match_PID[match_PID['Echodate'] == date]
#             if len(match_date) >0:
#                 for m in range(len(match_date)):
#                     new_testing_df.append(match_date.iloc[m])
#     new_testing_df = pd.DataFrame(new_testing_df)
#     return new_testing_df

def find_match_PID(PID_echodates, testing_csv_1, testing_csv_2):
    # 获取testing_csv_1的列结构
    target_columns = testing_csv_1.columns.tolist()
    # target_columns.append('LGE就是延迟强化（1=阳性，2=阴性）')
    
    new_testing_rows = []  # 存储行数据的列表
    
    for PID, date in PID_echodates:
        try:
            PID = int(PID)
        except:
            PID = PID
        date = int(date)
        
        # 先在testing_csv_1中查找
        match_PID = testing_csv_1[testing_csv_1['PID'] == PID]
        match_date = match_PID[match_PID['Echodate'] == date]
        
        if len(match_date) > 0:
            for i in range(len(match_date)):
                # 确保列顺序与testing_csv_1一致
                row_data = match_date.iloc[i][target_columns].to_dict()
                try:
                    row_data['LGE就是延迟强化（1=阳性，2=阴性）'] = row_data['LGE']
                except:
                    row_data['LGE就是延迟强化（1=阳性，2=阴性）'] = row_data['LGE就是延迟强化（1=阳性，2=阴性）']
                new_testing_rows.append(row_data)
        else:
            # 如果在testing_csv_1中没找到，在testing_csv_2中查找
            match_PID = testing_csv_2[testing_csv_2['PID'] == PID]
            match_date = match_PID[match_PID['Echodate'] == date]
            
            if len(match_date) > 0:
                for m in range(len(match_date)):
                    # 重新组织数据以匹配testing_csv_1的列结构
                    row_data = {}
                    for col in target_columns:
                        if col in match_date.columns:
                            row_data[col] = match_date.iloc[m][col]
                        else:
                            row_data[col] = None  # 对于testing_csv_2中缺少的列，填充None
                    if row_data['LGE就是延迟强化（1=阳性，2=阴性）'] is None:
                        print(match_date.iloc[m]['LGE就是延迟强化（1=阳性，2=阴性）'])
                    new_testing_rows.append(row_data)
    
    # 创建DataFrame，确保列顺序与testing_csv_1一致
    new_testing_df = pd.DataFrame(new_testing_rows, columns=target_columns)
    return new_testing_df

def return_pid_date(external_test, target=None):
    if target is not None:
        target_csv = external_test[external_test['source'].str.contains(target)]
        PIDs = target_csv['PID'].values.tolist()
        dates = target_csv['Echodate'].values.tolist()
        pid_date_list = sorted(list(set(zip(map(str, PIDs), map(str, dates)))))
    else:
        PIDs = external_test['PID'].values.tolist()
        dates = external_test['Echodate'].values.tolist()
        pid_date_list = sorted(list(set(zip(map(str, PIDs), map(str, dates)))))
    return pid_date_list

def filter_csv(csv_file):
    # 定义要检查的列
    columns_to_check = ['a4c', 'a3c', 'a2c', 'mid', 'mv']
    # 方法1: 使用 apply 和 all() 检查所有指定列是否都是空列表
    mask = ~csv_file[columns_to_check].apply(
        lambda row: all(len(eval(x)) == 0 for x in row), axis=1
    )
    filtered_myo_csv = csv_file[mask].copy()
    return filtered_myo_csv


def build(config, train, transform, aug_transform, logger, test=False):
    all_test_df, all_test_df_external = selected_cycles = testing_set()
    dataset_name = config.name
    csv_file = pd.read_csv(config.dataset_path, encoding='gbk')
    csv_file = csv_file.dropna(subset=['LGE'])
    train_ori = pd.read_csv('./csv_files/train_LGE_497_relabel_ori.csv')
    val_ori = pd.read_csv('./csv_files/val_LGE_497_relabel_ori.csv')
    test_ori = pd.read_csv('./csv_files/test_LGE_497_relabel_ori.csv')
    # 按照 train_ori val_ori test_ori 中的PID以及Echodate 从csv_file进行划分train_csv, val_csv, test_csv
    # Create unique identifiers for matching
    def create_id(df):
        return df['PID'].astype(str) + '_' + df['Echodate'].astype(str)
    
    csv_file['match_id'] = create_id(csv_file)
    train_ori['match_id'] = create_id(train_ori)
    val_ori['match_id'] = create_id(val_ori)
    test_ori['match_id'] = create_id(test_ori)
    
    # Split the data based on original splits
    train_csv = csv_file[csv_file['match_id'].isin(train_ori['match_id'])]
    val_csv = csv_file[csv_file['match_id'].isin(val_ori['match_id'])]
    test_csv = csv_file[csv_file['match_id'].isin(test_ori['match_id'])]
    
    # Verify no overlap between splits
    assert len(set(train_csv['match_id']) & set(val_csv['match_id'])) == 0
    assert len(set(train_csv['match_id']) & set(test_csv['match_id'])) == 0
    assert len(set(val_csv['match_id']) & set(test_csv['match_id'])) == 0
    if not os.path.exists('./csv_files/train_LGE_'+str(config.seed)+'_relabel.csv'):
        train_csv.to_csv('./csv_files/train_LGE_'+str(config.seed)+'_relabel.csv')
        val_csv.to_csv('./csv_files/val_LGE_'+str(config.seed)+'_relabel.csv')
        test_csv.to_csv('./csv_files/test_LGE_'+str(config.seed)+'_relabel.csv')
    if len(config.chamber_list.split('+')) == 1:
        
        train_csv = train_csv.dropna(subset=[config.chamber_list])
        val_csv = val_csv.dropna(subset=[config.chamber_list])
        test_csv = test_csv.dropna(subset=[config.chamber_list])
        print('after dropna', len(train_csv), len(val_csv), len(test_csv))
    # print(val_csv.head())
    # train_csv = train_csv[train_csv['LGE1']!='none'] ##############################
    # train_csv = train_csv[train_csv['LGE']!=0] #################################
    # val_ratio = np.sum(val_csv['灌注异常'].values.tolist())/len(val_csv['灌注异常'].values.tolist())
    # test_ratio = np.sum(test_csv['灌注异常'].values.tolist())/len(test_csv['灌注异常'].values.tolist())
    # if logger is not None:
    #     logger.info("val test ratios: {} {}".format(val_ratio, test_ratio))
    # if  val_ratio < 0.2 or test_ratio < 0.2:
    #     return None

    ### delete NaNs ##############################
    # train_csv = train_csv[train_csv['mri_old'] != './data_transfer/CMR/DMD_MRI_processed_psir_final/07419128_20220803_ren xuan hao/']
    # train_csv = train_csv[train_csv['mri_old'] != './data_transfer/CMR/DMD_MRI_processed_psir_final/09587733_20230503_xiao ai xin/']
    if config.lge_value and not test:
        train_csv = train_csv[train_csv['LGE1']!='not exists'] ##############################
        # train_csv = train_csv[train_csv['LGE']!=0] #################################
        val_csv = val_csv[val_csv['LGE1']!='not exists'] ##############################
        # val_csv = val_csv[val_csv['LGE']!=0] #################################
        test_csv = test_csv[test_csv['LGE1']!='not exists'] ##############################
        train_csv = train_csv.dropna(subset=['total_enhanced'])
        val_csv = val_csv.dropna(subset=['total_enhanced'])
        test_csv = test_csv.dropna(subset=['total_enhanced'])
    print(len(train_csv), len(val_csv), len(test_csv))

    # other tests
    all_external = pd.read_csv('./csv_files/external_tests_echo_mri_updated_indicators_第四批.csv', encoding='gbk')
    # external_test_1 = pd.read_csv('./csv_files/external_tests_echo_mri.csv', encoding='gbk')
    external_test_all_new = pd.read_excel('./csv_files/exam_subset_three_cycles_external_all.xlsx')
    #external_test_all_new = pd.read_csv('./csv_files/exam_subset_three_cycles_external_all.csv')
    all_external = all_external.dropna(subset=['LGE就是延迟强化（1=阳性，2=阴性）'])
    PID_echodates_KD = return_pid_date(all_external, 'KD')
    PID_echodates_DMD = return_pid_date(all_external, 'DMD')
    PID_echodates_myo = return_pid_date(all_external, '心肌')

    DMD_csv = find_match_PID(PID_echodates_DMD, all_test_df_external, external_test_all_new)
    KD_csv = find_match_PID(PID_echodates_KD, all_test_df_external, external_test_all_new)
    myo_csv = find_match_PID(PID_echodates_myo, all_test_df_external, external_test_all_new)
    DMD_csv = filter_csv(DMD_csv)
    KD_csv = filter_csv(KD_csv)
    myo_csv = filter_csv(myo_csv)
    DMD_csv.to_csv('./csv_files/DMD_input_test_threeCycle.csv')
    KD_csv.to_csv('./csv_files/KD_input_test_threeCycle.csv')
    myo_csv.to_csv('./csv_files/myo_input_test_threeCycle.csv')




    print(len(KD_csv),len(DMD_csv),len(myo_csv))
    print('external csv', len(return_pid_date(KD_csv, target=None)), len(return_pid_date(DMD_csv, target=None)), len(return_pid_date(myo_csv, target=None)))

    ## only_ecg
    if config.chamber_list == 'ecg':
        train_csv = train_csv[train_csv['ecg'] != 'none']
        all_test_df[0] = all_test_df[0][all_test_df[0]['ecg'] != 'none']
        all_test_df[1] = all_test_df[1][all_test_df[1]['ecg'] != 'none']
        all_test_df[2] = all_test_df[2][all_test_df[2]['ecg'] != 'none']
        KD_csv = KD_csv[KD_csv['ecg_path'] != 'none']
        DMD_csv = DMD_csv[DMD_csv['ecg_path'] != 'none']
        myo_csv = myo_csv[myo_csv['ecg_path'] != 'none']

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
            selected_cycles=None
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
            selected_cycles=all_test_df[1]
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
            selected_cycles=all_test_df[2]
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
            selected_cycles=all_test_df[0]
        )

        dataset_test_KD = DATASETS['huaxi_external'](
            args=config,
            info_csv=KD_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
            selected_cycles=KD_csv
        )

        dataset_test_DMD = DATASETS['huaxi_external'](
            args=config,
            info_csv=DMD_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
            selected_cycles=DMD_csv
        )

        dataset_test_myo = DATASETS['huaxi_external'](
            args=config,
            info_csv=myo_csv,
            dataset_path=config.dataset_path,
            mode=config.mode,
            max_frames=config.max_frames,
            transform=transform,
            aug_transform=None,
            split="test",
            use_seg_labels=config.use_seg_labels,
            max_clips=config.max_clips,
            selected_cycles=myo_csv
        )

    dataloaders = get_dataloaders(config, dataset_train, dataset_val, dataset_test, dataset_test_train, dataset_test_KD, dataset_test_DMD, dataset_test_myo, train)

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


def build_Fudan(config, train, transform, aug_transform, logger, test=False):
    dataset_name = config.name
    csv_file = pd.read_csv('./csv_files/Fudan-232-心超匹配MRI.csv', encoding='utf-8-sig')
    print('Fudan external', len(csv_file))

    ## only_ecg
    # dataset_train = (
    #     DATASETS[dataset_name](
    #         args=config,
    #         info_csv=train_csv,
    #         dataset_path=config.dataset_path,
    #         mode=config.mode,
    #         max_frames=config.max_frames,
    #         transform=transform,
    #         aug_transform=aug_transform,
    #         split=config.split if config.mode == "pretrain" else "train",
    #         use_seg_labels=config.use_seg_labels,
    #         max_clips=config.max_clips,
    #         selected_cycles=None
    #     )
    #     if train
    #     else None
    # )


    # dataset_val = DATASETS[dataset_name](
    #     args=config,
    #     info_csv=val_csv,
    #     dataset_path=config.dataset_path,
    #     mode=config.mode,
    #     max_frames=config.max_frames,
    #     transform=transform,
    #     aug_transform=None,
    #     split="val" if train else "test",
    #     use_seg_labels=config.use_seg_labels,
    #     max_clips=config.max_clips,
    #     selected_cycles=all_test_df[1]
    # )
    dataset_test = DATASETS['Fudan'](
        args=config,
        info_csv=csv_file,
        dataset_path=config.dataset_path,
        mode=config.mode,
        max_frames=config.max_frames,
        transform=transform,
        aug_transform=None,
        split="test",
        use_seg_labels=config.use_seg_labels,
        max_clips=config.max_clips,
        selected_cycles=csv_file
    )
    # dataset_test_train = DATASETS[dataset_name](
    #     args=config,
    #     info_csv=train_csv,
    #     dataset_path=config.dataset_path,
    #     mode=config.mode,
    #     max_frames=config.max_frames,
    #     transform=transform,
    #     aug_transform=None,
    #     split="test",
    #     use_seg_labels=config.use_seg_labels,
    #     max_clips=config.max_clips,
    #     selected_cycles=all_test_df[0]
    # )

    
    fudan_dataloader = DataLoader(
                        dataset_test,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)

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
        print("Len of test dataset: {}".format(len(dataset_test)))

    return (
        fudan_dataloader,
        None
        if dataset_name in ["huaxi", "as", "prostate_single_patch", "kinetics"]
        else dataset_val.patient_data_dirs,
    )

def one_dataset(csv_path, root_path, config, transform):
    try:
        csv_file = pd.read_csv(csv_path)
    except:
        csv_file = csv_path
    CTRCD_ehj = DATASETS['tumor'](
        args=config,
        info_csv=csv_file,
        dataset_path=config.dataset_path,
        mode=config.mode,
        max_frames=config.max_frames,
        transform=transform,
        aug_transform=None,
        split="test",
        use_seg_labels=config.use_seg_labels,
        max_clips=config.max_clips,
        selected_cycles=csv_file,
        root_path=root_path,
        transform_df = True
    )
    loader_CTRCD_ehj = DataLoader(
                    CTRCD_ehj,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False)
    return loader_CTRCD_ehj

def build_tumor(config, train, transform, aug_transform, logger, test=False):
    dataloaders = {}
    dataset_name = config.name
    csv_file = pd.read_csv('./csv_files/solid_tumor/solid_tumor_cycles_view_random_three.csv')
    print('tumor external', len(csv_file))

    dataset_test = DATASETS['tumor'](
        args=config,
        info_csv=csv_file,
        dataset_path=config.dataset_path,
        mode=config.mode,
        max_frames=config.max_frames,
        transform=transform,
        aug_transform=None,
        split="test",
        use_seg_labels=config.use_seg_labels,
        max_clips=config.max_clips,
        selected_cycles=csv_file,
        root_path='../data/qq_cycles/save_cycles/tumor_136/'
    )
    
    CTRCD_ehj = one_dataset('./csv_files/external/CTRCD_supp_cycles.csv',
    None, config, transform)
    dataloaders['CTRCD_ehj'] = CTRCD_ehj
    tumor_0901_ehj = one_dataset('./csv_files/external/test_tumor_0901_cycles.csv',
    None, config, transform)
    dataloaders['tumor_0901_ehj'] = tumor_0901_ehj
    tumor_0822_ehj = one_dataset('./csv_files/external/test_tumor_0822_quality_cycles.csv',
    None, config, transform)
    dataloaders['tumor_0822_ehj'] = tumor_0822_ehj
    wqh_0822_ehj = one_dataset('./csv_files/external/test_wqh_0822_cycles.csv',
    None, config, transform)
    dataloaders['wqh_0822_ehj'] = wqh_0822_ehj
    CLBBB_0822_ehj = one_dataset('./csv_files/external/test_CLBBB_quality_cycles.csv',
    None, config, transform)
    dataloaders['CLBBB_0822_ehj'] = CLBBB_0822_ehj
    tumor_0525_ehj = one_dataset('./csv_files/external/test_tumor_0525_cycles.csv',
    None, config, transform)
    dataloaders['tumor_0525_ehj'] = tumor_0525_ehj
    healthy_0525_ehj = one_dataset('./csv_files/external/test_normal_0525_cycles.csv',
    None, config, transform)
    dataloaders['healthy_0525_ehj'] = healthy_0525_ehj
    DMD_0603_ehj = one_dataset('./csv_files/external/test_DMD_0603_cycles.csv',
    None, config, transform)
    dataloaders['DMD_0603_ehj'] = DMD_0603_ehj
    tumor_0603_ehj = one_dataset('./csv_files/external/test_tumor_0603_cycle_quality_cycles.csv',
    None, config, transform)
    dataloaders['tumor_0603_ehj'] = tumor_0603_ehj
    BMD_0603_ehj = one_dataset('./csv_files/external/test_BMD_0603_quality_cycles.csv',
    None, config, transform)
    dataloaders['BMD_0603_ehj'] = BMD_0603_ehj
    tumor_202512 = one_dataset('./csv_files/new_data/test_tumor_2511_cycles.csv',
    None, config, transform)
    dataloaders['tumor_202512'] = tumor_202512

    healthy_202512 = one_dataset('./csv_files/new_data/test_healthy_2512_cycles.csv',
    None, config, transform)
    dataloaders['healthy_202512'] = healthy_202512

    CM_202512 = one_dataset('./csv_files/new_data/test_CM_2512_cycles.csv',
    None, config, transform)
    dataloaders['CM_202512'] = CM_202512

    special_202512 = one_dataset('./csv_files/new_data/test_special_2512_cycles.csv',
    None, config, transform)
    dataloaders['special_202512'] = special_202512


    df = pd.read_csv('../other_flow_estimation/flow_estimation_multi_l1/load/train_val_test_all.csv')
    target_paths = [
    './data/qq_raw_cycles/']
    # 提取满足条件的行
    filtered_df = df[df['root_path'].isin(target_paths)]
    zll_train = one_dataset(filtered_df, None, config, transform)
    dataloaders['zll_train'] = zll_train

    df = pd.read_csv('../other_flow_estimation/flow_estimation_multi_l1/load/train_val_test_all.csv')
    target_paths = [
    './data/qq_raw_cycles/']
    # 提取满足条件的行
    filtered_df = df[df['root_path'].isin(target_paths)]
    zhaoli_train = one_dataset(filtered_df, None, config, transform)
    dataloaders['zhaoli_train'] = zhaoli_train

    
    


    
    tumor_dataloader = DataLoader(
                        dataset_test,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False)
    dataloaders['tumor'] = tumor_dataloader
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
        print("Len of test dataset: {}".format(len(dataset_test)))

    return dataloaders




