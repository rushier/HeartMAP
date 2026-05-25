import torch
import cv2
import torchvision.transforms
import numpy as np
import SimpleITK as sitk
from torch.nn.utils.rnn import pad_sequence
import re
from operator import itemgetter
from math import isnan

import random
import math
import collections
import skimage.draw
import torch.nn.functional as F
import bz2
import _pickle as cPickle
import scipy.io as spio
import os
from os.path import join
from typing import Callable, Dict
import pandas as pd
import nibabel as nib

from scipy.io import loadmat
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import json
from glob import glob
from torch.nn import Upsample
import itertools
from torchvision.datasets import Kinetics

import numpy as np
import pandas as pd
from typing import Tuple, Any
import torch
import os
import joblib
from torch.utils.data import Dataset
import neurokit2 as nk
from scipy import signal

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import re

random.seed(0)
np.random.seed(0)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class EchoNetAp4Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        classification_classes = np.array([0, 30, 40, 55, 100])

        assert split in ["train", "val", "test"]

        # CSV file containing file names and labels
        file_list_df = pd.read_csv(os.path.join(dataset_path, "FileList.csv"))

        # Extract Split information
        splits = np.array(file_list_df["Split"].tolist())
        self.train_idx = np.where(splits == "TRAIN")[0]
        self.val_idx = np.where(splits == "VAL")[0]
        self.test_idx = np.where(splits == "TEST")[0]

        # Keep the correct slit
        if split == "train":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "TRAIN"]
        elif split == "val":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "VAL"]
        elif split == "test":
            file_list_df = file_list_df.loc[file_list_df["Split"] == "TEST"]

        # Get the list of video names and their ef values
        file_names = file_list_df["FileName"].tolist()
        labels = file_list_df["EF"].tolist()

        # Extract ES and ED frame indices
        es_frames = file_list_df["ESFrame"].tolist()
        self.es_frames = [
            None if isnan(es_frame) else int(es_frame) for es_frame in es_frames
        ]
        ed_frames = file_list_df["EDFrame"].tolist()
        self.ed_frames = [
            None if isnan(ed_frame) else int(ed_frame) for ed_frame in ed_frames
        ]

        # Extract LV segmentation masks
        if use_seg_labels:
            file_names, labels = self._extract_lv_trace(
                dataset_path, file_names, labels
            )

        # Full file paths
        self.patient_data_dirs = [
            os.path.join(dataset_path, "Videos", file_name + ".avi")
            for file_name in file_names
        ]

        # Categorize EF values
        self.classification_labels = (
            np.digitize(np.array(labels), classification_classes) - 1
        )
        self.classification_labels = torch.tensor(
            self.classification_labels, dtype=torch.long
        )

        # Bring EF values to [0,1]
        self.labels = list()
        for patient, _ in enumerate(self.patient_data_dirs):
            self.labels.append(labels[patient] / 100)

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Other attribues
        self.trans = transform
        self.aug_trans = aug_transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.max_frames = max_frames
        self.mode = mode
        self.train = split == "train"
        self.use_seg_labels = use_seg_labels
        self.max_clips = max_clips
        self.mean_std = mean_std

    def __getitem__(self, idx):

        # If the dataset is only created to find its mean and std
        if self.mean_std:
            ap4_cine_vid = self._loadvideo(self.patient_data_dirs[idx])
            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
            return ap4_cine_vid

        # extract cine vids
        ap4_cine_vid = self._loadvideo(self.patient_data_dirs[idx])
        orig_size = ap4_cine_vid.shape[0]

        # Mask indicating which frames are padding
        mask = torch.ones((1, self.max_frames), dtype=torch.bool)

        # Pad the video and extract clips
        ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
        (
            ap4_cine_vid,
            mask,
            lv_mask,
            ed_frame,
            ed_valid,
            es_frame,
            es_valid,
        ) = self._pad_vid(ap4_cine_vid, mask, idx, orig_size)

        # Perform augmentation during training
        if self.train and self.aug_trans is not None:
            ap4_cine_vid = self.aug_trans(ap4_cine_vid)

        # Interpolate the image
        if self.use_seg_labels:
            lv_mask = F.interpolate(
                lv_mask.unsqueeze(0).unsqueeze(1),
                size=(ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-1]),
            )
            lv_mask = lv_mask.squeeze(0)

        return {
            "vid": ap4_cine_vid.unsqueeze(0).unsqueeze(-1)
            if self.train
            else ap4_cine_vid.unsqueeze(1).unsqueeze(-1),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "mask": mask,
            "lv_mask": lv_mask,
            "ed_frame": torch.tensor(ed_frame),
            "ed_valid": torch.tensor(ed_valid),
            "es_frame": torch.tensor(es_frame),
            "es_valid": torch.tensor(es_valid),
            "class_label": self.classification_labels[idx],
        }

    def _extract_lv_trace(self, dataset_path, file_names, labels):
        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        with open(os.path.join(dataset_path, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))
        for filename in self.frames:
            for frame in self.frames[filename]:
                self.trace[filename][frame] = np.array(self.trace[filename][frame])

        keep = [len(self.frames[f + ".avi"]) >= 2 for f in file_names]
        file_names = [f for (f, k) in zip(file_names, keep) if k]
        labels = [f for (f, k) in zip(labels, keep) if k]
        self.ed_frames = [f for (f, k) in zip(self.ed_frames, keep) if k]
        self.es_frames = [f for (f, k) in zip(self.es_frames, keep) if k]

        return file_names, labels

    def _pad_vid(self, vid, mask, patient_idx, orig_size=None):

        file_name = os.path.basename(self.patient_data_dirs[patient_idx])

        # Combine the LV mask for ED and ES frames
        lv_mask_collated = torch.zeros(1)
        if self.use_seg_labels:
            for i in range(2):
                t = self.trace[file_name][self.frames[file_name][i]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(
                    np.rint(y).astype(np.int),
                    np.rint(x).astype(np.int),
                    (orig_size, orig_size),
                )
                lv_mask = np.zeros((orig_size, orig_size), np.bool)
                lv_mask[r, c] = 1
                lv_mask_collated = (
                    lv_mask if i == 0 else np.bitwise_or(lv_mask_collated, lv_mask)
                )
            lv_mask_collated = torch.from_numpy(lv_mask_collated.astype(np.float32))

        # If the number of frames is less than max frames, pad with 0's
        if vid.shape[0] <= self.max_frames:
            mask[0, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )

            ed_frame_idx, ed_valid, es_frame_idx, es_valid = self._frame_idx_in_clip(
                patient_idx, np.arange(self.max_frames)
            )

            if not self.train:
                mask = mask.unsqueeze(0)
                vid = vid.unsqueeze(0)
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)

                (
                    ed_frame_idx,
                    ed_valid,
                    es_frame_idx,
                    es_valid,
                ) = self._frame_idx_in_clip(
                    patient_idx, np.arange(starting_idx, starting_idx + self.max_frames)
                )

                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                # During validation and testing use all available clips
                ed_valid = []
                ed_frame_idx = []
                es_valid = []
                es_frame_idx = []

                num_clips = min(
                    math.ceil(vid.shape[0] / self.max_frames), self.max_clips
                )

                curated_clips = None
                for clip_idx in range(num_clips - 1):
                    curated_clips = (
                        vid[0 : self.max_frames].unsqueeze(0)
                        if curated_clips is None
                        else torch.cat(
                            (
                                curated_clips,
                                vid[
                                    self.max_frames
                                    * clip_idx : self.max_frames
                                    * (clip_idx + 1)
                                ].unsqueeze(0),
                            ),
                            dim=0,
                        )
                    )

                    (
                        clip_ed_idx,
                        clip_ed_valid,
                        clip_es_idx,
                        clip_es_valid,
                    ) = self._frame_idx_in_clip(
                        patient_idx,
                        np.arange(
                            self.max_frames * clip_idx, self.max_frames * (clip_idx + 1)
                        ),
                    )

                    ed_valid.append(clip_ed_valid)
                    ed_frame_idx.append(clip_ed_idx)
                    es_valid.append(clip_es_valid)
                    es_frame_idx.append(clip_es_idx)

                # The last clip is allowed to overlap with the previous one
                curated_clips = (
                    vid[-self.max_frames :].unsqueeze(0)
                    if curated_clips is None
                    else torch.cat(
                        (curated_clips, vid[-self.max_frames :].unsqueeze(0)), dim=0
                    )
                )

                (
                    clip_ed_idx,
                    clip_ed_valid,
                    clip_es_idx,
                    clip_es_valid,
                ) = self._frame_idx_in_clip(
                    patient_idx,
                    np.arange(vid.shape[0] - self.max_frames, vid.shape[0]),
                )

                ed_valid.append(clip_ed_valid)
                ed_frame_idx.append(clip_ed_idx)
                es_valid.append(clip_es_valid)
                es_frame_idx.append(clip_es_idx)

                vid = curated_clips
                mask = torch.cat([mask.unsqueeze(0)] * num_clips, dim=0)

        return (
            vid,
            mask,
            lv_mask_collated,
            ed_frame_idx,
            ed_valid,
            es_frame_idx,
            es_valid,
        )

    def _frame_idx_in_clip(self, data_idx, clip_idx):
        ed_frame, ed_valid, es_frame, es_valid = 0, False, 0, False

        if self.ed_frames[data_idx] in clip_idx:
            ed_frame = np.where(clip_idx == self.ed_frames[data_idx])[0].item()
            ed_valid = True

        if self.es_frames[data_idx] in clip_idx:
            es_frame = np.where(clip_idx == self.es_frames[data_idx])[0].item()
            es_valid = True

        return ed_frame, ed_valid, es_frame, es_valid

    def __len__(self):
        """
        Returns number of available samples

        :return: Number of graphs
        """
        return self.num_samples

    @staticmethod
    def _loadvideo(filename: str):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        :param filename: str, path to video to load
        :return: numpy array of dimension H*W*T
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError(
                    "Failed to load frame #{} of {}.".format(count, filename)
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v


class LVBiplaneEFDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "ef", "Currently only EF is supported for this dataset."

        classification_classes = np.array([0, 0.30, 0.40, 0.55, 1.0])

        # CSV file containing file names and labels
        report_df = pd.read_csv(os.path.join(dataset_path, "report.csv"))
        dicom_df = pd.read_csv(os.path.join(dataset_path, "dicom_info.csv"))

        # # Keep the correct slit
        if split == "train":
            report_df = report_df.loc[report_df["Split"] == "Train"]
        elif split == "val":
            report_df = report_df.loc[report_df["Split"] == "Val"]
        elif split == "test":
            report_df = report_df.loc[report_df["Split"] == "Test"]

        # Get patient IDs
        patient_ids = np.array(report_df["PatientID"].tolist())

        # Labels list
        self.labels = list()

        # Find path to available videos for each patient
        self.patient_data_dirs = dict()
        patient_num = 0
        for patient_id in patient_ids:
            file_paths = dicom_df.loc[
                dicom_df["PatientID_o"] == patient_id, "cleaned_filepath"
            ].tolist()

            # Have to change from pbz2 to mat since the dataset structure has changed since when the report csv was
            # created
            file_paths = [
                os.path.join(
                    dataset_path,
                    "batch_1",
                    *os.path.normpath(file_path).split("\\")[-3:],
                )[0:-4]
                + "mat"
                for file_path in file_paths
            ]

            all_files_exist = len(file_paths) > 1
            for file in file_paths:
                if not os.path.exists(file):
                    all_files_exist = False

            label_exists = (
                len(report_df.loc[report_df["PatientID"] == patient_id, "EF"].tolist())
                > 0
            )

            if all_files_exist and label_exists:
                self.patient_data_dirs[patient_num] = file_paths
                self.labels.append(
                    report_df.loc[report_df["PatientID"] == patient_id, "EF"].tolist()[
                        0
                    ]
                    / 100
                )
                patient_num += 1
            else:
                report_df = report_df.drop(
                    report_df[report_df["PatientID"] == patient_id].index
                )

        self.classification_labels = (
            np.digitize(np.array(self.labels), classification_classes) - 1
        )
        self.classification_labels = torch.tensor(
            self.classification_labels, dtype=torch.long
        )

        # Extract Split information
        splits = np.array(report_df["Split"].tolist())
        self.train_idx = np.where(splits == "Train")[0]
        self.val_idx = np.where(splits == "Val")[0]
        self.test_idx = np.where(splits == "Test")[0]

        # Extract the number of available data samples
        self.num_samples = len(self.patient_data_dirs)

        # Other attributes
        self.max_frames = max_frames
        self.mode = mode
        self.trans = transform
        self.train = split == "train"
        self.use_seg_labels = use_seg_labels
        self.max_clips = max_clips
        self.aug_trans = aug_transform
        self.mean_std = mean_std

    def __getitem__(self, idx):

        # Extract the AP4 Video and its labels
        data = self.loadmat(self.patient_data_dirs[idx][0])
        ap4_cine_vid = data["cropped"]

        if self.mean_std:
            return self.trans(np.array(ap4_cine_vid, dtype=np.uint8))

        ap4_ed_seg = None
        ap4_es_seg = None
        ap2_ed_seg = None
        ap2_es_seg = None

        ed_valid = [False, False]
        ed_frame_idx = [0, 0]
        es_valid = [False, False]
        es_frame_idx = [0, 0]
        try:
            for key in data["labels"]:
                if "LV_vol_d" in key:
                    ed_frame_idx[1] = data["labels"][key]["frame_num"] - 1
                    ed_valid[1] = True
                    ap4_ed_seg = data["labels"][key]["trace"]["mask_cropped"]
                elif "LV_vol_s" in key:
                    es_frame_idx[1] = data["labels"][key]["frame_num"] - 1
                    es_valid[1] = True
                    ap4_es_seg = data["labels"][key]["trace"]["mask_cropped"]
        except KeyError:
            print(
                "{} does not contain frame location labels. "
                "Setting all frame locations to 0.".format(
                    self.patient_data_dirs[idx][0]
                )
            )

        if ap4_ed_seg is None or ap4_es_seg is None:
            ap4_mask = None
        else:
            ap4_mask = torch.from_numpy(
                np.bitwise_or(ap4_ed_seg, ap4_es_seg).astype(np.float32)
            )

        # Extract the AP2 Video and its labels
        data = self.loadmat(self.patient_data_dirs[idx][1])
        ap2_cine_vid = data["cropped"]

        try:
            for key in data["labels"]:
                if "LV_vol_d" in key:
                    ed_frame_idx[0] = data["labels"][key]["frame_num"] - 1
                    ed_valid[0] = True
                    ap2_ed_seg = data["labels"][key]["trace"]["mask_cropped"]
                elif "LV_vol_s" in key:
                    es_frame_idx[0] = data["labels"][key]["frame_num"] - 1
                    es_valid[0] = True
                    ap2_es_seg = data["labels"][key]["trace"]["mask_cropped"]
        except KeyError:
            print(
                "{} does not contain frame location labels. "
                "Setting all frame locations to 0.".format(
                    self.patient_data_dirs[idx][1]
                )
            )

        if ap2_ed_seg is None or ap2_es_seg is None:
            ap2_mask = None
        else:
            ap2_mask = torch.from_numpy(
                np.bitwise_or(ap2_ed_seg, ap2_es_seg).astype(np.float32)
            )

        # During validation/test time, extract multiple clips per video
        num_clips = min(
            math.ceil(
                max(ap2_cine_vid.shape[-1], ap4_cine_vid.shape[-1]) / self.max_frames
            ),
            self.max_clips,
        )
        if not self.train:
            mask = torch.ones((num_clips, 2, self.max_frames), dtype=torch.bool)
            ap2_temp = list()
            ap4_temp = list()

            ed_valid = np.stack([ed_valid for _ in range(num_clips)])
            ed_frame_idx = np.stack([ed_frame_idx for _ in range(num_clips)])
            es_valid = np.stack([es_valid for _ in range(num_clips)])
            es_frame_idx = np.stack([es_frame_idx for _ in range(num_clips)])

            ap2_cine_vid = self.trans(np.array(ap2_cine_vid, dtype=np.uint8))
            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))

            for clip_idx in range(num_clips):
                (
                    ap2_cine_vid_temp,
                    mask[clip_idx],
                    ed_frame_idx[clip_idx][0],
                    ed_valid[clip_idx][0],
                    es_frame_idx[clip_idx][0],
                    es_valid[clip_idx][0],
                ) = self.pad_vid(
                    ap2_cine_vid,
                    mask[clip_idx],
                    0,
                    ed_frame_idx[clip_idx][0],
                    ed_valid[clip_idx][0],
                    es_frame_idx[clip_idx][0],
                    es_valid[clip_idx][0],
                    clip_idx,
                )
                ap2_temp.append(ap2_cine_vid_temp.unsqueeze(0))

                (
                    ap4_cine_vid_temp,
                    mask[clip_idx],
                    ed_frame_idx[clip_idx][1],
                    ed_valid[clip_idx][1],
                    es_frame_idx[clip_idx][1],
                    es_valid[clip_idx][1],
                ) = self.pad_vid(
                    ap4_cine_vid,
                    mask[clip_idx],
                    1,
                    ed_frame_idx[clip_idx][1],
                    ed_valid[clip_idx][1],
                    es_frame_idx[clip_idx][1],
                    es_valid[clip_idx][1],
                    clip_idx,
                )
                ap4_temp.append(ap4_cine_vid_temp.unsqueeze(0))

            ap2_cine_vid = torch.cat(ap2_temp, dim=0)
            ap4_cine_vid = torch.cat(ap4_temp, dim=0)
            output = torch.cat(
                (
                    ap2_cine_vid.unsqueeze(1).unsqueeze(-1),
                    ap4_cine_vid.unsqueeze(1).unsqueeze(-1),
                ),
                dim=1,
            )
        else:
            mask = torch.ones((2, self.max_frames), dtype=torch.bool)
            ap2_cine_vid = self.trans(np.array(ap2_cine_vid, dtype=np.uint8))
            (
                ap2_cine_vid,
                mask,
                ed_frame_idx[0],
                ed_valid[0],
                es_frame_idx[0],
                es_valid[0],
            ) = self.pad_vid(
                ap2_cine_vid,
                mask,
                0,
                ed_frame_idx[0],
                ed_valid[0],
                es_frame_idx[0],
                es_valid[0],
            )

            ap4_cine_vid = self.trans(np.array(ap4_cine_vid, dtype=np.uint8))
            (
                ap4_cine_vid,
                mask,
                ed_frame_idx[1],
                ed_valid[1],
                es_frame_idx[1],
                es_valid[1],
            ) = self.pad_vid(
                ap4_cine_vid,
                mask,
                1,
                ed_frame_idx[1],
                ed_valid[1],
                es_frame_idx[1],
                es_valid[1],
            )

            output = torch.cat(
                (
                    ap2_cine_vid.unsqueeze(0).unsqueeze(-1),
                    ap4_cine_vid.unsqueeze(0).unsqueeze(-1),
                ),
                dim=0,
            )

            output = self.aug_trans(output)

        # Prepare the LV mask if needed
        if self.use_seg_labels:
            if not (ap2_mask is None or ap4_mask is None):
                ap2_lv_mask = F.interpolate(
                    ap2_mask.unsqueeze(0).unsqueeze(1),
                    size=(ap2_cine_vid.shape[-1], ap2_cine_vid.shape[-2]),
                ).squeeze(0)
                ap4_lv_mask = F.interpolate(
                    ap4_mask.unsqueeze(0).unsqueeze(1),
                    size=(ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2]),
                ).squeeze(0)
                lv_mask = torch.cat((ap2_lv_mask, ap4_lv_mask), dim=0)

                if not self.train:
                    lv_mask = torch.cat(
                        [lv_mask.unsqueeze(0) for _ in range(num_clips)], dim=0
                    )
            else:
                if self.train:
                    lv_mask = torch.ones(
                        (2, ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2])
                    )
                else:
                    lv_mask = torch.ones(
                        (num_clips, 2, ap4_cine_vid.shape[-1], ap4_cine_vid.shape[-2])
                    )
        else:
            lv_mask = torch.zeros(1)

        ed_frame_idx = torch.tensor(ed_frame_idx, dtype=torch.long)
        ed_valid = torch.tensor(ed_valid, dtype=torch.bool)
        es_frame_idx = torch.tensor(es_frame_idx, dtype=torch.long)
        es_valid = torch.tensor(es_valid, dtype=torch.bool)

        return {
            "vid": output,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "mask": mask,
            "lv_mask": lv_mask,
            "ed_frame": ed_frame_idx,
            "ed_valid": ed_valid,
            "es_frame": es_frame_idx,
            "es_valid": es_valid,
            "class_label": self.classification_labels[idx],
        }

    def __len__(self):
        """
        Returns number of available samples

        :return: Number of graphs
        """

        return self.num_samples

    @staticmethod
    def decompress_pickle(file):
        """
        Decomporesses PBZ2 files
        Code from https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e

        :param file: str, path to file
        :return: decomporessed video
        """
        data = bz2.BZ2File(file, "rb")
        data = cPickle.load(data)
        return data

    # The code below is directly copied from
    # https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    def loadmat(self, filename):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        """
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict

    def pad_vid(
        self, vid, mask, mask_idx, ed_frame, ed_valid, es_frame, es_valid, clip_idx=0
    ):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )

            ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                np.arange(self.max_frames), ed_frame, ed_valid, es_frame, es_valid
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]

                ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                    np.arange(starting_idx, starting_idx + self.max_frames),
                    ed_frame,
                    ed_valid,
                    es_frame,
                    es_valid,
                )
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]

                    ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                        np.arange(
                            self.max_frames * clip_idx, self.max_frames * (clip_idx + 1)
                        ),
                        ed_frame,
                        ed_valid,
                        es_frame,
                        es_valid,
                    )
                else:
                    vid = vid[-self.max_frames :]

                    ed_frame, ed_valid, es_frame, es_valid = self._frame_idx_in_clip(
                        np.arange(vid.shape[0] - self.max_frames, vid.shape[0]),
                        ed_frame,
                        ed_valid,
                        es_frame,
                        es_valid,
                    )

        return vid, mask, ed_frame, ed_valid, es_frame, es_valid

    def _frame_idx_in_clip(self, clip_idx, ed_frame, ed_valid, es_frame, es_valid):

        if ed_frame in clip_idx and ed_valid:
            ed_frame = np.where(clip_idx == ed_frame)[0].item()
            ed_valid = True
        else:
            ed_valid = False

        if es_frame in clip_idx and es_valid:
            es_frame = np.where(clip_idx == es_frame)[0].item()
            es_valid = True
        else:
            es_valid = False

        return ed_frame, ed_valid, es_frame, es_valid


label_schemes = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}


def bicuspid_filter(df: pd.DataFrame):
    return df[~df["Bicuspid"]]


filtering_functions: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "bicuspid": bicuspid_filter
}


class AorticStenosisDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "~/as",
        split: str = "train",
        mode: str = "as",
        max_frames: int = 16,
        transform=None,
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "as", "Only AS mode is supported"

        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])

        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(dataset_path, "annotations-all.csv"))
        # append dataset root to each path in the dataframe
        dataset["path"] = dataset["path"].map(lambda x: join(dataset_path, x))

        dataset = dataset[dataset["as_label"].map(lambda x: x in label_schemes.keys())]

        # Take train/test/val
        dataset = dataset[dataset.split == split]
        # Apply an arbitrary filter
        # filtering_function = filtering_functions["bicuspid"]
        # dataset = filtering_function(dataset)

        self.patient_studies = list()
        for patient_id in list(set(dataset["patient_id"])):
            for study_date in list(
                set(dataset[dataset["patient_id"] == patient_id]["date"])
            ):
                self.patient_studies.append((patient_id, study_date))

        self.dataset = dataset
        self.max_frames = max_frames
        self.train = split == "train"
        self.trans = transform
        self.aug_trans = aug_transform
        self.max_clips = max_clips
        self.mean_std = mean_std

    def class_samplers(self):
        labels_AS = list()
        for pid in self.patient_studies:
            patient_id, study_date = pid
            data_info = self.dataset[self.dataset["patient_id"] == patient_id]
            data_info = data_info[data_info["date"] == study_date]

            labels_AS.append(label_schemes[data_info["as_label"].iloc[0]])

        class_sample_count_AS = np.array(
            [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
        )
        weight_AS = 1.0 / class_sample_count_AS

        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS, 0, 0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()

        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))

        return sampler_AS

    def __len__(self) -> int:
        return len(self.patient_studies)

    def __getitem__(self, item):
        patient_id, study_date = self.patient_studies[item]
        data_info = self.dataset[self.dataset["patient_id"] == patient_id]
        data_info = data_info[data_info["date"] == study_date]
        # label = torch.tensor(self.labelling_scheme[data_info[self.label_key]])

        available_views = list(data_info["view"])
        num_plax = available_views.count("plax")
        num_psax = available_views.count("psax")

        frame_nums = list()

        if self.mean_std:
            return self.trans(
                np.moveaxis(loadmat(data_info["path"].iloc[0])["cine"], 0, -1)
            )

        all_plax_cine = list()
        if num_plax > 0:
            plax_indices = (
                [np.random.randint(num_plax)] if self.train else list(range(num_plax))
            )

            for plax_idx in plax_indices:
                plax_data_info = data_info[data_info["view"] == "plax"].iloc[plax_idx]

                # Transform and augment PLAX vid
                plax_cine = self.trans(
                    np.moveaxis(loadmat(plax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    plax_cine = plax_cine.unsqueeze(0)
                    plax_cine = self.aug_trans(plax_cine)
                    plax_cine = plax_cine.squeeze(0)

                all_plax_cine.append(plax_cine)
                frame_nums.append(plax_cine.shape[0])

        all_psax_cine = list()
        if num_psax > 0:
            psax_indices = (
                [np.random.randint(num_psax)] if self.train else list(range(num_psax))
            )

            for psax_idx in psax_indices:
                psax_data_info = data_info[data_info["view"] == "psax"].iloc[psax_idx]

                # Transform and augment psax vid
                psax_cine = self.trans(
                    np.moveaxis(loadmat(psax_data_info["path"])["cine"], 0, -1)
                )
                if self.aug_trans is not None:
                    psax_cine = psax_cine.unsqueeze(0)
                    psax_cine = self.aug_trans(psax_cine)
                    psax_cine = psax_cine.squeeze(0)

                all_psax_cine.append(psax_cine)
                frame_nums.append(psax_cine.shape[0])

        no_plax = False
        no_psax = False
        if num_plax == 0:
            all_plax_cine.append(torch.zeros_like(all_psax_cine[0]))
            num_plax = 1
            no_plax = True
        elif num_psax == 0:
            all_psax_cine.append(torch.zeros_like(all_plax_cine[0]))
            num_psax = 1
            no_psax = True

        if not self.train:

            num_clips = min(
                math.ceil(max(frame_nums) / self.max_frames),
                self.max_clips,
            )

            plax_psax_comb = list(
                itertools.product(list(range(num_plax)), list(range(num_psax)))
            )

            if len(plax_psax_comb) > 6:
                plax_psax_comb = plax_psax_comb[:6]

            mask = torch.ones(
                (
                    num_clips * len(plax_psax_comb),
                    2,
                    self.max_frames,
                ),
                dtype=torch.bool,
            )

            plax_temp = list()
            psax_temp = list()

            for combination_idx in range(len(plax_psax_comb)):
                for clip_idx in range(num_clips):
                    (
                        plax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_plax_cine[plax_psax_comb[combination_idx][0]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        0,
                        clip_idx,
                    )
                    plax_temp.append(plax_cine_temp.unsqueeze(0))

                    (
                        psax_cine_temp,
                        mask[(num_clips * combination_idx) + clip_idx],
                    ) = self.pad_vid(
                        all_psax_cine[plax_psax_comb[combination_idx][1]],
                        mask[(num_clips * combination_idx) + clip_idx],
                        1,
                        clip_idx,
                    )
                    psax_temp.append(psax_cine_temp.unsqueeze(0))

            plax_cine = torch.cat(plax_temp, dim=0)
            psax_cine = torch.cat(psax_temp, dim=0)
            cine = torch.cat(
                (
                    plax_cine.unsqueeze(1).unsqueeze(-1),
                    psax_cine.unsqueeze(1).unsqueeze(-1),
                ),
                dim=1,
            )

            if no_plax:
                mask[:, 0, :] = False
            elif no_psax:
                mask[:, 1, :] = False
        else:
            mask = torch.ones((2, self.max_frames), dtype=torch.bool)

            plax_cine, mask = self.pad_vid(all_plax_cine[0], mask, 0)
            psax_cine, mask = self.pad_vid(all_psax_cine[0], mask, 1)

            cine = torch.cat(
                (
                    plax_cine.unsqueeze(0).unsqueeze(-1),
                    psax_cine.unsqueeze(0).unsqueeze(-1),
                ),
                dim=0,
            )

            if no_plax:
                mask[0, :] = False
            elif no_psax:
                mask[1, :] = False

        label = label_schemes[data_info["as_label"].iloc[0]]

        return {
            "vid": cine,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "lv_mask": torch.zeros(1),
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
        }

    def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]
                else:
                    vid = vid[-self.max_frames :]

        return vid, mask


class KineticsDataset(Kinetics):
    def __init__(
        self,
        dataset_path,
        mode,
        max_frames=32,
        transform=None,
        split="train",
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
        mean=0.413165,
        std=0.278993,
    ):

        super().__init__(
            root=dataset_path,
            frames_per_clip=max_frames,
            num_classes="400",
            split=split,
            transform=transform,
            step_between_clips=max_frames,
            num_workers=16,
            download=False,
        )

        self.max_frames = max_frames
        self.mean_std = mean_std
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        vid, _, label = super().__getitem__(idx)

        # Move color channel to last
        vid = vid.permute(0, 2, 3, 1).to(torch.float32) / 255

        if self.mean_std:
            return vid

        vid = (vid - self.mean) / self.std

        return {
            "vid": vid.unsqueeze(0),
            "label": label,
            "mask": torch.ones((1, self.max_frames), dtype=torch.bool),
            "lv_mask": torch.zeros(1),
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
        }

def gammacorrection(src, gamma, vendor_dict):
    if np.random.randint(2):
        if np.random.randint(2):
            table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(0, 256)]).astype("uint8")
            src = cv2.LUT(src, table)
        elif vendor_dict is not None:
            file = random.choice(vendor_dict)
            target = np.load(file)['array'][np.newaxis]
            src = src[np.newaxis]
            src = hist_match(src, target)[0]
    return src

from  albumentations  import (ShiftScaleRotate, RandomRotate90,
    Transpose, ShiftScaleRotate,
    GaussNoise, MotionBlur,PixelDropout, Flip, Compose
)
def strong_aug():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=0),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=25, border_mode=0, p=0.5),
        #RandomContrast(0.3,p=0),
        PixelDropout(dropout_prob=0.005, p=0)])
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1)])

def weak_aug():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=0),
        ShiftScaleRotate(shift_limit=0.001, scale_limit=0.01, rotate_limit=25, border_mode=0, p=0.5),
        #RandomContrast(0.3,p=0),
        PixelDropout(dropout_prob=0.005, p=0)])
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1)])
    
def noise_aug():
    return Compose([
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=1),
        #RandomContrast(0.3,p=0)
        ])

# imgaug = strong_aug()
# class HuaxiLGEDataset(Dataset):
#     def __init__(
#         self,
#         args,
#         info_csv,
#         dataset_path: str = "~/as",
#         split: str = "train",
#         mode: str = "as",
#         max_frames: int = 16,
#         transform=None,
#         use_seg_labels=False,
#         aug_transform=None,
#         max_clips=1,
#         mean_std=False,
#     ):

#         super().__init__()

#         assert mode == "lge", "Only lge mode is supported"
#         self.args = args
#         # navigation for linux environment
#         # dataset_root = dataset_root.replace('~', os.environ['HOME'])

#         # read in the data directory CSV as a pandas dataframe
#         dataset = info_csv
#         # Take train/test/val
#         # Apply an arbitrary filter
#         # filtering_function = filtering_functions["bicuspid"]
#         # dataset = filtering_function(dataset)

#         self.dataset = dataset
#         self.length = self.args.length
#         self.size = self.args.size
#         self.max_frames = max_frames
#         self.train = split == "train"
#         self.trans = transform
#         self.aug_trans = aug_transform
#         self.max_clips = max_clips
#         self.mean_std = mean_std
#         self.apical_chamber = ['a4c', 'a3c', 'a2c']
#         self.short_chamber = ['mid', 'mv']
#         self.apical_seg_path='../data/flow_Inference_seg_save/SpaceTimeUnet_Echo3d_True_112_a4c_a2c_a3c_prior_True_pretrain_True_240704/' 
#         self.short_seg_path='../data/flow_Inference_seg_save/SpaceTimeUnet_Huaxi3d_False_112_mid_mv_prior_True_pretrain_True_240716/'
#         self.transform = None
#     def class_samplers(self):
#         labels_AS = self.dataset['LGE'].values.tolist()
#         class_sample_count_AS = np.array(
#             [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
#         )
#         weight_AS = 1.0 / class_sample_count_AS
#         samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
#         samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
#         sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
#         return sampler_AS

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def __getitem__(self, item):
#         row = self.dataset.iloc[item]
#         root_path = row['root_path']
#         label = row['LGE']
#         mid_path = row['mid']
#         mid_array, mid_mask = self.read_chamber(mid_path, 'mid', root_path)
#         mid_array = skimage.transform.resize(mid_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
#         mid_mask = skimage.transform.resize(mid_mask, (self.length, self.args.frame_size, self.args.frame_size), 0)


#         a4c_path = row['a4c']
#         a4c_array, a4c_mask = self.read_chamber(a4c_path, 'a4c', root_path)
#         a4c_array = skimage.transform.resize(a4c_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
#         a4c_mask = skimage.transform.resize(a4c_mask, (self.length, self.args.frame_size, self.args.frame_size), 0)

#         if self.args.add_mri:
#             mri_path = row['mri_path']
#             mri_array, mri_mask = self.read_mri(mri_path)
#             array = np.concatenate([mid_array[np.newaxis], a4c_array[np.newaxis], mri_array[np.newaxis]], axis=0)
#             mask = np.concatenate([mid_mask[np.newaxis], a4c_mask[np.newaxis], mri_mask[np.newaxis]], axis=0)
#         else:
#             array = np.concatenate([mid_array[np.newaxis], a4c_array[np.newaxis], mri_array[np.newaxis]], axis=0)
#             mask = np.concatenate([mid_mask[np.newaxis], a4c_mask[np.newaxis], mri_mask[np.newaxis]], axis=0)
#         array = array[:,:,:,:,np.newaxis]
#         np.save('array.npy', array)
#         cine = torch.tensor(array).type(torch.FloatTensor)
#         mask = torch.tensor(mask).type(torch.FloatTensor)
#         lv_mask = mask==2
        


#         return {
#             "vid": cine,
#             "label": torch.tensor(label).type(torch.FloatTensor),
#             "mask": torch.ones((3, self.length), dtype=torch.bool),
#             "lv_mask": lv_mask,
#             "ed_frame": 0,
#             "ed_valid": False,
#             "es_frame": 0,
#             "es_valid": False,
#             "class_label": torch.zeros(1),
#         }

#     def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
#         if vid.shape[0] < self.max_frames:
#             mask[mask_idx, vid.shape[0] :] = False
#             vid = torch.cat(
#                 (
#                     vid,
#                     torch.zeros(
#                         self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
#                     ),
#                 ),
#                 dim=0,
#             )
#         else:
#             if self.train:
#                 starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
#                 vid = vid[starting_idx : starting_idx + self.max_frames]
#             else:
#                 if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
#                     vid = vid[
#                         self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
#                     ]
#                 else:
#                     vid = vid[-self.max_frames :]

#         return vid, mask

#     def window(self, array, window_cen, window_wid):
#         # if len(window_cen) != array.shape[0] or len(window_wid) != array.shape[0]:
#         #     raise ValueError("Length of window center and width lists must match the number of slices.")
#         if not isinstance(window_wid, list):
#             window_wid = [window_wid] * len(window_cen)
#         window_cen = [float(c) for c in window_cen]
#         window_wid = [float(w) for w in window_wid]
#         window_cen_array = np.array(window_cen)
#         window_wid_array = np.array(window_wid)

#         window_cen_broadcast = window_cen_array[:, np.newaxis, np.newaxis]  
#         window_wid_broadcast = window_wid_array[:, np.newaxis, np.newaxis]
        
#         window_min = window_cen_broadcast - window_wid_broadcast / 2.0
#         window_max = window_cen_broadcast + window_wid_broadcast / 2.0
        
#         windowed_array = np.clip(array, window_min, window_max)
#         windowed_array = (windowed_array - window_min) / (window_max - window_min)
#         return windowed_array

    
#     def load_json(self, json_file):
#         with open(json_file, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#         return data

#     def load_mri(self, file_path):
#         data = nib.load(file_path)
#         array = np.array(data.dataobj)
#         orientation = nib.aff2axcodes(data.affine)
#         if orientation == ('P', 'I', 'R'):
#             array = array.transpose(1, 0, 2)
#             orientation = ('L', 'P', 'S')
#         return array
        
#     def read_mri(self, mri_path):
#         if isinstance(mri_path, str):
#             mri_files = glob(mri_path+'/*.nii')
#             mri_file = random.choice(mri_files)
#             json_file = mri_file.replace('.nii', '.json')
#             array_mri = self.load_mri(mri_file)
#             json_mri = self.load_json(json_file)
#             window_cen = json_mri['window_cen']
#             window_wid = json_mri['window_wid']
#             array_mri = array_mri.transpose(2,0,1)
#             array_mri = self.window(array_mri, window_cen, window_wid)
#             array_mri = skimage.transform.resize(array_mri, (self.length, self.args.frame_size, self.args.frame_size), 1)
#             mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
#         else:
#             array_mri = np.zeros((self.length, self.args.frame_size, self.args.frame_size))
#             mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
#         return array_mri, mask_mri

#     def read_chamber(self, ch_path, chamber, root_path):
#         if not isinstance(ch_path, str) :
#             have_zero = True
#             image_array = np.zeros((self.length,self.size, self.size))
#             mask = np.zeros((self.length,self.size, self.size))
#         else:
#             if chamber in self.apical_chamber:
#                 seg_path = self.apical_seg_path
#             if chamber in self.short_chamber:
#                 seg_path = self.short_seg_path
#             #print('_'.join(ch_path.split('/')), seg_path + '/*/' + '_'.join(ch_path.split('/'))+'_*_*.nii.gz')
#             seg_files = glob(seg_path + '/*/*/' + '_'.join(ch_path.split('/'))+'_*_*.nii.gz')
#             seg_files = [seg_file for seg_file in seg_files if int(seg_file.split('_')[-2])<4]
#             seg_file = random.choice(seg_files)
            
#             save_processed_file = seg_file.replace(seg_path, '../data/save_processed/EchoLGE_multi_task/')
#             save_processed_file = save_processed_file.replace('.nii.gz', '.npz')
#             if not os.path.exists(save_processed_file):
#                 if not os.path.exists('/'.join(save_processed_file.split('/')[:-1])):
#                     os.makedirs('/'.join(save_processed_file.split('/')[:-1]))
#                 mask_dcm = sitk.ReadImage(seg_file)
#                 mask = sitk.GetArrayFromImage(mask_dcm)
#                 cycle_num = seg_file.split('_')[-2]
#                 #print(seg_files,cycle_num)

#                 image_paths = glob(root_path + '/' + ch_path + '_' + str(cycle_num) + '_*' + '.npz')
#                 image_path = random.choice(image_paths)
#                 image_array = np.load(image_path)['array']
#                 image_array = skimage.transform.resize(image_array, (self.length, self.size, self.size), 1)
#                 mask = skimage.transform.resize(mask, (self.length, self.size, self.size), 0)
#                 np.savez(save_processed_file, processed_img=image_array, processed_mask=mask)
#             else:
#                 array = np.load(save_processed_file)
#                 image_array = array['processed_img']
#                 mask = array['processed_mask']
#                 # if image_array != [16,128,128]:
#                 #     image_array = skimage.transform.resize(image_array, (self.length, self.size, self.size), 1)

#             # if self.transform and self.train:
#             #     image_array = image_array.transpose(1,2,0)
#             #     mask = mask.transpose(1,2,0)
#             #     augs = self.transform(image=image_array, mask=mask)
#             #     image_array, mask = augs['image'].transpose(2,0,1), augs['mask'].transpose(2,0,1)

#             # if self.train:
#             #     gamma = random.uniform(0.1, 2.0)
#             #     image_array = gammacorrection((image_array*255).astype('uint8'), gamma, None)
#             image_array = (image_array - np.amin(image_array)) / (np.amax(image_array)-np.amin(image_array))
#         return image_array, mask

imgaug = strong_aug()
mriaug = weak_aug()
SAMPLES_IN_5_SECONDS_AT_500HZ = 2500
SAMPLES_IN_10_SECONDS_AT_500HZ = 5000

def modify_pid(pid_list, date=False):
    pids_new = []
    for pid in pid_list:
        try:
            pid = str(int(pid))
        except:
            pid = pid
        if '.' in pid and not date:
            pid = pid.split('.')[0]
        pids_new.append(pid)
    return pids_new

def process_date(dates, seperate='/'):
    new_dates = []
    for date in dates:
        if '.' in date:
            seperate = '.'
        if '/' in date:
            seperate = '/'
        if '-' in date:
            seperate = '-'
        
        if len(date.split(seperate)) > 2:
            y = str(date.split(seperate)[0])
            m = '0'*(2-len(str(date.split(seperate)[1]))) + str(date.split(seperate)[1])
            d = '0'*(2-len(str(date.split(seperate)[2]))) + str(date.split(seperate)[2])
            new_dates.append(y+m+d)
        else:
            new_dates.append(date)
    return new_dates

def csv2str(csv_file, columns_list):
    for column in columns_list:
        values = csv_file[column].values.tolist()
        values = [str(v) for v in values]
        csv_file[column] = values
    return csv_file


class HuaxiLGEDataset(Dataset):
    def __init__(
        self,
        args,
        info_csv,
        dataset_path: str = "~/as",
        split: str = "train",
        mode: str = "as",
        max_frames: int = 16,
        transform=None,
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "lge", "Only lge mode is supported"
        self.args = args
        self.beat_based_attention_mask = False
        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])

        # read in the data directory CSV as a pandas dataframe
        dataset = info_csv
        # Take train/test/val
        # Apply an arbitrary filter
        # filtering_function = filtering_functions["bicuspid"]
        # dataset = filtering_function(dataset)

        self.dataset = dataset
        self.dataset = csv2str(self.dataset, ['PID', 'Echodate'])
        self.length = self.args.length
        self.size = self.args.size
        self.max_frames = max_frames
        self.train = split == "train"
        self.trans = transform
        self.aug_trans = aug_transform
        self.max_clips = max_clips
        self.mean_std = mean_std
        self.chamber_list = self.args.chamber_list.split('+')
        self.apical_chamber = ['a4c', 'a3c', 'a2c']
        self.short_chamber = ['mid', 'mv']
        self.apical_seg_path='../data/flow_Inference_seg_save/SpaceTimeUnet_Echo3d_True_112_a4c_a2c_a3c_prior_True_pretrain_True_240704/' 
        self.short_seg_path='../data/flow_Inference_seg_save/SpaceTimeUnet_Huaxi3d_False_112_mid_mv_prior_True_pretrain_True_240716/'
        self.flow_path='../data/flow/DMD_MF/' 
        self.all_chamber_info = pd.read_csv('./LGE/csv_files/input_all_chamber_videos.csv')
        self.all_chamber_info = csv2str(self.all_chamber_info, ['PID', 'Echodate'])
        self.transform = imgaug
        self.mri_transform = mriaug

        self.encode = False
        self.pretrain = False
    def class_samplers(self):
        labels_AS = self.dataset['LGE'].values.tolist()
        class_sample_count_AS = np.array(
            [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
        )
        weight_AS = 1.0 / class_sample_count_AS
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        return sampler_AS

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        row = self.dataset.iloc[item]
        PID = row['PID']
        echodate = row['Echodate']
        root_path = row['root_path']
        label = row['LGE']
        name = row['demography_name']
        perfusion = int(row['灌注异常'])
        all_chamber_videos = self.all_chamber_info[self.all_chamber_info['PID']==PID]
        all_chamber_videos = all_chamber_videos[all_chamber_videos['Echodate']==echodate]
        if len(all_chamber_videos) == 0:
            all_chamber_videos = self.all_chamber_info[self.all_chamber_info['demography_name']==name]
            all_chamber_videos = all_chamber_videos[all_chamber_videos['Echodate']==echodate]
        if len(all_chamber_videos) == 0:
            print(PID, echodate)
        assert len(all_chamber_videos) > 0
        all_chamber_videos_row = all_chamber_videos.iloc[0]
        try:
            lge_value = row[['LGE1','LGE2','LGE3','LGE4','LGE5','LGE6','LGE7','LGE8','LGE9','LGE10','LGE11','LGE12','LGE13','LGE14','LGE15','LGE16']].values.tolist()
            if lge_value[0] == 'none':
                lge_value = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                lgevalue_weight = 0
            else:
                lge_value = list(map(float, lge_value))
                lgevalue_weight = 1
        except:
            lge_value = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            lgevalue_weight = 0
        # lge_value = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #####################
        # if sum(lge_value) > 1000:
        #     print('error', lge_value)
        lge_value = np.array(lge_value)
        all_arrays = []
        mask_arrays = []
        self.masked_attentions = []
        complete_view = True
        if self.args.chamber_list != 'ecg':  
            for chamber in self.chamber_list:
                if isinstance(all_chamber_videos_row[chamber], str):
                    all_paths = eval(all_chamber_videos_row[chamber])
                    find = False
                    while not find:
                        path = random.choice(all_paths)
                        seg_files = glob('./data/EchoLGE_multi_task/' + '/*/*/' + '_'.join(path.split('/'))+'_*_*.npz')
                        if len(seg_files) != 0:
                            # videos = glob(root_path + '/' + path +'_*' + '.npz')
                            # if len(videos) != 0:
                            find = True
                    # path = row[chamber]
                else:
                    path = row[chamber]
                if chamber != 'mri' and chamber != 'ecg' :
                    if not isinstance(path, str):
                        complete_view = False
                    if not self.args.add_flow:
                        array, mask = self.read_chamber(path, chamber, root_path)
                        # array = skimage.transform.resize(array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                        # mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)
                        if self.args.add_mask:
                            array = np.concatenate([array[:,:,:,np.newaxis], mask[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)
                        else:
                            array = np.concatenate([array[:,:,:,np.newaxis], array[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)
                    else:
                        array, mask = self.read_chamber_flow(path, chamber, root_path)
                        # array = skimage.transform.resize(array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                        # mask = skimage.transform.resize(mask, (2, self.length, self.args.frame_size, self.args.frame_size), 0)
                        array = array[:,:,:,np.newaxis]
                        mask = mask.transpose(1,2,3,0)
                        array = np.concatenate([array, mask], axis=-1)

                elif chamber == 'mri':
                    path = row[chamber]
                    array, mask = self.read_mri(path)
                    all_zeros = np.all(array == 0)
                    # if all_zeros:
                    #     print(path)
                    array = np.concatenate([array[:,:,:,np.newaxis], array[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)
                all_arrays.append(array)
                mask_arrays.append(mask)
            all_arrays = np.array(all_arrays)
            # mask_arrays = np.array(mask_arrays)

            # random_echo
            if self.train and random.choice([0,0,1]) and complete_view and len(self.args['chamber_list'].split('+'))>1:
                random_index = random.choice([0,1,2,3,4])
                all_arrays[random_index] = np.zeros((self.length, self.args.frame_size, self.args.frame_size, 3))
                self.masked_attentions[random_index] = np.zeros((self.length))
                # mask_arrays[random_index] = np.zeros((self.length, self.args.frame_size, self.args.frame_size))
            self.masked_attentions = np.array(self.masked_attentions)
        else:
            all_arrays = np.zeros((5, self.length, self.args.frame_size, self.args.frame_size, 3))
            self.masked_attentions = np.zeros((5, self.length))
        cine = torch.tensor(all_arrays).type(torch.FloatTensor)
        #########################################EchoPrime###############################################################
        cine_seg = cine[:,:,:,:,1:2].clone()
        cine[:,:,:,:,1] = cine[:,:,:,:,0]
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(1, 1, 1, 1,3)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(1, 1, 1, 1,3)
        cine = (cine*255).sub_(mean).div_(std)

        # ecg
        random_ecg = False
        if self.train:
            if self.args.chamber_list != 'ecg':
                random_ecg = random.choice([0,1])
            else:
                random_ecg = False
        if not random_ecg:
            if self.args.add_ecg:
                ecg_path = row['ecg']
                ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(ecg_path)
            else:
                ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(None)
        else:
            ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(None)


    
        return {
            "vid": cine,
            "seg": cine_seg,
            "ecg_data":ecg_data,
            "ecg_mask":ecg_mask,
            "label": torch.tensor(label).type(torch.FloatTensor),
            "perfusion": torch.tensor(perfusion).type(torch.FloatTensor),
            "lge_value":torch.tensor(lge_value).type(torch.FloatTensor),
            "lge_loc":torch.tensor(lge_value>0).type(torch.FloatTensor),
            "lgevalue_weight":torch.tensor(lgevalue_weight).type(torch.FloatTensor),
            "mask": torch.tensor(self.masked_attentions, dtype=torch.bool),
            "ecg_concat_mask":ecg_concat_mask,
            # "lv_mask": lv_mask,
            "lv_mask": False,
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
            "sample_id":str(PID)+'_'+str(echodate)
        }

    def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]
                else:
                    vid = vid[-self.max_frames :]

        return vid, mask

    def window(self, array, window_cen, window_wid):
        # if len(window_cen) != array.shape[0] or len(window_wid) != array.shape[0]:
        #     raise ValueError("Length of window center and width lists must match the number of slices.")
        if not isinstance(window_wid, list):
            window_wid = [window_wid] * len(window_cen)
        window_cen = [float(c) for c in window_cen]
        window_wid = [float(w) for w in window_wid]
        window_cen_array = np.array(window_cen)
        window_wid_array = np.array(window_wid)

        window_cen_broadcast = window_cen_array[:, np.newaxis, np.newaxis]  
        window_wid_broadcast = window_wid_array[:, np.newaxis, np.newaxis]
        
        window_min = window_cen_broadcast - window_wid_broadcast / 2.0
        window_max = window_cen_broadcast + window_wid_broadcast / 2.0
        
        windowed_array = np.clip(array, window_min, window_max)
        windowed_array = (windowed_array - window_min) / (window_max - window_min)
        return windowed_array

    
    def load_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def load_mri(self, file_path):
        data = nib.load(file_path)
        array = np.array(data.dataobj)
        orientation = nib.aff2axcodes(data.affine)
        if orientation == ('P', 'I', 'R'):
            array = array.transpose(1, 0, 2)
            array = np.flip(array, axis=0)
            orientation = ('L', 'P', 'S')
        return array
        
    def read_mri(self, mri_path):
        if isinstance(mri_path, str):
            mri_files = glob(mri_path+'/*.nii')
            if len(mri_files) == 0:
                print(mri_path, mri_files)
            mri_file = random.choice(mri_files)
            json_file = mri_file.replace('.nii', '.json')
            #seg_file = mri_file.replace('.nii', '_lge_seg.npy')
            #if not os.path.exists(seg_file):
            seg_file = '/'.join(mri_file.split('/')[:-1]) + '/' + 'lge_seg.npy'
            if not os.path.exists(seg_file):
                print('not exists', seg_file)
                seg_file = '../data/HuaxiCMR_processed/DMD_MRI_processed/00339480_20200905_zhao jia rui/lge_seg.npy'
            seg_array = np.load(seg_file)
            array_mri = self.load_mri(mri_file)
            json_mri = self.load_json(json_file)
            window_cen = json_mri['window_cen']
            window_wid = json_mri['window_wid']
            array_mri = array_mri.transpose(2,0,1)
            array_mri = self.window(array_mri, window_cen, window_wid)
            seg_array = skimage.transform.resize(seg_array, (seg_array.shape[0], array_mri.shape[1], array_mri.shape[2]), 0)

            array_mri, coord = self.crop(array_mri, seg_array)
            try:
                array_mri = skimage.transform.resize(array_mri, (self.length, self.args.frame_size, self.args.frame_size), 1)
            except:
                print(array_mri.shape, coord, mri_path)
            mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
            if self.mri_transform and self.train:
            # if 0:
                array_mri = array_mri.transpose(1,2,0)
                mask_mri = mask_mri.transpose(1,2,0)
                augs = self.mri_transform(image=array_mri, mask=mask_mri)
                array_mri, mask_mri = augs['image'].transpose(2,0,1), augs['mask'].transpose(2,0,1)

            if self.train:
            # if 0:
                gamma = random.uniform(0.8,1.3)
                array_mri = gammacorrection((array_mri*255).astype('uint8'), gamma, None)
        else:
            array_mri = np.zeros((self.length, self.args.frame_size, self.args.frame_size))
            mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
            
        return array_mri, mask_mri

    def crop(self, image, seg):
        image = image.squeeze()
        seg = seg.squeeze()
        _, xx, yy = np.where(seg)
        xx_min, xx_max, yy_min, yy_max = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        if len(image.shape) == 2:
            image = image[np.newaxis]
        #thresh = min([xx_min, yy_min, image.shape[1]-xx_max, image.shape[2]-yy_max, 20])
        bias = random.randint(0,20)
        minx = max([xx_min-bias, 0])
        maxx = min([xx_max+bias, image.shape[1]])
        miny = max([yy_min-bias, 0])
        maxy = min([yy_max+bias, image.shape[2]])
        croped = image[:, minx:maxx, miny:maxy]
        return croped, [minx, maxx, miny, maxy]

    def read_chamber(self, ch_path, chamber, root_path):
        if not isinstance(ch_path, str) :
            have_zero = True
            image_array = np.zeros((self.length,self.size, self.size))
            mask = np.zeros((self.length,self.size, self.size))
            self.masked_attentions.append(np.zeros(self.length))
        else:
            if chamber in self.apical_chamber:
                seg_path = self.apical_seg_path
            if chamber in self.short_chamber:
                seg_path = self.short_seg_path
            #print('_'.join(ch_path.split('/')), seg_path + '/*/' + '_'.join(ch_path.split('/'))+'_*_*.nii.gz')
            seg_files = glob('./data/EchoLGE_multi_task/' + '/*/*/' + '_'.join(ch_path.split('/'))+'_*_*.npz')
            save_processed_file = random.choice(seg_files)
            
            # save_processed_file = seg_file.replace(seg_path, './linux_code/data/EchoLGE_multi_task/')
            # save_processed_file = save_processed_file.replace('.nii.gz', '.npz')
            if not os.path.exists(save_processed_file):
                if not os.path.exists('/'.join(save_processed_file.split('/')[:-1])):
                    os.makedirs('/'.join(save_processed_file.split('/')[:-1]))
                print('saved', save_processed_file)
                mask_dcm = sitk.ReadImage(seg_file)
                mask = sitk.GetArrayFromImage(mask_dcm)
                cycle_num = seg_file.split('_')[-2]
                #print(seg_files,cycle_num)

                image_paths = glob(root_path + '/' + ch_path + '_' + str(cycle_num) + '_*' + '.npz')
                image_path = random.choice(image_paths)
                image_array = np.load(image_path)['array']
                # array = skimage.transform.resize(array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                # mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)
                
                image_array = skimage.transform.resize(image_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)
                np.savez(save_processed_file, processed_img=image_array, processed_mask=mask)
            else:
                array = np.load(save_processed_file)
                image_array = array['processed_img']
                mask = array['processed_mask']
                if image_array.shape[-1] != self.args.frame_size:
                    image_array = skimage.transform.resize(image_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                    mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)

                # if image_array != [16,128,128]:
                #     image_array = skimage.transform.resize(image_array, (self.length, self.size, self.size), 1)

            if self.transform and self.train:
                image_array = image_array.transpose(1,2,0)
                mask = mask.transpose(1,2,0)
                augs = self.transform(image=image_array, mask=mask)
                image_array, mask = augs['image'].transpose(2,0,1), augs['mask'].transpose(2,0,1)

            if self.train:
                gamma = random.uniform(0.1, 2.0)
                image_array = gammacorrection((image_array*255).astype('uint8'), gamma, None)
            image_array = (image_array - np.amin(image_array)) / (np.amax(image_array)-np.amin(image_array))
            self.masked_attentions.append(np.ones(self.length))
        return image_array, mask

    def read_chamber_flow(self, ch_path, chamber, root_path):
        if not isinstance(ch_path, str) :
            have_zero = True
            image_array = np.zeros((self.length,self.size, self.size))
            masked_flow = np.zeros((2, self.length,self.size, self.size))
        else:
            if chamber in self.apical_chamber:
                seg_path = self.apical_seg_path
            if chamber in self.short_chamber:
                seg_path = self.short_seg_path
            #print('_'.join(ch_path.split('/')), seg_path + '/*/' + '_'.join(ch_path.split('/'))+'_*_*.nii.gz')
            flow_mask_files = glob(self.flow_path + '/' + ch_path + '_*' + '.npz')
            flow_mask_file = random.choice(flow_mask_files)
            save_processed_file = flow_mask_file.replace(self.flow_path, '../data/save_processed/EchoLGE_flow/')
            if not os.path.exists(save_processed_file):
                if not os.path.exists('/'.join(save_processed_file.split('/')[:-1])):
                    os.makedirs('/'.join(save_processed_file.split('/')[:-1]))
                flow_mask = np.load(flow_mask_file)
                flow, _, flow_rgb = flow_mask['flow'], flow_mask['mask'], flow_mask['flow_rgb']
                cycle_num = flow_mask_file.split('.')[0].split('_')[-1]

                seg_files = glob(seg_path + '/*/*/' + '_'.join(ch_path.split('/'))+ '_' + str(cycle_num) + '_*' + '.nii.gz')
                seg_file = seg_files[0]
                mask_dcm = sitk.ReadImage(seg_file)
                mask = sitk.GetArrayFromImage(mask_dcm)
                mask = skimage.transform.resize(mask, (flow.shape[0], flow.shape[2], flow.shape[3]), 0)
                masked = mask == np.amax(mask)
                if len(masked.shape) == 3:
                    masked = masked[:16,np.newaxis]
                else:
                    masked = masked[:16]

                masked_flow = masked*flow
                masked_flow = np.concatenate([masked_flow[:,0],masked_flow[:,1]], axis=0)
                masked_flow = skimage.transform.resize(masked_flow, (masked_flow.shape[0], self.size, self.size), 0)

                image_paths = glob(root_path + '/' + ch_path + '_' + str(cycle_num) + '_*' + '.npz')
                image_path = random.choice(image_paths)
                image_array = np.load(image_path)['array']
                image_array = skimage.transform.resize(image_array, (self.length, self.size, self.size), 1)
                np.savez(save_processed_file, processed_img=image_array, processed_flow=masked_flow, processed_mask=mask)
            else:
                array = np.load(save_processed_file)
                image_array = array['processed_img']
                masked_flow = array['processed_flow']


            if self.transform and self.train:
                image_array = image_array.transpose(1,2,0)
                masked_flow = masked_flow.transpose(1,2,0)
                augs = self.transform(image=image_array, mask=masked_flow)
                image_array, masked_flow = augs['image'].transpose(2,0,1), augs['mask'].transpose(2,0,1)

            if self.train:
                gamma = random.uniform(0.1, 2.0)
                image_array = gammacorrection((image_array*255).astype('uint8'), gamma, None)
            image_array = (image_array - np.amin(image_array)) / (np.amax(image_array)-np.amin(image_array))
            masked_flow = np.concatenate([masked_flow[:16][np.newaxis], masked_flow[16:][np.newaxis]], axis=0)
        return image_array, masked_flow

    def load_ecg(self, ecg_filename):
        #ecg_filename = 'G:/HuaxiECG/processed_ecg/' + '_'.join(ecg_filename.split('/')[-2:])[:-4] + '.npy'
        # ecg_data = self.load_myecg(ecg_filename)
        if isinstance(ecg_filename, str):
            ecg_filename = ecg_filename.replace('../data/HuaxiECG','./data/HuaxiECG')
            ecg_concat_mask = np.ones(1)
            ecg_data = np.load(ecg_filename)
            if self.args.random_crop: 
                start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1)
                ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
            
            mask = np.isnan(ecg_data)
            ecg_data = np.where(mask, ecg_data[~mask].mean(), ecg_data)
            
            # flatten the leads 
            ecg_data = ecg_data.reshape(-1) # (12*SAMPLES_IN_5_SECONDS_AT_500HZ,)
            
            # downsampling 
            if self.args.downsampling_factor is not None:
                ecg_data = signal.decimate(ecg_data, self.args.downsampling_factor)
                
            # compute attention mask
            if not self.encode:
                if self.beat_based_attention_mask:
                    attention_mask = self.compute_beat_based_attention_mask(ecg_data)
                else:
                    attention_mask = self.compute_attention_mask_for_padding(ecg_data)
            
                
            if self.pretrain:
                
                feat_path = os.path.join(self.features_path, ecg_filename)
                features = np.load(feat_path, allow_pickle=True)                
                
                # [ensamble_length, n_tokens], where values on row i-th are in [0, V_i - 1] and V_i is the number of clusters for the i-th kmeans model
                labels = [kmeans.predict(features).tolist() for kmeans in self.ensamble_kmeans] 
                
                output = (
                    torch.from_numpy(ecg_data.copy()).float(),
                    torch.from_numpy(attention_mask.copy()).long(),
                    torch.Tensor(labels).long()    
                )

                return output
            
            elif self.encode:
                
                return torch.from_numpy(ecg_data.copy()).float(), ecg_filename
            
            else: # finetuning
                # output = (
                #     torch.from_numpy(ecg_data.copy()).float(),
                #     torch.from_numpy(attention_mask.copy()).long(),
                #print('ecg_data', ecg_data.shape, attention_mask.shape)
                return torch.from_numpy(ecg_data.copy()).float(), torch.from_numpy(attention_mask.copy()).long(),torch.from_numpy(ecg_concat_mask).bool()
        else:
            ecg_concat_mask = np.zeros(1)
            ecg_data = np.zeros(6000)
            attention_mask = np.ones(6000)
            return torch.from_numpy(ecg_data.copy()).float(), torch.from_numpy(attention_mask.copy()).long(), torch.from_numpy(ecg_concat_mask).bool()


        
    def compute_attention_mask_for_padding(self, array):
        array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ   
        for index in range(array.shape[1]):
            if np.any(array[:, index]):
                break
        start = index
        for index in range(array.shape[1]-1, -1, -1):
            if np.any(array[:, index]):
                break
        end = index
        attention_mask = np.zeros(array.shape[1])
        attention_mask[start:end+1] = 1
        attention_mask = np.repeat([attention_mask], 12, axis=0)
        attention_mask = np.concatenate(attention_mask, axis=0)
        return attention_mask
    
    def compute_beat_based_attention_mask(self, ecg_data):
        ''' 
        Computes attention mask focusing only on P wave, QRS complex and T wave
        '''
        
        ecg_data = ecg_data.reshape(12, SAMPLES_IN_5_SECONDS_AT_500HZ)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1
        
        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop
        
        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected
        
        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected
        
        
        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected
        
        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the t waves    
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the qrs complexes    
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1
        
        # global attention mask merging all interest regions    
        attention_mask = (p_wave | t_wave | qrs_complex).tolist() 
        attention_mask = np.repeat([attention_mask], 12, axis=0) # since the leads are temporally aligned, interest regions should be located within the same intervals
        attention_mask = np.concatenate(attention_mask, axis=0) 
        
        return attention_mask


class HuaxiLGEDataset_external_test(Dataset):
    def __init__(
        self,
        args,
        info_csv,
        dataset_path: str = "~/as",
        split: str = "train",
        mode: str = "as",
        max_frames: int = 16,
        transform=None,
        use_seg_labels=False,
        aug_transform=None,
        max_clips=1,
        mean_std=False,
    ):

        super().__init__()

        assert mode == "lge", "Only lge mode is supported"
        self.args = args
        self.beat_based_attention_mask = False
        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])

        # read in the data directory CSV as a pandas dataframe
        dataset = info_csv
        # Take train/test/val
        # Apply an arbitrary filter
        # filtering_function = filtering_functions["bicuspid"]
        # dataset = filtering_function(dataset)

        self.dataset = dataset
        self.dataset = csv2str(self.dataset, ['PID', 'Echodate'])
        self.length = self.args.length
        self.size = self.args.size
        self.max_frames = max_frames
        self.train = False
        self.trans = transform
        self.aug_trans = aug_transform
        self.max_clips = max_clips
        self.mean_std = mean_std
        self.chamber_list = self.args.chamber_list.split('+')
        self.apical_chamber = ['a4c', 'a3c', 'a2c']
        self.short_chamber = ['mid', 'mv']
        self.transform = imgaug

        self.encode = False
        self.pretrain = False
    def class_samplers(self):
        labels_AS = self.dataset['LGE'].values.tolist()
        class_sample_count_AS = np.array(
            [len(np.where(labels_AS == t)[0]) for t in np.unique(labels_AS)]
        )
        weight_AS = 1.0 / class_sample_count_AS
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        return sampler_AS

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        row = self.dataset.iloc[item]
        PID = row['PID']
        echodate = row['Echodate']
        root_path = row['root_path']
        label = row['LGE就是延迟强化（1=阳性，2=阴性）']
        name = row['demography_name']
        perfusion = int(row['灌注缺损（1=是，2=否）'])

        try:
            lge_value = row[['LGE1','LGE2','LGE3','LGE4','LGE5','LGE6','LGE7','LGE8','LGE9','LGE10','LGE11','LGE12','LGE13','LGE14','LGE15','LGE16']].values.tolist()
            if lge_value[0] is None:
                lge_value = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                lgevalue_weight = 0
            elif lge_value[0] == 'none':
                lge_value = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                lgevalue_weight = 0
            else:
                lge_value = list(map(float, lge_value))
                lgevalue_weight = 1
        except:
            lge_value = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            lgevalue_weight = 0
        lge_value = np.array(lge_value)
        all_arrays = []
        mask_arrays = []
        self.masked_attentions = []
        complete_view = True
        if self.args.chamber_list != 'ecg':  
            for chamber in self.chamber_list:
                all_paths = eval(row[chamber])
                if len(all_paths) > 0:
                    path = random.choice(all_paths)
                    path = path.replace('\\','/')
                else:
                    path = False
                if chamber != 'mri' and chamber != 'ecg' :
                    if not isinstance(path, str):
                        complete_view = False
                    if not self.args.add_flow:
                        array, mask = self.read_chamber(path, chamber, root_path)
                        # array = skimage.transform.resize(array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                        # mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)
                        if self.args.add_mask:
                            array = np.concatenate([array[:,:,:,np.newaxis], mask[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)
                        else:
                            array = np.concatenate([array[:,:,:,np.newaxis], array[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)

                elif chamber == 'mri':
                    path = row[chamber]
                    array, mask = self.read_mri(path)
                    all_zeros = np.all(array == 0)
                    # if all_zeros:
                    #     print(path)
                    array = np.concatenate([array[:,:,:,np.newaxis], array[:,:,:,np.newaxis], array[:,:,:,np.newaxis]],axis=-1)
                all_arrays.append(array)
                mask_arrays.append(mask)
            all_arrays = np.array(all_arrays)
            self.masked_attentions = np.array(self.masked_attentions)
        else:
            all_arrays = np.zeros((5, self.length, self.args.frame_size, self.args.frame_size, 3))
            self.masked_attentions = np.zeros((5, self.length))
        cine = torch.tensor(all_arrays).type(torch.FloatTensor)
        
        #########################################EchoPrime###############################################################
        cine_seg = cine[:,:,:,:,1:2].clone()
        cine[:,:,:,:,1] = cine[:,:,:,:,0]
        mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(1, 1, 1, 1,3)
        std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(1, 1, 1, 1,3)
        cine = (cine*255).sub_(mean).div_(std)

        # ecg
        random_ecg = False
        if not random_ecg:
            if self.args.add_ecg:
                ecg_root_path = '../data/HuaxiECG/processed_ecg_0228/'
                ecg_path = row['ecg_path']
                ecg_paths = glob(ecg_root_path + '/' + ecg_path + '*.npy')
                if len(ecg_paths) > 0:
                    ecg_path = ecg_paths[0]
                else:
                    ecg_path = None
                ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(ecg_path)
            else:
                ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(None)
        else:
            ecg_data, ecg_mask, ecg_concat_mask = self.load_ecg(None)


        print(torch.tensor(lge_value>0).type(torch.FloatTensor).shape)
        return {
            "vid": cine,
            "seg": cine_seg,
            "ecg_data":ecg_data,
            "ecg_mask":ecg_mask,
            "label": torch.tensor(label).type(torch.FloatTensor),
            "perfusion": torch.tensor(perfusion).type(torch.FloatTensor),
            "lge_value":torch.tensor(lge_value).type(torch.FloatTensor),
            "lge_loc":torch.tensor(lge_value>0).type(torch.FloatTensor),
            "lgevalue_weight":torch.tensor(lgevalue_weight).type(torch.FloatTensor),
            "mask": torch.tensor(self.masked_attentions, dtype=torch.bool),
            "ecg_concat_mask":ecg_concat_mask,
            # "lv_mask": lv_mask,
            "lv_mask": False,
            "ed_frame": 0,
            "ed_valid": False,
            "es_frame": 0,
            "es_valid": False,
            "class_label": torch.zeros(1),
            "sample_id":str(PID)+'_'+str(echodate)
        }

    def pad_vid(self, vid, mask, mask_idx, clip_idx=0):
        if vid.shape[0] < self.max_frames:
            mask[mask_idx, vid.shape[0] :] = False
            vid = torch.cat(
                (
                    vid,
                    torch.zeros(
                        self.max_frames - vid.shape[0], vid.shape[1], vid.shape[2]
                    ),
                ),
                dim=0,
            )
        else:
            if self.train:
                starting_idx = random.randint(0, vid.shape[0] - self.max_frames)
                vid = vid[starting_idx : starting_idx + self.max_frames]
            else:
                if (clip_idx + 1) * self.max_frames <= vid.shape[0]:
                    vid = vid[
                        self.max_frames * clip_idx : self.max_frames * (clip_idx + 1)
                    ]
                else:
                    vid = vid[-self.max_frames :]

        return vid, mask

    def window(self, array, window_cen, window_wid):
        # if len(window_cen) != array.shape[0] or len(window_wid) != array.shape[0]:
        #     raise ValueError("Length of window center and width lists must match the number of slices.")
        if not isinstance(window_wid, list):
            window_wid = [window_wid] * len(window_cen)
        window_cen = [float(c) for c in window_cen]
        window_wid = [float(w) for w in window_wid]
        window_cen_array = np.array(window_cen)
        window_wid_array = np.array(window_wid)

        window_cen_broadcast = window_cen_array[:, np.newaxis, np.newaxis]  
        window_wid_broadcast = window_wid_array[:, np.newaxis, np.newaxis]
        
        window_min = window_cen_broadcast - window_wid_broadcast / 2.0
        window_max = window_cen_broadcast + window_wid_broadcast / 2.0
        
        windowed_array = np.clip(array, window_min, window_max)
        windowed_array = (windowed_array - window_min) / (window_max - window_min)
        return windowed_array

    
    def load_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def load_mri(self, file_path):
        data = nib.load(file_path)
        array = np.array(data.dataobj)
        orientation = nib.aff2axcodes(data.affine)
        if orientation == ('P', 'I', 'R'):
            array = array.transpose(1, 0, 2)
            array = np.flip(array, axis=0)
            orientation = ('L', 'P', 'S')
        return array
        
    def read_mri(self, mri_path):
        if isinstance(mri_path, str):
            mri_files = glob(mri_path+'/*.nii')
            if len(mri_files) == 0:
                print(mri_path, mri_files)
            mri_file = random.choice(mri_files)
            json_file = mri_file.replace('.nii', '.json')
            #seg_file = mri_file.replace('.nii', '_lge_seg.npy')
            #if not os.path.exists(seg_file):
            seg_file = '/'.join(mri_file.split('/')[:-1]) + '/' + 'lge_seg.npy'
            if not os.path.exists(seg_file):
                print('not exists', seg_file)
                seg_file = '../data/HuaxiCMR_processed/DMD_MRI_processed/00339480_20200905_zhao jia rui/lge_seg.npy'
            seg_array = np.load(seg_file)
            array_mri = self.load_mri(mri_file)
            json_mri = self.load_json(json_file)
            window_cen = json_mri['window_cen']
            window_wid = json_mri['window_wid']
            array_mri = array_mri.transpose(2,0,1)
            array_mri = self.window(array_mri, window_cen, window_wid)
            seg_array = skimage.transform.resize(seg_array, (seg_array.shape[0], array_mri.shape[1], array_mri.shape[2]), 0)

            array_mri, coord = self.crop(array_mri, seg_array)
            try:
                array_mri = skimage.transform.resize(array_mri, (self.length, self.args.frame_size, self.args.frame_size), 1)
            except:
                print(array_mri.shape, coord, mri_path)
            mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
            if self.mri_transform and self.train:
            # if 0:
                array_mri = array_mri.transpose(1,2,0)
                mask_mri = mask_mri.transpose(1,2,0)
                augs = self.mri_transform(image=array_mri, mask=mask_mri)
                array_mri, mask_mri = augs['image'].transpose(2,0,1), augs['mask'].transpose(2,0,1)

            if self.train:
            # if 0:
                gamma = random.uniform(0.8,1.3)
                array_mri = gammacorrection((array_mri*255).astype('uint8'), gamma, None)
        else:
            array_mri = np.zeros((self.length, self.args.frame_size, self.args.frame_size))
            mask_mri = np.ones((self.length, self.args.frame_size, self.args.frame_size))
            
        return array_mri, mask_mri

    def crop(self, image, seg):
        image = image.squeeze()
        seg = seg.squeeze()
        _, xx, yy = np.where(seg)
        xx_min, xx_max, yy_min, yy_max = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        if len(image.shape) == 2:
            image = image[np.newaxis]
        #thresh = min([xx_min, yy_min, image.shape[1]-xx_max, image.shape[2]-yy_max, 20])
        bias = random.randint(0,20)
        minx = max([xx_min-bias, 0])
        maxx = min([xx_max+bias, image.shape[1]])
        miny = max([yy_min-bias, 0])
        maxy = min([yy_max+bias, image.shape[2]])
        croped = image[:, minx:maxx, miny:maxy]
        return croped, [minx, maxx, miny, maxy]

    def read_chamber(self, ch_path, chamber, root_path):
        if not isinstance(ch_path, str) :
            have_zero = True
            image_array = np.zeros((self.length,self.size, self.size))
            mask = np.zeros((self.length,self.size, self.size))
            self.masked_attentions.append(np.zeros(self.length))
        else:
            try:
                image_path = root_path + '/' + ch_path
                save_processed_file = image_path.replace(root_path, './data/EchoLGE_multi_task/')
                save_processed_file = save_processed_file.replace('.nii.gz', '.npz')
                if not os.path.exists(save_processed_file):
                    if not os.path.exists('/'.join(save_processed_file.split('/')[:-1])):
                        os.makedirs('/'.join(save_processed_file.split('/')[:-1]))
                    print('saved', save_processed_file)
                    mask = np.zeros((self.length,self.size, self.size))
                    image_array = np.load(image_path)['array']
                    # array = skimage.transform.resize(array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                    # mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)

                    image_array = skimage.transform.resize(image_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                    mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)
                    np.savez(save_processed_file, processed_img=image_array, processed_mask=mask)
                else:
                    array = np.load(save_processed_file)
                    image_array = array['processed_img']
                    mask = array['processed_mask']
                    if image_array.shape[-1] != self.args.frame_size:
                        image_array = skimage.transform.resize(image_array, (self.length, self.args.frame_size, self.args.frame_size), 1)
                        mask = skimage.transform.resize(mask, (self.length, self.args.frame_size, self.args.frame_size), 0)

                    # if image_array != [16,128,128]:
                    #     image_array = skimage.transform.resize(image_array, (self.length, self.size, self.size), 1)
                image_array = (image_array - np.amin(image_array)) / (np.amax(image_array)-np.amin(image_array))
                self.masked_attentions.append(np.ones(self.length))
            except:
                print('error', image_path)
                have_zero = True
                image_array = np.zeros((self.length,self.size, self.size))
                mask = np.zeros((self.length,self.size, self.size))
                self.masked_attentions.append(np.zeros(self.length))

        return image_array, mask

    def load_ecg(self, ecg_filename):
        #ecg_filename = 'G:/HuaxiECG/processed_ecg/' + '_'.join(ecg_filename.split('/')[-2:])[:-4] + '.npy'
        # ecg_data = self.load_myecg(ecg_filename)
        if isinstance(ecg_filename, str):
            ecg_concat_mask = np.ones(1)
            ecg_data = np.load(ecg_filename)
            if self.args.random_crop: 
                start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1)
                ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
            
            mask = np.isnan(ecg_data)
            ecg_data = np.where(mask, ecg_data[~mask].mean(), ecg_data)
            
            # flatten the leads 
            ecg_data = ecg_data.reshape(-1) # (12*SAMPLES_IN_5_SECONDS_AT_500HZ,)
            
            # downsampling 
            if self.args.downsampling_factor is not None:
                ecg_data = signal.decimate(ecg_data, self.args.downsampling_factor)
                
            # compute attention mask
            if not self.encode:
                if self.beat_based_attention_mask:
                    attention_mask = self.compute_beat_based_attention_mask(ecg_data)
                else:
                    attention_mask = self.compute_attention_mask_for_padding(ecg_data)
            
                
            if self.pretrain:
                
                feat_path = os.path.join(self.features_path, ecg_filename)
                features = np.load(feat_path, allow_pickle=True)                
                
                # [ensamble_length, n_tokens], where values on row i-th are in [0, V_i - 1] and V_i is the number of clusters for the i-th kmeans model
                labels = [kmeans.predict(features).tolist() for kmeans in self.ensamble_kmeans] 
                
                output = (
                    torch.from_numpy(ecg_data.copy()).float(),
                    torch.from_numpy(attention_mask.copy()).long(),
                    torch.Tensor(labels).long()    
                )

                return output
            
            elif self.encode:
                
                return torch.from_numpy(ecg_data.copy()).float(), ecg_filename
            
            else: # finetuning
                # output = (
                #     torch.from_numpy(ecg_data.copy()).float(),
                #     torch.from_numpy(attention_mask.copy()).long(),
                #print('ecg_data', ecg_data.shape, attention_mask.shape)
                return torch.from_numpy(ecg_data.copy()).float(), torch.from_numpy(attention_mask.copy()).long(),torch.from_numpy(ecg_concat_mask).bool()
        else:
            ecg_concat_mask = np.zeros(1)
            ecg_data = np.zeros(6000)
            attention_mask = np.ones(6000)
            return torch.from_numpy(ecg_data.copy()).float(), torch.from_numpy(attention_mask.copy()).long(), torch.from_numpy(ecg_concat_mask).bool()


        
    def compute_attention_mask_for_padding(self, array):
        array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ   
        for index in range(array.shape[1]):
            if np.any(array[:, index]):
                break
        start = index
        for index in range(array.shape[1]-1, -1, -1):
            if np.any(array[:, index]):
                break
        end = index
        attention_mask = np.zeros(array.shape[1])
        attention_mask[start:end+1] = 1
        attention_mask = np.repeat([attention_mask], 12, axis=0)
        attention_mask = np.concatenate(attention_mask, axis=0)
        return attention_mask
    
    def compute_beat_based_attention_mask(self, ecg_data):
        ''' 
        Computes attention mask focusing only on P wave, QRS complex and T wave
        '''
        
        ecg_data = ecg_data.reshape(12, SAMPLES_IN_5_SECONDS_AT_500HZ)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1
        
        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop
        
        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected
        
        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected
        
        
        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected
        
        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the t waves    
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the qrs complexes    
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1
        
        # global attention mask merging all interest regions    
        attention_mask = (p_wave | t_wave | qrs_complex).tolist() 
        attention_mask = np.repeat([attention_mask], 12, axis=0) # since the leads are temporally aligned, interest regions should be located within the same intervals
        attention_mask = np.concatenate(attention_mask, axis=0) 
        
        return attention_mask
