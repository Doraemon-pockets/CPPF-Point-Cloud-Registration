"""Data loader
"""
import argparse
import logging
import os
from typing import List

import h5py
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision

import data_loader.transforms as Transforms
import common.math.se3 as se3

_logger = logging.getLogger()


def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    if args.train_categoryfile:
        train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
        train_categories.sort()
    if args.val_categoryfile:
        val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
        val_categories.sort()

    train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                                      args.num_points, args.partial)
    _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'modelnet_hdf':
        train_data = ModelNetHdf(args.dataset_path, subset='train', categories=train_categories,
                                 transform=train_transforms)
        val_data = ModelNetHdf(args.dataset_path, subset='test', categories=val_categories,
                               transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    test_categories = None
    if args.test_category_file:
        test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
        test_categories.sort()

    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                        args.num_points, args.partial)
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'modelnet_hdf':
        test_data = ModelNetHdf(args.dataset_path, subset='test', categories=test_categories,
                                transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        # test_transforms = [Transforms.SetDeterministic(),
        #                    Transforms.FixedResampler(num_points),
        #                    Transforms.SplitSourceRef(),
        #                    Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
        #                    Transforms.ShufflePoints()]
        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag)]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', categories: List = None, transform=None,
                 angle_threshold: float = 30.0):
        """ModelNet40 dataset with adaptive normal angle-based downsampling.

        Args:
            dataset_path (str): Folder containing processed dataset.
            subset (str): Dataset subset, either 'train' or 'test'.
            categories (list): Categories to use.
            transform (callable, optional): Optional transform to be applied on a sample.
            angle_threshold (float): Threshold angle (in degrees) for downsampling.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self.angle_threshold = angle_threshold  # Angle threshold for downsampling

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))
        self._logger.info('Loading data from {} for {}'.format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._download_dataset(dataset_path)

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info('Categories used: {}.'.format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info('Using all categories.')

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        self._data = self._data.astype(np.float32)  # Convert to float32

        self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(self._data.shape[0], subset))

    def __getitem__(self, item):
        points = self._data[item, :, :]
        normals = points[:, 3:]  # Assuming normals are stored in the last columns

        # Apply normal angle-based downsampling
        downsampled_points = self.normal_angle_downsampling(points, normals)

        sample = {'points': downsampled_points, 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def normal_angle_downsampling(self, points, normals):
        """
        Downsamples the point cloud based on the angle between the normal vectors.
        Points with larger normal angles are kept, and those with smaller angles are discarded.

        Args:
            points (np.ndarray): Point cloud data of shape (N, 3).
            normals (np.ndarray): Normals of the point cloud of shape (N, 3).

        Returns:
            np.ndarray: Downsampled point cloud.
        """
        num_points = points.shape[0]
        keep_indices = []

        # Calculate angles between normal vectors
        for i in range(1, num_points - 1):

            normal1 = normals[i - 1]
            normal2 = normals[i]
            normal3 = normals[i + 1]

            angle = np.arccos(
                np.clip(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)), -1.0, 1.0))
            angle = np.degrees(angle)  # Convert to degrees

            # If the angle is above the threshold, keep the point
            if angle >= self.angle_threshold:
                keep_indices.append(i)
        return points[keep_indices]

    @staticmethod
    def _read_h5_files(fnames, categories):
        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]
