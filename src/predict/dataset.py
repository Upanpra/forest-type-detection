import os
from typing import Optional, Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.MultipleImageDataloader import MultiImageFolder
from src.data_loader import paired_image_transforms_aug
from src.functions import my_tiff_loader, set_hyper_no_data_values_as_zero, set_chm_no_data_values_as_zero


class PredictFolderDataset(Dataset):
    """Load all images in a folder."""

    def __init__(self, input_folder: str, transform=None, ext: Optional[Tuple[str]] = (".tif", ".TIF")):
        """
        Args:
            input_folder: str: path to the input folder of images to load
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ext: if non-None, only load files matching this extension
        """
        self.input_folder = input_folder
        self.images = os.listdir(self.input_folder)
        if ext is not None:
            self.images = [x for x in self.images if any(x.endswith(ex) for ex in ext)]
        self.transform = transform
        print(f"Built PredictFolderDataset of length {len(self)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.input_folder, self.images[idx])
        image = my_tiff_loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'filename': self.images[idx], "full_input_path": img_name}

        return sample


class MultiModalityPredictFolderDataset(Dataset):
    """Load all images from a list of folders."""

    def __init__(self, input_folder: List[str], transform=None, input_shape: Optional[Tuple] = None, ext: Optional[Tuple[str]] = (".tif", ".TIF")):
        """
        Args:
            input_folder: list of str: paths to the input folder of images to load and concat
            transform (callable, optional): Optional transform to be applied
                on a sample.
            input_shape: optional 2-tuple of image shape
            ext: if non-None, only load files matching this extension
        """
        self.input_folder = input_folder
        self.transform = transform
        self.input_shape = input_shape

        self.images = os.listdir(self.input_folder[0])
        if ext is not None:
            self.images = [x for x in self.images if any(x.endswith(ex) for ex in ext)]
        print(f"Built PredictFolderDataset of length {len(self)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, Any]:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        full_input_path = os.path.join(self.input_folder[0], self.images[idx])  # this is used later to get the geospatial information when writing out the prediction to a new tiff file

        basename = self.images[idx]

        images = []
        paths = []  # only used on exception
        for i, root in enumerate(self.input_folder):
            path = os.path.join(root, basename)
            paths.append(path)
            img = my_tiff_loader(path)

            if i == 1:
                # use CHM no-data cleaner for second root
                img_tensor = set_chm_no_data_values_as_zero(torch.from_numpy(img))
            else:
                img_tensor = set_hyper_no_data_values_as_zero(torch.from_numpy(img))

            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(-1)
            else:
                assert len(
                    img_tensor.shape) == 3, f"Loaded tensors should be rank 2 or 3 but found {len(img_tensor.shape)} for {path}"
            img_tensor = torch.transpose(img_tensor, 0, 2)

            if self.input_shape is not None:
                img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(
                    0)
            images.append(img_tensor)
        try:
            combined_torch = torch.cat(images, dim=0)
        except Exception as e:
            print(f"shapes: {[x.shape for x in images]}")
            print(f"paths: {paths}")
            raise e
        if self.transform is not None:
            image = self.transform(combined_torch)

        sample = {'image': image, 'filename': self.images[idx], "full_input_path": full_input_path}

        return sample

def transforms_aug(input_size, mean, std, channel_indices=None):
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
         transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
         #          transforms.Lambda(lambda x: (x - mean) / std)
         transforms.Normalize(mean, std),
         # FIXME: add back in augs
         # transforms.ToPILImage(),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomHorizontalFlip(p=0.5),
         #transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
         #transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio = (0.8,1.2))
         ])
    # if ch_indices is not None:
    # trans_train.append(ChannelSelectorTransform(ch_indices)

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
         transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
         #          transforms.Lambda(lambda x: (x - mean) / std)
         transforms.Normalize(mean, std)
         ])

    return transform_train, transform_test


def get_predict_loader(tiff_input_folder: str, input_size, mean, std, batch_size: int):
    transform_train, transform_test = transforms_aug(input_size, mean, std)

    dataset = PredictFolderDataset(tiff_input_folder, transform_test)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

def multi_image_data_loader(train_folders: List[str], input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = paired_image_transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = MultiModalityPredictFolderDataset(train_folders, transform = transform_train, input_shape=input_size)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader