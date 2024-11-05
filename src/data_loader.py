from typing import List
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from src.data_loader_utils import create_balanced_sampler
from src.functions import my_tiff_loader, set_hyper_no_data_values_as_zero, set_chm_no_data_values_as_zero


class ChannelSelectorTransform:

    def __init__(self, channel_indices: List[int]):
        print("using channel selector")

        self.channel_indices = channel_indices

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs should be of shape C X H X W
        assert isinstance(inputs, torch.Tensor), f"{type(inputs)} != torch.Tensor"
        assert len(inputs.shape) == 3, f"{inputs.shape} should be rank 3"
        res = inputs[self.channel_indices, :, :]
        return res


def transforms_aug(input_size, mean, std, channel_indices=None):
    transform_train = [
     transforms.ToTensor(),
     transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
     #transforms.Lambda(lambda x: set_chm_no_data_values_as_zero(x)),
     transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
     transforms.Normalize(mean, std),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
     #transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio = (0.8,1.2))
     ]

    if channel_indices is not None:
        transform_train.append(ChannelSelectorTransform(channel_indices))
    transform_train = transforms.Compose(transform_train)

    transform_test = [
    transforms.ToTensor(),
    transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
    #transforms.Lambda(lambda x: set_chm_no_data_values_as_zero(x)),
    transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
    transforms.Normalize(mean, std)
     ]

    if channel_indices is not None:
        transform_test.append(ChannelSelectorTransform(channel_indices))

    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test

def paired_image_transforms_aug(input_size, mean, std, channel_indices=None):
    print(f"shape of mean, {mean.shape}")
    print(f"shape of std, {std.shape}")
    transform_train = [
     #transforms.Lambda(lambda x: set_chm_no_data_values_as_zero(x)), #
     transforms.Normalize(mean, std),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
     transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio = (0.8,1.2))
     ]

    if channel_indices is not None:
        transform_train.append(ChannelSelectorTransform(channel_indices))
    transform_train = transforms.Compose(transform_train)

    transform_test = [
    #transforms.Lambda(lambda x: set_chm_no_data_values_as_zero(x)),
    transforms.Normalize(mean, std)
     ]

    if channel_indices is not None:
        transform_test.append(ChannelSelectorTransform(channel_indices))
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def data_loader_balanced(train_folder, test_folder, input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = datasets.ImageFolder(train_folder, loader=my_tiff_loader, transform=transform_train)
    balance_sampler = create_balanced_sampler(trainset)
    testset = datasets.ImageFolder(root=test_folder, transform=transform_test, loader=my_tiff_loader)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,sampler= balance_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def data_loader(train_folder, test_folder, input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = datasets.ImageFolder(train_folder, loader=my_tiff_loader, transform=transform_train)
    testset = datasets.ImageFolder(root=test_folder, transform=transform_test, loader=my_tiff_loader)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader