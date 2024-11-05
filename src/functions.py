# all functions used for CNN run

# calculating mean and standard deviation for normalizing the data
import math
import torch  # Overall PyTorch import
import torchvision.datasets as datasets  # To download data
import torchvision.transforms as transforms  # For pre-processing data
import tifffile
from typing import Optional
import numpy as np

# custom loader for more than 3 channel tif input


def rasterio_tiff_loader(filename, default_value_for_nans: Optional[float] = 0.0):
    with rasterio.open(filename) as src:
        res = src.read()
    if default_value_for_nans is not None:
        res[np.isnan(res)] = default_value_for_nans
    return res


def my_tiff_loader(filename, default_value_for_nans: Optional[float] = 0.0):
    res = tifffile.imread(filename)
    if default_value_for_nans is not None:
        res[np.isnan(res)] = default_value_for_nans
    return res

# Setting NAN to zero (particularly used for hyperspectral data)
def set_hyper_no_data_values_as_zero(data):
    #data[data < -9999] = 0
    data[torch.isnan(data)] = 0
    data[torch.isinf(data)] = 0
    return data

#setting any negative CHM value to 0
def set_chm_no_data_values_as_zero(data):
    data[data < 0] = 0
    data[torch.isnan(data)] = 0
    data[torch.isinf(data)] = 0
    return data


def get_mean_std(train_folder, modality: str = "chm"):  # str was set to CHM
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.ImageFolder(train_folder, loader=my_tiff_loader, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for image_label in train_loader:
        data = image_label[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        if modality == "hyper":
            data = set_hyper_no_data_values_as_zero(data)
        elif modality == "chm":
            data = set_chm_no_data_values_as_zero(data)
        mean += torch.mean(data, 2).sum(0)
        std += torch.std(data, 2).sum(0)
        #     mean += data.mean(2).sum(0)
        #     std += data.std(2).sum(0)
        nb_samples += batch_samples

    # print(mean)
    # print(std)

    mean /= nb_samples
    std /= nb_samples
    # print(mean)
    # print(std)
    return mean, std


def load_model(net: torch.nn.Module, checkpoint: str) -> torch.nn.Module:
    if not torch.cuda.is_available():
        net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        net.load_state_dict(torch.load(checkpoint))
    return net

