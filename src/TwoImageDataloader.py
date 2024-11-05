import typing as t
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from torchvision import datasets

from src.data_loader import transforms_aug, paired_image_transforms_aug
from src.functions import my_tiff_loader, set_hyper_no_data_values_as_zero, set_chm_no_data_values_as_zero


class PairedImageFolder(datasets.ImageFolder):

    def __init__(self, root_1, root_2, transform=None, target_transform=None, input_shape: t.Optional[t.Tuple] = (16, 16)):
        super().__init__(root_1, transform, target_transform)
        self.root_2 = root_2
        self.input_shape = input_shape

        # Check that the two directories have the same number of images
        if len(self) != sum(len(os.listdir(os.path.join(root_2, x))) for x in os.listdir(root_2)):
            raise RuntimeError(f"The two directories must contain the same number of images. {len(self)} != {sum(len(os.listdir(os.path.join(root_2, x))) for x in os.listdir(root_2))}")

    def __getitem__(self, index):
        path_1, label_idx = self.samples[index]
        class_name = self.classes[label_idx]
        basename = os.path.basename(path_1)
        path_2 = os.path.join(self.root_2, class_name, basename)
        #print(basename)
        # print(f"path_1: {path_1}")
        # print(f"path_2: {path_2}")

        img1 = my_tiff_loader(path_1)
        # this is work in code except geting NAN
        #img2 = my_tiff_loader(path_2)
        img2 = my_tiff_loader(path_2)

        #assert img1.shape[-1] == 44, f"{img1.shape}"
        #assert img1.shape[0] == 16, f"{img1.shape}"
        #assert img1.shape[1] == 16, f"{img1.shape}"
        img1_tensor = set_hyper_no_data_values_as_zero(torch.from_numpy(img1))
        img1_tensor = torch.transpose(img1_tensor, 0, 2)

        img2_tensor = set_chm_no_data_values_as_zero(torch.from_numpy(img2))
        img2_tensor = img2_tensor.unsqueeze(-1)
        img2_tensor = torch.transpose(img2_tensor, 0, 2)

        if self.input_shape is not None:
            img1_tensor = torch.nn.functional.interpolate(img1_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
            img2_tensor = torch.nn.functional.interpolate(img2_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
        try:
            #combined_torch = torch.cat((img1_tensor, img2_tensor.unsqueeze(-1)), dim=2)
            combined_torch = torch.cat((img1_tensor, img2_tensor), dim=0)
            #print(f"shape 1: {img1_tensor.shape}, shape 2: {img2_tensor.shape}")
            #print(f"path_1: {path_1}")
            #print(f"path_2: {path_2}")
            #print(f"Shape of combined torch, {combined_torch.shape}")
        except Exception as e:
            print(f"shape 1: {img1_tensor.shape}, shape 2: {img2_tensor.shape}")
            print(f"path_1: {path_1}, path_2: {path_2}")
            raise e
        #combined_torch = torch.permute(combined_torch, (2, 0, 1))
        if self.transform is not None:
            combined_torch = self.transform(combined_torch)
        return combined_torch, label_idx


        """
         if self.input_shape is not None:
            img1 = torch.nn.functional.interpolate(img1.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
            img2 = torch.nn.functional.interpolate(img2.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
        #combined_image = torch.cat((img1, img2.unsqueeze(-1)), dim=-1)
        combined_image = torch.stack((img1, img2), dim=-1)
        print(combined_image.shape)
        combined_image = torch.permute(combined_image, (2, 0, 1))
        #combined_image = np.concatenate((img_1, img_2), axis=0)
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        return combined_image, label_idx

        """


def paired_image_data_loader(train_folder: t.Tuple[str, str], test_folder: t.Tuple[str, str], input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = paired_image_transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = PairedImageFolder(train_folder[0], train_folder[1], transform = transform_train)#, transform=transform_train)# loader=my_tiff_loader)
    testset = PairedImageFolder(test_folder[0], test_folder[1], transform = transform_test)#, transform=transform_test)#, loader=my_tiff_loader)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader




