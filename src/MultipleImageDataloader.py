import typing as t
import os
import warnings

import numpy as np
import torch
from torchvision import datasets

from src.data_loader import paired_image_transforms_aug
from src.data_loader_utils import create_balanced_sampler
from src.functions import my_tiff_loader, set_hyper_no_data_values_as_zero, set_chm_no_data_values_as_zero


class MultiImageFolder(datasets.ImageFolder):

    def __init__(self, roots: t.List[str], transform=None, target_transform=None, input_shape: t.Optional[t.Tuple] = (16, 16), assert_all_roots_have_same_number_files: bool = False):
        assert isinstance(roots, t.Sequence), f"{type(roots)} must be list"
        # pass root list name roots
        super().__init__(roots[0], transform, target_transform)

        self.roots = roots
        self.input_shape = input_shape
        self._n_failed_in_process = 0
        self._max_consecutive_failures = 100

        # Check that all directories have the same number of images
        n_files_same = all((sum(len(os.listdir(os.path.join(x, y))) for y in os.listdir(x)) == len(self)) for x in self.roots)
        if assert_all_roots_have_same_number_files:
            assert n_files_same, "All roots must contain the same number of files"
        else:
            if not n_files_same:
                n_files_per = {x: sum(len(os.listdir(os.path.join(x, y))) for y in os.listdir(x)) for x in self.roots}
                warnings.warn(f"roots have different number of files: {n_files_per}")

    def __getitem__(self, index):
        try:
            path_1, label_idx = self.samples[index]
            assert self.roots[0] in path_1, f"{self.roots[0]} should be in {path_1}"
            class_name = self.classes[label_idx]
            basename = os.path.basename(path_1)

            images = []
            paths = []  # only used on exception
            for i, root in enumerate(self.roots):
                path = os.path.join(root, class_name, basename)
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
                    assert len(img_tensor.shape) == 3, f"Loaded tensors should be rank 2 or 3 but found {len(img_tensor.shape)} for {path}"
                img_tensor = torch.transpose(img_tensor, 0, 2)

                if self.input_shape is not None:
                    img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
                images.append(img_tensor)
            try:
                combined_torch = torch.cat(images, dim=0)
            except Exception as e:
                print(f"shapes: {[x.shape for x in images]}")
                print(f"paths: {paths}")
                raise e
            if self.transform is not None:
                combined_torch = self.transform(combined_torch)
            self._n_failed_in_process = 0
        except Exception as e:
            if self._n_failed_in_process >= self._max_consecutive_failures:
                # this ensures we don't get stuck in an infinite recursion if there's a bug and nothing loads
                print(f"Too many consecutive failures {self._n_failed_in_process}")
                raise e
            print(f"encountered exception loading {path_1}. Will try to load another")
            print(e)
            self._n_failed_in_process += 1
            idx = np.random.randint(0, len(self))
            return self[idx]
        return combined_torch, label_idx



class PairedImageFolder(datasets.ImageFolder):

    def __init__(self, root_1, root_2, root_3, transform=None, target_transform=None, input_shape: t.Optional[t.Tuple] = (16, 16)):
        # pass root list
        super().__init__(root_1, transform, target_transform)
        self.root_2 = root_2
        self.root_3 = root_3
        self.input_shape = input_shape

        # Check that the two directories have the same number of images
        if len(self) != sum(len(os.listdir(os.path.join(root_2, x))) for x in os.listdir(root_2)):
            raise RuntimeError(f"The two directories must contain the same number of images. {len(self)} != {sum(len(os.listdir(os.path.join(root_2, x))) for x in os.listdir(root_2))}")

    def __getitem__(self, index):
        path_1, label_idx = self.samples[index]
        class_name = self.classes[label_idx]
        basename = os.path.basename(path_1)
        # images = []
        #for i, root in enumerate(self.roots):
        path_2 = os.path.join(self.root_2, class_name, basename)
        path_3 = os.path.join(self.root_3, class_name, basename)

        img1 = my_tiff_loader(path_1)
        img2 = my_tiff_loader(path_2)
        img3 = my_tiff_loader(path_3)



        #assert img1.shape[-1] == 44, f"{img1.shape}"
        #assert img1.shape[0] == 16, f"{img1.shape}"
        #assert img1.shape[1] == 16, f"{img1.shape}"


        img1_tensor = set_hyper_no_data_values_as_zero(torch.from_numpy(img1))
        img1_tensor = torch.transpose(img1_tensor, 0, 2)

        img2_tensor = set_chm_no_data_values_as_zero(torch.from_numpy(img2))
        img2_tensor = img2_tensor.unsqueeze(-1)
        img2_tensor = torch.transpose(img2_tensor, 0, 2)

        img3_tensor = set_hyper_no_data_values_as_zero(torch.from_numpy(img3))
        img3_tensor = img3_tensor.unsqueeze(-1)
        img3_tensor = torch.transpose(img3_tensor, 0, 2)


        if self.input_shape is not None:
            img1_tensor = torch.nn.functional.interpolate(img1_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
            img2_tensor = torch.nn.functional.interpolate(img2_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
            img3_tensor = torch.nn.functional.interpolate(img3_tensor.unsqueeze(0), tuple(self.input_shape)).squeeze(0)
        # images.append(img)
        #for loop ends here
        try:
            #combined_torch = torch.cat((img1_tensor, img2_tensor.unsqueeze(-1)), dim=2)
            combined_torch = torch.cat((img1_tensor, img2_tensor, img3_tensor), dim=0)
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



def multi_image_data_loader(train_folders: t.List[str], test_folders: t.List[str], input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = paired_image_transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = MultiImageFolder(train_folders, transform = transform_train, input_shape=input_size)
    testset = MultiImageFolder(test_folders, transform = transform_test, input_shape=input_size)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def multi_image_data_loader_balanced(train_folders: t.List[str], test_folders: t.List[str], input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = paired_image_transforms_aug(input_size, mean, std, channel_indices=channel_indices)

    trainset = MultiImageFolder(train_folders, transform = transform_train)#, transform=transform_train)# loader=my_tiff_loader)
    balance_sampler = create_balanced_sampler(trainset)
    testset = MultiImageFolder(test_folders, transform = transform_test)#, transform=transform_test)#, loader=my_tiff_loader)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, sampler= balance_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader




