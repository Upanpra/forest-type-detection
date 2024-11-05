import torch
import torchvision.transforms as transforms 

def transforms_aug(input_size, mean, std):
    transform_train = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
             transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
    #          transforms.Lambda(lambda x: (x - mean) / std)
             transforms.Normalize(mean, std),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
             transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ration = (0.8,1.2))
            ])
            
            
    transform_test = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
             transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
    #          transforms.Lambda(lambda x: (x - mean) / std)
             transforms.Normalize(mean, std)
            ])
    
    return transform_train, transform_test
            