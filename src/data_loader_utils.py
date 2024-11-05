# Building sampler to attempt to balance classes during training
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def balance_classes():
    class_sample_counts = train_target_class_index.value_counts()
    # print(class_sample_counts)
    weights = {class_idx: 1.0 / class_count for class_idx, class_count in class_sample_counts.items()}
    # print(weights)
    samples_weight = np.array([weights[t] for t in train_target_class_index])
    # print(train_target_class_index[:10])
    # print(samples_weight[:10])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler

    
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight




def create_balanced_sampler(dataset):
    # Get the number of classes in the dataset
    num_classes = len(dataset.classes)

    # Calculate the weight for each class to balance the dataset
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[label] += 1

    class_weights = [1.0 / count for count in class_counts]

    # Create a list of weights for each sample in the dataset
    weights = [class_weights[label] for _, label in dataset.samples]

    # Create a WeightedRandomSampler with the calculated weights
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return sampler

# # Example usage:
# # Specify the path to your dataset and define the transformation
# data_path = "/path/to/your/dataset"
# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#
# # Create the ImageFolder dataset
# dataset = datasets.ImageFolder(root=data_path, transform=transform)
#
# # Create the balanced sampler
# balanced_sampler = create_balanced_sampler(dataset)
#
# # Create a DataLoader using the balanced sampler
# dataloader = DataLoader(dataset, batch_size=32, sampler=balanced_sampler)
#
# # Now, you can use the dataloader for training your model
