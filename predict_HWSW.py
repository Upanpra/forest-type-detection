from functools import partial
from src.convnext import convnext_single_block_hyperspectral, convnext_single_block_hyperspectral_dropout, \
    convnext_minimal_hyperspectral, convnext_minimal_hyperspectral_dropout, \
    convnext_minimal_hyperspectral_final_layer_dropout
from src.minimal_cnn import min_cnn_dropout, min_cnn
from src.model import DroughtCNN
from src.predict.dataset import get_predict_loader
from src.predict.predictor import predictCNN
from src.functions import *
from src.data_loader import transforms_aug
import time
import torch
from src.predict.dataset import multi_image_data_loader

# directory of your training data
train_folder = [' data/train/vis',
                'data/train/chm']

means = []
stds = []
for i, root in enumerate(train_folder):
    if i == 1:
        modality = "chm"
    else:
        modality = "hyper"
    mean, std = get_mean_std(root, modality=modality)
    means.append(mean)
    stds.append(std)
mean = torch.cat(means, dim=0)
std = torch.cat(stds, dim=0)

print(mean)
print(std)
print("Alaska VIs model")

# set all the below parameters change based on which data type cnn is being running for
INPUT_CHANNELS = 48
INPUT_SIZE = [32, 32]
NUM_CLASSES = 3            # The number of output classes. In this case, from 1 to 4
NUM_EPOCHS = 100           # change to 200 # The number of times we loop over the whole dataset during training
BATCH_SIZE = 16           # Change to 16 or 32 The size of input data took for one iteration of an epoch
LEARNING_RATE = 1e-3          # The speed of convergence

#OUTPUT_FOLDER = "/mmfs1/gscratch/stf/upanpra/model_checkpoints/" # For Hyak

# Prediction for inference area
tiff_input_folder = ['data/unlabeled/vis',
                'data/unlabeled/chm'] #directory of your unlabeled crops such as vis, chm, dtm

tiff_output_folder = "data/prediction/hardwood-softwood-nonforest/" # directory of predicted files


# Define model name
model_name = "DroughtCNN"
model_path = "data/models/checkpoint/ "#path to saved model checkpoint

# use transforms_aug function from data loader to do augmentation
transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

all_models = {#"resnet18": partial(resnet18, input_channels=44),
            "DroughtCNN": partial(DroughtCNN, input_size=[INPUT_CHANNELS] + INPUT_SIZE, num_classes=NUM_CLASSES),
            "min_cnn": partial(min_cnn, num_classes=NUM_CLASSES),
            "min_cnn_dropout": partial(min_cnn_dropout, num_classes=NUM_CLASSES),
            "conv_next_single_block_model": partial(convnext_single_block_hyperspectral, num_classes=NUM_CLASSES),
            "dropout_conv_next_single_block_model": partial(convnext_single_block_hyperspectral_dropout, num_classes=NUM_CLASSES),
            "conv_next_model": partial(convnext_minimal_hyperspectral, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "dropout_conv_next_model": partial(convnext_minimal_hyperspectral_dropout, num_classes=NUM_CLASSES),
            "final_layer_dropout_conv_next_model": partial(convnext_minimal_hyperspectral_final_layer_dropout, num_classes=NUM_CLASSES)
              }

#predict_loader = get_predict_loader(tiff_input_folder, INPUT_SIZE, mean, std, BATCH_SIZE)

predict_loader = multi_image_data_loader(tiff_input_folder, INPUT_SIZE, BATCH_SIZE, mean, std)

# calling cnn functions 
net = all_models[model_name]()
net = load_model(net, model_path)
print(net)

start = time.time()
print(start/60)

predictCNN(net, predict_loader, tiff_output_folder)

end = time.time()
print(end)
print(f" Elapsed time: {(end-start)/60} minutes")
print(f" Elapsed time: {(end-start)/60/60} hours")
print(time.time())
