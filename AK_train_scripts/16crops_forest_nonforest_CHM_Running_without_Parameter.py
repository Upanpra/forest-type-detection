# import files for functions
from functools import partial
from sklearn.metrics import confusion_matrix

import torchvision
import os

from src.convnext import convnext_single_block_hyperspectral, convnext_single_block_hyperspectral_dropout, \
    convnext_minimal_hyperspectral, convnext_minimal_hyperspectral_dropout, \
    convnext_minimal_hyperspectral_final_layer_dropout
from src.minimal_cnn import min_cnn_dropout, min_cnn
from src.model_old import DroughtCNN
from src.trainer import trainCNN
from src.accuracy import get_Ytrue_YPredict
from src.functions import *
from src.data_loader import transforms_aug, data_loader
import time
# new labelled data cross checked
#from src.resnet import resnet18

# new labelled data cross checked
train_folder = "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/forest_vs_nonforest/chm/train"
test_folder = "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/forest_vs_nonforest/chm/test"

mean, std = get_mean_std(train_folder, test_folder, modality = "chm")

print(mean)
print(std)
print("Alaska CHM with 16x16 Crops")

# set all the below parameters change based on which data type cnn is being running for
INPUT_CHANNELS = 1
INPUT_SIZE = [16, 16]
NUM_CLASSES = 2           # The number of output classes. In this case, from 1 to 4
NUM_EPOCHS = 100           # change to 200 # The number of times we loop over the whole dataset during training
BATCH_SIZE = 16           # Change to 16 or 32 The size of input data took for one iteration of an epoch
LEARNING_RATE = 1e-3          # The speed of convergence

BASE_MODEL_OUTPUT_FOLDER = "/mmfs1/gscratch/stf/upanpra/model_checkpointsAK" # For Hyak

# Define model name
model_name = "conv_next_model" #resnet18

channel_indices = None  # Set to None to not select any channels
if channel_indices is not None:
    INPUT_CHANNELS = len(channel_indices)

timestamp = time.time()
checkpoint_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"model-VIs-{model_name}-{timestamp}.pt")
bestmodel_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"best-test-model-Vis{model_name}-{timestamp}.pt")
print(f"Saving model to: {bestmodel_path}")

# use transforms_aug function from data loader to do augmentation
transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

# use data_loader function from data loader
train_loader, test_loader = data_loader(train_folder, test_folder, INPUT_SIZE, BATCH_SIZE, mean, std)

all_models = {#"resnet18": partial(resnet18, input_channels=44),
            "DroughtCNN": partial(DroughtCNN, input_size=[INPUT_CHANNELS] + INPUT_SIZE),
            "min_cnn": partial(min_cnn, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "min_cnn_dropout": partial(min_cnn_dropout, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "conv_next_single_block_model": partial(convnext_single_block_hyperspectral, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "dropout_conv_next_single_block_model": partial(convnext_single_block_hyperspectral_dropout, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "conv_next_model": partial(convnext_minimal_hyperspectral, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "dropout_conv_next_model": partial(convnext_minimal_hyperspectral_dropout, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS),
            "final_layer_dropout_conv_next_model": partial(convnext_minimal_hyperspectral_final_layer_dropout, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
              }

# calling cnn functions either resnet or droughtCNN
net = all_models[model_name]()
print(net)

start = time.time()
print(start/60)

net, train_history, test_history = trainCNN(net, train_loader, test_loader,
                                        num_epochs=NUM_EPOCHS,
                                        learning_rate=LEARNING_RATE,
                                        compute_accs=True)

end = time.time()
print(end)
print(f" Elapsed time: {(end-start)/60} minutes")
print(f" Elapsed time: {(end-start)/60/60} hours")
print(time.time())

torch.save(net.state_dict(), checkpoint_path)
print("Saved model to", checkpoint_path)

# calling for accuracy
y_true, y_predict = get_Ytrue_YPredict(net, test_loader )

confusion_results = confusion_matrix(y_true, y_predict)
print(confusion_results)

confusion_results1 = confusion_matrix(y_true, y_predict, normalize="true")
print(confusion_results1)

confusion_results2 = confusion_matrix(y_true, y_predict, normalize="pred")
print(confusion_results2)
