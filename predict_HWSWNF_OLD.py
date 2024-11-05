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
# from src.resnet import resnet18

#train_folder = "/mmfs1/gscratch/stf/upanpra/AK_data/Vis16/train_test/train_Vis16"
#test_folder = "/mmfs1/gscratch/stf/upanpra/AK_data/Vis16//train_test/test_Vis16"


train_folder = "E:/AguAlaska/Vis16m/train_test/train_Vis16"
test_folder = "E:/AguAlaska/Vis16m/train_test/test_Vis16"



mean, std = get_mean_std(train_folder, test_folder)
print(mean)
print(std)
print("Alaska VIs model")

# set all the below parameters change based on which data type cnn is being running for
INPUT_CHANNELS = 44
INPUT_SIZE = [16, 16]
NUM_CLASSES = 3            # The number of output classes. In this case, from 1 to 4
NUM_EPOCHS = 100           # change to 200 # The number of times we loop over the whole dataset during training
BATCH_SIZE = 16           # Change to 16 or 32 The size of input data took for one iteration of an epoch
LEARNING_RATE = 1e-4          # The speed of convergence

#OUTPUT_FOLDER = "/mmfs1/gscratch/stf/upanpra/model_checkpoints/" # For Hyak
#tiff_input_folder = "E:/AguAlaska/inference"
#tiff_output_folder = "E:/AguAlaska/predictions"

# prediction for test set
tiff_input_folder = "E:/AguAlaska/secondstrip_inf"
tiff_output_folder = "E:/AguAlaska/secondstrip_inf/secondstrip_prediction"

# Define model name
model_name = "conv_next_model" #resnet18
model_path = "G:/repositories/Alaska-Project/model_checkpoint/model-conv_next_model-1669960892.788017_81.pt"

# use transforms_aug function from data loader to do augmentation
transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

all_models = {#"resnet18": partial(resnet18, input_channels=44),
            "DroughtCNN": partial(DroughtCNN, input_size=[INPUT_CHANNELS] + INPUT_SIZE),
            "min_cnn": partial(min_cnn, num_classes=3),
            "min_cnn_dropout": partial(min_cnn_dropout, num_classes=3),
            "conv_next_single_block_model": partial(convnext_single_block_hyperspectral, num_classes=3),
            "dropout_conv_next_single_block_model": partial(convnext_single_block_hyperspectral_dropout, num_classes=3),
            "conv_next_model": partial(convnext_minimal_hyperspectral, num_classes=3, input_channels=INPUT_CHANNELS),
            "dropout_conv_next_model": partial(convnext_minimal_hyperspectral_dropout, num_classes=3),
            "final_layer_dropout_conv_next_model": partial(convnext_minimal_hyperspectral_final_layer_dropout, num_classes=3)
              }

predict_loader = get_predict_loader(tiff_input_folder, INPUT_SIZE, mean, std, BATCH_SIZE)

# calling cnn functions either resnet or droughtCNN
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
