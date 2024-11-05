# import files for functions
import argparse
from functools import partial
from sklearn.metrics import confusion_matrix

import torch
import torchvision
import os

from src.MultipleImageDataloader import multi_image_data_loader
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


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Process data from input file and save to output file.")

    # Add command-line arguments
    parser.add_argument("--train_folder", type=str, required=True, help="Path(s) to the input train folder.", nargs="+")
    parser.add_argument("--test_folder", type=str, required=True, help="Path(s) to the input test folder.", nargs="+")
    parser.add_argument("--input_channels", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True, default=100, help="20 50")
    parser.add_argument("--model", type=str, required=False, default="DroughtCNN", help="conv_next_model min_cnn final_layer_dropout_conv_next_model")
    parser.add_argument("--pretrained_checkpoint", type=str, required=False, default=None, help="location of checkpoint- full path")
    parser.add_argument("--vars", type=str, required=False, default="vis-chm", help="vis-chm vis-chm-dtm")

    parser.add_argument("--modality", type=str, required=False, default="hyper", help="chm or hyper")

    parser.add_argument("--input_size", nargs='+', required=False, default= [16, 16], type=int, help="List of integers.")

    # Parse the command-line arguments
    args = parser.parse_args()
    assert len(args.input_size) == 2, args.input_size
    return args


def main():
    args = parse_cli_arguments()
    print(f"Running script with arguments: {args}")

    # new labelled data cross checked
    # train_folder = "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/forest_vs_nonforest/chm/train"
    # test_folder = "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/forest_vs_nonforest/chm/test"
    train_folder = args.train_folder
    test_folder = args.test_folder
    pretrained_checkpoint = args.pretrained_checkpoint

    if len(train_folder) == 1:
        assert len(test_folder) == 1
        mean, std = get_mean_std(train_folder[0], modality=args.modality)
    else:
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


    # set all the below parameters change based on which data type cnn is being running for
    INPUT_CHANNELS = args.input_channels
    INPUT_SIZE = args.input_size
    NUM_CLASSES = args.num_classes           # The number of output classes. In this case, from 1 to 4
    NUM_EPOCHS = args.epochs           # change to 200 # The number of times we loop over the whole dataset during training
    BATCH_SIZE = 16           # Change to 16 or 32 The size of input data took for one iteration of an epoch
    LEARNING_RATE = 1e-3          # The speed of convergence

    BASE_MODEL_OUTPUT_FOLDER = "/mmfs1/gscratch/stf/upanpra/2024_AK_CKPT"# For Hyak

    # Define model name
    model_name = args.model #"DroughtCNN" #resnet18

    channel_indices = None  # Set to None to not select any channels
    if channel_indices is not None:
        INPUT_CHANNELS = len(channel_indices)

    timestamp = time.time()
    checkpoint_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"model-{args.vars}-{model_name}-{timestamp}.pt")
    bestmodel_path = os.path.join(BASE_MODEL_OUTPUT_FOLDER, f"best-test-model-{args.vars}-{model_name}-{timestamp}.pt")
    print(f"Saving model to: {bestmodel_path}")

    # use transforms_aug function from data loader to do augmentation
    transform_train, transform_test = transforms_aug(INPUT_SIZE, mean, std)

    # use data_loader function from data loader
    if len(train_folder) == 1:
        assert len(test_folder) == 1
        train_loader, test_loader = data_loader(train_folder[0], test_folder[0], INPUT_SIZE, BATCH_SIZE, mean, std)
    else:
        train_loader, test_loader = multi_image_data_loader(train_folder, test_folder, INPUT_SIZE, BATCH_SIZE, mean, std)

    all_models = {#"resnet18": partial(resnet18, input_channels=44),
                "DroughtCNN": partial(DroughtCNN, input_size=[INPUT_CHANNELS] + INPUT_SIZE, num_classes=NUM_CLASSES),
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

    # load model checkpoint if pretrained:
    if pretrained_checkpoint is not None:
        assert isinstance(pretrained_checkpoint, str), pretrained_checkpoint
        state_dict = torch.load(pretrained_checkpoint)
        net.load_state_dict(state_dict=state_dict)
        print(f"Loaded state dict weights from {pretrained_checkpoint} into {model_name}")

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


if __name__ == "__main__":
    main()
