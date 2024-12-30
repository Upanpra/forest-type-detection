# Synthesizing Field Plot and Airborne Remote Sensing Data to Enhance National Forest Inventory Mapping in the Boreal Forest of Interior Alaska
This is a code repository for our paper https://doi.org/10.1016/j.srs.2024.100192 .

## Environment
To run our code, create a python virtual environment using the included `requirements.txt`. For training to run quickly, you will need a machine with a GPU.

## Training CNN model
To train our models use the included training scripts where you need to change the path to the dataset.  

## Training ML model
To train our models use the included ipython notebook. The uploaded script runs ml models for forest type classification.

## Running Inference
Our tranined checkpoints are available inside data/model_checkpoints and an example of how to run the model on unlabeled data is shown in predict.py

## Data
The data for this project is not available due to data privacy of United States Forest Services.

## License
The code is released under the included MIT [license](LICENSE) 
