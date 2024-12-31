#!/bin/bash
#SBATCH --job-name=chmFNF
#SBATCH -p gpu-2080ti
#SBATCH -A stf
#SBATCH -c 1


#SBATCH --partition=gpu-2080ti
#SBATCH --gpus=1
#SBATCH --account=stf


#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G

#SBATCH --chdir=.

## module load: path to your virtual environment
source activate "/miniconda3/envs/deeplearningCuda"


# Set the additional Python path you want to include
additional_path="/mmfs1/gscratch/stf/username/"

# Check if PYTHONPATH already exists, and append the additional path if it does
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$additional_path"
else
  export PYTHONPATH="$additional_path:$PYTHONPATH"
fi

## module load contrib/keras
python /forest-type-detection/train_script_example.py --train_folder "location of train folder of vis" "location of train folder of chm" \
	--test_folder "location of test folder of vis" "location of test folder of chm" \
	--input_channels 45  --num_classes 2 --input_size 16 16


