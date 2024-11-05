#!/bin/bash
#SBATCH --job-name=VCD
#SBATCH -p gpu-2080ti
#SBATCH -A stf
#SBATCH -c 1


#SBATCH --partition=gpu-2080ti
#SBATCH --gpus=1
#SBATCH --account=stf


#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G

#SBATCH --chdir=.

## module load /mmfs1/home/upanpra/miniconda3/
source activate "/mmfs1/home/upanpra/miniconda3/envs/deeplearningCuda"


# Set the additional Python path you want to include
additional_path="/mmfs1/gscratch/stf/upanpra/"

# Check if PYTHONPATH already exists, and append the additional path if it does
if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH="$additional_path"
else
  export PYTHONPATH="$additional_path:$PYTHONPATH"
fi

## module load contrib/keras
python /mmfs1/gscratch/stf/upanpra/AK_train_scripts/train_script_parameterized.py --train_folder "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/vis/train" "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/chm/train" "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/dtm/train" \
	--test_folder "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/vis/test" "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/chm/test" "/mmfs1/gscratch/stf/upanpra/AK_paper_data/16crops/hardwood_vs_softwood/dtm/test" \
	--input_channels 46 --num_classes 3 --input_size 16 16



