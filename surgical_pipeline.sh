#!/bin/bash

SINGULARITY_IMG=/om2/user/jgauthie/singularity_images/deepo-cpu.simg
LM_PATH=/om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd -P )"

source /etc/profile.d/modules.sh
module add openmind/singularity/3.0.3

file_identifier="TEST"
# 1. Find most important coefficients
cd garden_path
echo Finding important coefficients
singularity exec -B $PROJECT_PATH $SINGULARITY_IMG ls ../data
singularity exec -B /om/group/cpl -B "$PROJECT_PATH" -B "$PROJECT_PATH/data" "$SINGULARITY_IMG" python3 avg_suprisal_vbd_decoder.py \
	--model_path "$LM_PATH" --training_cells reduced --cross_validation 10 --file_identifier $file_identifier \
	|| exit 1


cd ..
# 2. Perform Surgery
cd evaluate_model

for hidden_value in 0.1 1 10
do

	echo Current Hidden Value:: $hidden_value
	singularity exec -B /om/group/cpl -B "$PROJECT_PATH" "$SINGULARITY_IMG" python3 evaluate_target_word_test_lower_surprisal.py \
		--data ../data/colorlessgreenRNNs --checkpoint "$LM_PATH" \
		--prefixfile prefixes.txt --surprisalmode True --outf surgical_gradient_r2_decrease_${file_identifier}_${hidden_value}.txt \
		--modify_cell True --surgical_difference $hidden_value --file_identifier $file_identifier --gradient_type weight \
		|| exit 2
done

# 3. Plot the results

python3 generate_surprisal_plots_vbd.py --decrease_file_base surgical_gradient_r2_decrease_${file_identifier}_ \
	--decrease_file_unique_ids 0.1 1 10 --surgical_decrease True --file_title Testing_Mechanism \
	|| exit 3
