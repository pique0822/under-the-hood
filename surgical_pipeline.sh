#!/bin/bash
module add openmind/singularity/3.0.3

file_identifier = "TEST"
# 1. Find most important coefficients
cd garden_path/

echo Finding important coefficients
singularity exec -B /om/group/cpl -B /net/storage001.ib.cluster/om2/user/drmiguel/representational_analysis/under-the-hood /om2/user/jgauthie/singularity_images/deepo-cpu.simg python3 avg_surprisal_vbd_decoder.py --model_path /om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --training_cells reduced --cross_validation 10 --file_identifier $file_identifier


cd ..
# 2. Perform Surgery
cd evaluate_model

for hidden_value in 0.1 1 10
do

	echo Current Hidden Value:: $hidden_value
	singularity exec -B /om/group/cpl -B /net/storage001.ib.cluster/om2/user/drmiguel/representational_analysis/under-the-hood /om2/user/jgauthie/singularity_images/deepo-cpu.simg python3 evaluate_target_word_test_lower_surprisal.py --data ../data/colorlessgreenRNNs --checkpoint /om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --prefixfile prefixes.txt --surprisalmode True --outf surgical_gradient_r2_decrease_${file_identifier}_${hidden_value}.txt --modify_cell True --surgical_difference $hidden_value --file_identifier $file_identifier --gradient_type weight
done

# 3. Plot the results

python3 generate_surprisal_plots_vbd.py --decrease_file_base surgical_gradient_r2_decrease_${file_identifier}_ --decrease_file_unique_ids 0.1 1 10 --surgical_decrease True --file_title Testing_Mechanism
