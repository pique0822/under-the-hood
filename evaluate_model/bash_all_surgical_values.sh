#!/bin/bash

module add openmind/singularity/3.0.3

for hidden_value in 0.1 1 10
do
	echo Current Hidden Value:: $hidden_value
	singularity exec -B /om/group/cpl -B /net/storage001.ib.cluster/om2/user/drmiguel/representational_analysis/under-the-hood /om2/user/jgauthie/singularity_images/deepo-cpu.simg python3 evaluate_target_word_test_lower_surprisal.py --data ../data/colorlessgreenRNNs --checkpoint /om/group/cpl/language-models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --prefixfile prefixes.txt --surprisalmode True --outf surgical_gradient_r2_decrease_FINAL_$hidden_value.txt --modify_cell True --surgical_difference $hidden_value
done
