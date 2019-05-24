for hidden_value in 10
do
	echo Current Hidden Value:: $hidden_value
	python3 evaluate_target_word_test_lower_surprisal.py --data ../data/colorlessgreenRNNs --checkpoint ../models/colorlessgreenRNNs/hidden650_batch128_dropout0.2_lr20.0.pt --prefixfile prefixes.txt --surprisalmode True --outf surgical_gradient_r2_decrease_FINAL_$hidden_value.txt --modify_cell True --surgical_difference $hidden_value
done