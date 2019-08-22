#!/usr/bin/env nextflow

// baseDir as prepared by nextflow references a particular FS share. not good.
omBaseDir = "/om2/user/jgauthie/under-the-hood"

params.experiment_file = "${omBaseDir}/garden_path/experiment.yml"
params.stimuli_file = "${omBaseDir}/garden_path/data/verb-ambiguity-with-intervening-phrase.csv"
// TODO auto generate prefixes file :/
// longer-term : just remove this required input
params.prefixes_file = "${omBaseDir}/evaluate_model/prefixes.txt"

params.model_dir = "/om/group/cpl/language-models/colorlessgreenRNNs"
params.model_checkpoint_path = "${params.model_dir}/hidden650_batch128_dropout0.2_lr20.0.pt"
// params.model_data_path = "${params.model_dir}/data"
params.model_data_path = "${omBaseDir}/data/colorlessgreenRNNs"

params.decoder_cv_folds = 10

params.surgery_coefs = "0.1,1,10"
surgery_coefs = Channel.from(params.surgery_coefs.tokenize(","))

//////////

// prefix for tmp output files. no reason to change this
file_prefix = "decoder"


// Given an experiment spec, generate prefixes for LM evaluation.
process generatePrefixes {
    output:
    file "prefixes.txt" into prefixes_ch

    script:
    """
#!/usr/bin/env bash
python3 ${omBaseDir}/evaluate_model/generate_prefixes.py \
    ${params.experiment_file} \
    --outf prefixes.txt
    """
}


process getSurprisals {
    label "om_deepo"

    input:
    file(stimuli_csv) from Channel.from(params.stimuli_file)
    file(prefixes_file) from prefixes_ch

    output:
    file "surprisals.txt" into surprisals_ch

    script:
    """
#!/usr/bin/env bash
python3 ${omBaseDir}/evaluate_model/evaluate_target_word_test.py \
    --data ${params.model_data_path} \
    --checkpoint ${params.model_checkpoint_path} \
    --prefixfile ${prefixes_file} \
    --surprisalmode True \
    --outf surprisals.txt
    """
}

process learnBaseDecoder {
    label "om_deepo"

    input:
    file(surprisals_file) from surprisals_ch

    output:
    file("*.pkl") into base_decoder_ch

    script:
    """
#!/usr/bin/env bash
python3 ${omBaseDir}/garden_path/avg_suprisal_vbd_decoder.py \
    ${params.experiment_file} \
    ${surprisals_file} \
    --model_path ${params.model_checkpoint_path} \
    --data_path ${params.model_data_path} \
    --training_cells reduced \
    --cross_validation ${params.decoder_cv_folds} \
    --file_identifier ${file_prefix}
    """
}

process doSurgery {
    label "om_deepo"

    input:
    set val(surgery_coef), file(decoder_files) from surgery_coefs.combine(base_decoder_ch)

    """
#!/usr/bin/env bash
echo hello
    """
}
