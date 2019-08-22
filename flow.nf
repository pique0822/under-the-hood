#!/usr/bin/env nextflow

import org.yaml.snakeyaml.Yaml

// baseDir as prepared by nextflow references a particular FS share. not good.
omBaseDir = "/om2/user/jgauthie/under-the-hood"

params.experiment_file = "${omBaseDir}/garden_path/experiment.yml"

def Map experiment_yaml = new Yaml().load((params.experiment_file as File).text)["experiment"] as Map

stimuli_file = new File((params.experiment_file as File).getParentFile(), experiment_yaml["stimuli"])

params.model_dir = "/om/group/cpl/language-models/colorlessgreenRNNs"
params.model_checkpoint_path = "${params.model_dir}/hidden650_batch128_dropout0.2_lr20.0.pt"
// params.model_data_path = "${params.model_dir}/data"
params.model_data_path = "${omBaseDir}/data/colorlessgreenRNNs"

surgery_coefs = Channel.from(experiment_yaml["surgery"]["coefficients"])

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
    file(stimuli_csv) from Channel.from(stimuli_file)
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
    set file("best_mse_coefs_decoder.pkl"), file("best_r2_coefs_decoder.pkl"), file("significant_coefs_decoder.pkl") into base_decoder_ch

    script:
    """
#!/usr/bin/env bash
python3 ${omBaseDir}/garden_path/avg_suprisal_vbd_decoder.py \
    ${params.experiment_file} \
    ${surprisals_file} \
    --model_path ${params.model_checkpoint_path} \
    --data_path ${params.model_data_path} \
    --file_identifier ${file_prefix}
    """
}


process doSurgery {
    label "om_deepo"

    input:
    set val(surgery_coef), file("best_mse_coefs_decoder.pkl"), file("best_r2_coefs_decoder.pkl"), file("significant_coefs_decoder.pkl") from surgery_coefs.combine(base_decoder_ch)
    file(prefixes_file) from prefixes_ch

    """
#!/usr/bin/env bash
python3 ${omBaseDir}/evaluate_model/evaluate_target_word_test_lower_surprisal.py \
    --data ${params.model_data_path} \
    --checkpoint ${params.model_checkpoint_path} \
    --prefixfile ${prefixes_file} \
    --surprisalmode True \
    --outf surgical_gradient_r2_decrease_${file_prefix}_${surgery_coef}.txt \
    --modify_cell True \
    --surgical_difference ${surgery_coef} \
    --file_identifier ${file_prefix} \
    --gradient_type weight
    """
}
