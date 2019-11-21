#!/usr/bin/env nextflow

import org.yaml.snakeyaml.Yaml

// baseDir as prepared by nextflow references a particular FS share. not good.
omBaseDir = "/om/user/jgauthie/under-the-hood"

params.experiment_file = "${omBaseDir}/experiments/base.yml"
def Map experiment_yaml = new Yaml().load((params.experiment_file as File).text)["experiment"] as Map
params.experiment_name = experiment_yaml.name
params.experiment_clean_name = experiment_yaml.clean_name

stimuli_file = new File((params.experiment_file as File).getParentFile(), experiment_yaml["stimuli"])

params.model_dir = "/om/group/cpl/language-models/colorlessgreenRNNs"
params.model_checkpoint_path = "${params.model_dir}/hidden650_batch128_dropout0.2_lr20.0.pt"
// params.model_data_path = "${params.model_dir}/data"
params.model_data_path = "${omBaseDir}/data/colorlessgreenRNNs"

surgery_coefs = Channel.from(experiment_yaml["surgery"]["coefficients"])

params.output_dir = "output/${params.experiment_clean_name}"

//////////

// Given an experiment spec, generate prefixes for LM evaluation.
process generatePrefixes {
    label "local"

    output:
    set file("prefixes.txt"), file("extract_idxs.txt") into prefixes_ch

    script:
    """
#!/usr/bin/env bash
export PYTHONPATH="${omBaseDir}:\$PYTHONPATH"
python3 ${omBaseDir}/evaluate_model/generate_prefixes.py \
    ${params.experiment_file} \
    --outf prefixes.txt \
    --extract_idx_outf extract_idxs.txt
    """
}


process getSurprisals {
    label "om_deepo"

    input:
    set file(prefixes_file), file(extract_idxs_file) from prefixes_ch

    output:
    file "surprisals.txt" into surprisals_ch

    script:
    """
#!/usr/bin/env bash
export PYTHONPATH="${omBaseDir}:\$PYTHONPATH"
python3 ${omBaseDir}/evaluate_model/evaluate_target_word_test.py \
    --data ${params.model_data_path} \
    --checkpoint ${params.model_checkpoint_path} \
    --prefixfile ${prefixes_file} \
    --surprisalmode True \
    --outf surprisals.txt
    """
}

surprisals_ch.into { surprisals_for_decoder_ch; surprisals_for_plot_ch }

process learnBaseDecoder {
    label "om_deepo"

    input:
    file(surprisals_file) from surprisals_for_decoder_ch

    output:
    set file("best_mse_coefs.pkl"), file("best_r2_coefs.pkl"), \
        file("significant_coefs.pkl") \
        into base_decoder_ch

    script:
    """
#!/usr/bin/env bash
export PYTHONPATH="${omBaseDir}:\$PYTHONPATH"
python3 ${omBaseDir}/garden_path/avg_suprisal_vbd_decoder.py \
    ${params.experiment_file} \
    ${surprisals_file} \
    --model_path ${params.model_checkpoint_path} \
    --data_path ${params.model_data_path}
    """
}


process doSurgery {
    label "om_deepo"

    input:
    set val(surgery_coef), \
        file("best_mse_coefs.pkl"), file("best_r2_coefs.pkl"), file("significant_coefs.pkl"), \
        file(prefixes_file), file(extract_idxs_file) \
        from surgery_coefs.combine(base_decoder_ch).combine(prefixes_ch)

    output:
    file("surgery_out_*.pkl") into surgery_ch

    """
#!/usr/bin/env bash
export PYTHONPATH="${omBaseDir}:\$PYTHONPATH"
python3 ${omBaseDir}/evaluate_model/evaluate_target_word_test.py \
    --data ${params.model_data_path} \
    --checkpoint ${params.model_checkpoint_path} \
    --prefixfile ${prefixes_file} \
    --surprisalmode True \
    --do_surgery True \
    --surgery_idx_file ${extract_idxs_file} \
    --surgery_coef_file best_r2_coefs.pkl \
    --surgery_scale ${surgery_coef} \
    --surgery_outf surgery_out_${surgery_coef}.pkl \
    --gradient_type weight
    """
}


process renderPlots {
    label "local"
    publishDir params.output_dir

    input:
    file("surprisals.txt") from surprisals_for_plot_ch
    file("*.pkl") from surgery_ch.collect()

    output:
    file "*.png"
    file "graph_data.csv"

    script:
    """
#!/usr/bin/env bash
export PYTHONPATH="${omBaseDir}:\$PYTHONPATH"
python3 ${omBaseDir}/evaluate_model/generate_surprisal_plots_vbd.py \
    ${params.experiment_file} surprisals.txt \
    --surgery_files *.pkl
    """
}
