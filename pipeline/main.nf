#!/usr/bin/env nextflow

nextflow.enable.dsl = 2
params.outdir = "results"
params.ntrials = 200
params.infiles = false
params.task = "regression"

def helpMessage() {
    log.info "# autoselect"
}


process model {

    label 'python'
    label 'cpu_high'
    label 'ram_high'
    label 'time_medium'

    tag { name }

    input:
    tuple val(name), path(infile), val(model)
    val task_
    val ntrials

    output:
    tuple val(name), path(infile), path("${infile.simpleName}_outdir"), val(model)

    script:
    """
    tar --dereference --strip-components=1 -zxf "${infile}"

    mkdir "${infile.simpleName}_outdir"

    selectml \
      optimise \
      "${task_}" \
      "${model}" \
      "${infile.simpleName}/markers_train.tsv" \
      "${infile.simpleName}/phenos_train.tsv" \
      -r response \
      -n name \
      -o "${infile.simpleName}_outdir/${task_}_${model}_results.tsv" \
      --full "${infile.simpleName}_outdir/${task_}_${model}_full_results.tsv" \
      --pickle "${infile.simpleName}_outdir/${task_}_${model}_pickle.pkl" \
      --best "${infile.simpleName}_outdir/${task_}_${model}_best.json" \
      --ntasks 1 \
      --cpu "${task.cpus}" \
      --ntrials ${ntrials} \
      --timeout 6

    rm -rf -- "${infile.simpleName}"
    """
}


process eval {

    label 'python'
    label 'cpu_high'
    label 'ram_high'
    label 'time_medium'

    publishDir { params.outdir }

    tag { name }

    input:
    tuple val(name), path(infile), path("optimised"), val(model)
    val task_

    output:
    path "${infile.simpleName}_${task_}_${model}"

    script:
    output_dir = "${infile.simpleName}_${task_}_${model}"
    """
    tar --dereference --strip-components=1 -zxf "${infile}"

    cat ${infile.simpleName}/markers_train.tsv > ./markers.tsv
    cat ${infile.simpleName}/markers_test.tsv | tail -n+2 >> ./markers.tsv

    cat ${infile.simpleName}/phenos_train.tsv \
    | awk -F '\t' 'BEGIN {OFS="\t"} {print \$1, \$2, \$3, \$4, \$5, "train"}'\
    > ./phenos.tsv

    cat ${infile.simpleName}/phenos_test.tsv \
    | tail -n+2 \
    | awk -F '\t' '
        BEGIN {OFS="\t"}
        \$2==2 {print \$1, \$2, \$3, \$4, \$5, "pop2"}
        \$2==4 {print \$1, \$2, \$3, \$4, \$5, "backcross"}
        \$2==3 {print \$1, \$2, \$3, \$4, \$5, "random"}
    ' >> ./phenos.tsv

    mkdir "${output_dir}"
    cp -L optimised/${task_}_${model}_results.tsv "${output_dir}"
    cp -L optimised/${task_}_${model}_full_results.tsv "${output_dir}"
    # cp -L optimised/${task_}_${model}_importance.tsv "${output_dir}"
    cp -L optimised/${task_}_${model}_best.json "${output_dir}"

    selectml predict \
      -t train \
      -r response \
      -n name \
      -o "${output_dir}/${task_}_${model}_predictions.tsv" \
      -s "${output_dir}/${task_}_${model}_stats.tsv" \
      --outmodel "${output_dir}/${task_}_${model}_model.pkl" \
      --cpu 1 \
      regression \
      sgd \
      optimised/${task_}_${model}_best.json \
      "markers.tsv" \
      "phenos.tsv"

    rm markers.tsv phenos.tsv
    rm -rf -- "${infile.simpleName}"
    """
}


workflow {

    main:

    if ( params.help ) {
        helpMessage()
        exit 0
    }

    if ( ! params.infiles ) {
        println "Nup"
        exit 1
    }

    infiles = Channel.fromPath( params.infiles, checkIfExists: true).map { f -> [f.simpleName, f] }

    models = Channel.of( "sgd" )
    // models = Channel.of( "sgd", "bglr", "knn", "xgb" )

    combined = infiles.combine(models)
    results = model(combined, params.task, params.ntrials)
    stats = eval(results, params.task)
}
