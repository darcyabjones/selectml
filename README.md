# selectml

A package for performing and optimising genomic prediction using machine learning.
Genomic prediction attempts to predict a phenotypic value given an individuals genetic data and other context.

This package provides a range of models, feature selection methods, and data transformers following the [scikit-learn API](https://scikit-learn.org/stable/index.html) tailored specifically for genomic prediction.

This includes:

Models provided:
- A wrapper around the [BGLR](https://github.com/gdlc/BGLR-R) bayesian mixed modelling package.
- A convolutional neural network tailored to genomic prediction.

Feature selectors provided:
- A feature selector wrapper around the [GEMMA](https://github.com/genetics-statistics/GEMMA) GWAS software.
- A minibatched [MultiSURF](https://github.com/EpistasisLab/scikit-rebate) feature selector, which produces comparable results to the non-minibatched implementation with reduced runtime and memory requirements in the presence of many samples.
- A minor allele frequency filter.

Marker transformers:
- The [Van Raden](https://doi.org/10.3168/jds.2007-0980) scaling and similarity metrics.
- The [NOIA additive and dominance schemes](https://doi.org/10.1534/genetics.106.067348), and [epistatic transformers using the hadamard product](https://doi.org/10.1159/000339906).
- A rank transformer to convert continuous values into quantised classes.
- An ordinal transformer, which converts ordinal values into cumulative multi-class values, which can be used with the SKlearn [`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor) to perform ordinal regression.


In addition to the scikit learn API compatible models, we also provide Keras layers, models, and regularisers which can assist in genomic prediction.


## Hyperparameter optimisation

We also offer a comprehensive hyperparameter optimisation scheme, which can take markers, grouping factors, and continuous covariates to perform regression, ranking, ordinal regression, or classification tasks.
These optimisers can sample numerous combinations of input feature selectors and transformers to optimise the hyperparameters and find the best performing models in cross validated samples.

These are primarily intended for use using the following python scripts.

`selectml optimise` takes markers and optionally grouping factors and covariates, and runs the optimisation program for a specific family of models (e.g. penalised linear models, random forests, neural networks).
It will then return the information on the performance of models and the best combination of hyper-parameters for the data.

`selectml predict` then takes the parameters from the optimise command, trains a model on all of the training data, and runs predictions in samples with new data.
Data in the test set with known phenotypes are used to calculate performance metrics, which is useful for benchmarking different modelling methods with simulated data.


### `selectml optimise`

```
positional arguments:
  {regression,ranking,ordinal,classification}
                        The type of modelling task to optimise.
  {xgb,knn,rf,ngb,svm,sgd,lars,bglr_sk,bglr,tf}
                        The model type to optimise.
  markers               The marker tsv file to parse as input.
  experiment            The experimental data tsv file.

options:
  -h, --help            show this help message and exit
  -r RESPONSE_COL, --response-col RESPONSE_COL
                        The column to use from experiment as the y value
  -n NAME_COL, --name-col NAME_COL
                        The column to for names to align experiment and marker tables.
  -g GROUP_COLS [GROUP_COLS ...], --group-cols GROUP_COLS [GROUP_COLS ...]
                        The column(s) in the experiment table to use for grouping factors (e.g. different environments) that should be included.
  -c COVARIATE_COLS [COVARIATE_COLS ...], --covariate-cols COVARIATE_COLS [COVARIATE_COLS ...]
                        The column(s) in experiment to use as covariates.
  -o OUTFILE, --outfile OUTFILE
                        Where to write the output to. Default: stdout
  --full FULL           Write the full results of CVs
  --continue CONTINUE_  Where to continue from.
  --pickle PICKLE       Where to save the trials to.
  --importance IMPORTANCE
                        Where to write the output to. Default: stdout
  -b BEST, --best BEST  Write out the best parameters in JSON format
  --ntrials NTRIALS     The number of iterations to try for optimisation.
  --cpu CPU             The number CPUs to use.
  --ntasks NTASKS       The number of optuna tasks to use.
  --seed SEED           The random seed to use.
  --timeout TIMEOUT     The maximum time in hours to run for.

example command:

selectml optimise \
  -r yield \
  -n individual \
  -g "environment1" "environment2" \
  -c "average_rainfall" \
  -o results.tsv \
  --full full_results.tsv \
  --pickle checkpoint.pkl \
  --importance feature_importances.json \
  -b best_params.json \
  --ntrials 200 \
  --cpu 1 \
  --ntasks 1 \
  regression \
  bglr \
  markers.tsv \
  phenos.tsv
```

### `selectml predict`

```
positional arguments:
  {regression,ranking,ordinal,classification}
                        The type of modelling task to optimise.
  {xgb,knn,rf,ngb,svm,sgd,lars,bglr_sk,bglr,tf}
                        The model type to optimise.
  params                The parameters to model with.
  markers               The marker tsv file to parse as input.
  experiment            The experimental data tsv file.

options:
  -h, --help            show this help message and exit
  -t TRAIN_COL, --train-col TRAIN_COL
                        The column to use from experiment to set the training values
  -r RESPONSE_COL, --response-col RESPONSE_COL
                        The column to use from experiment as the y value
  -g GROUP_COLS [GROUP_COLS ...], --group-cols GROUP_COLS [GROUP_COLS ...]
                        The column(s) in the experiment table to use for grouping factors (e.g. different environments) that should be included.
  -c COVARIATE_COLS [COVARIATE_COLS ...], --covariate-cols COVARIATE_COLS [COVARIATE_COLS ...]
                        The column(s) in experiment to use as covariates.
  -n NAME_COL, --name-col NAME_COL
                        The column in experiment that indicates which individual it is.
  -o OUTFILE, --outfile OUTFILE
                        Where to write the output to. Default: stdout
  -s STATS, --stats STATS
                        Where to write the evaluation stats to.
  --outmodel OUTMODEL   Where to write the model
  --cpu CPU             The number CPUs to use.
  --seed SEED           The random seed to use.

example command:

selectml predict \
  -t "dataset" \
  -r "yield" \
  -g "environment1" "environment2" \
  -c "average_rainfall" \
  -n individual \
  -o predictions.tsv \
  -s statistics.tsv \
  --outmodel model.pkl \
  --cpu 1 \
  regression \
  bglr \
  best_params.json \
  markers.tsv \
  phenos.tsv
```
