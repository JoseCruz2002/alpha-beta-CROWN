conda activate alpha-beta-crown
export PYTHONPATH="${PYTHONPATH}:complete_verifier"
echo ${PYTHONPATH}

## *************************************** FFNN from elsa_cybersecurity models *************************************** ##
python expl_algos/src/algorithms/deletion_based_algo.py -config_file_name FFNN_oraclePrepare_Variance.yaml -classifier FFNN_normal_small_CEL0109__VarianceFS-04 -feat_set_file VarianceFS-04.json -sampleSHA EDB121E71FE32DE368BB82AF493CE2AB657402E9E4D7A80462A4599EA49A1DF1 -distance 5

python expl_algos/src/algorithms/deletion_based_algo.py -config_file_name FFNN_oraclePrepare_Univariate.yaml -classifier FFNN_normal_small_CEL0109__UnivariateFS-k_best-mutual_info_classif-10000 -feat_set_file UnivariateFS-k_best-mutual_info_classif-10000.json -sampleSHA EDB121E71FE32DE368BB82AF493CE2AB657402E9E4D7A80462A4599EA49A1DF1 -distance 1
python expl_algos/src/algorithms/deletion_based_algo.py -config_file_name FFNN_oraclePrepare_Univariate.yaml -classifier FFNN_normal_small_CEL0109__UnivariateFS-k_best-mutual_info_classif-10000 -feat_set_file UnivariateFS-k_best-mutual_info_classif-10000.json -sampleSHA EDB121E71FE32DE368BB82AF493CE2AB657402E9E4D7A80462A4599EA49A1DF1 -distance 5

python expl_algos/src/algorithms/deletion_based_algo.py -config_file_name FFNN_oraclePrepare_Univariate.yaml -classifier FFNN_normal_small_CEL0109__UnivariateFS-k_best-mutual_info_classif-500 -feat_set_file UnivariateFS-k_best-mutual_info_classif-500.json -sampleSHA 56A988BA09D4840E4E110A0191E8B4D9C46F03C88BB2E96601781AE553C531DB -distance 1
python expl_algos/src/algorithms/deletion_based_algo.py -config_file_name FFNN_oraclePrepare_Univariate_GCP.yaml -classifier FFNN_normal_small_CEL0109__UnivariateFS-k_best-mutual_info_classif-500 -feat_set_file UnivariateFS-k_best-mutual_info_classif-500.json -sampleSHA 56A988BA09D4840E4E110A0191E8B4D9C46F03C88BB2E96601781AE553C531DB -distance 1
## *************************************** *********************************** *************************************** ##

## *************************************** VNNCOMP benchmarks *************************************** ##
python expl_algos/src/algorithms/deletion_based_algo.py 
## *************************************** ****************** *************************************** ##
