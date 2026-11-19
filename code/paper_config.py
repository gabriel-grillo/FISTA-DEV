#######################################################################################################################
############################# SHARED CONFIGURATION OF paper_experiments.py AND paper_results.py #######################
#######################################################################################################################
# Edit the SELECTED PARAMETERS after each experiment: both scripts read them from here, so that the results are
# always processed with the same configurations that were trained and tested.

#### Date to save results (fixed, so that all the experiments are found by paper_results.py)
DATE = 'paper'

#### Problem
DATASET = 'mayo_clinic_512'
PROBLEM = 'lstv'

#### Learned iterations: the networks/stepsizes act on iterations 1,...,K (training uses N in [MINITER, K+10])
MINITER = 1
K       = 10

#### Training
TRAIN_BATCH_SIZE  = 1
LOAD_SAVED_MODELS = False

#### Testing
TEST_ITERS      = 200
TEST_BATCH_SIZE = 32

#### Experiment 1: parameter selection for FISTA (untrained)
GAMMA_FISTA       = [ 1.1, 1.5, 2.0, 2.5 ]
ALPHA_FISTA       = [ 0.0, 0.25, 0.5, 0.75, 0.9 ]
T_VARIANT_FISTA   = [ 1 ]
VAL_DATASET_RATIO = 1.0

#### Experiment 2A: screening for FISTA-DEV and DeepOpt
GAMMA_DEV                  = [ 1.1, 1.5, 2.0 ]
ALPHA_DEV                  = [ 0.25, 0.5, 0.75 ]
T_VARIANT_DEV              = [ 1 ]
ALPHA_DO                   = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
SCREEN_TRAIN_DATASET_RATIO = 0.2
SCREEN_EPOCHS_TO_SAVE      = [ 10 ]

#### Experiments 2B, 3 and 4: full budget
FULL_TRAIN_DATASET_RATIO = 1.0
FULL_EPOCHS_TO_SAVE      = [ 10, 20 ]   # epochs saved in Experiment 2B
FINAL_EPOCHS             = 20           # epochs used in Experiments 3 and 4
TEST_DATASET_RATIO       = 1.0

#### Experiment 3: ablation (the full model 'all' is trained in Experiment 2B)
ABLATION = [ 'd', 'e', 'v', 'dev', 'theta' ]

#######################################################################################################################
##################################### SELECTED PARAMETERS (EDIT AFTER EXPERIMENTS) ####################################
#######################################################################################################################

#### Best FISTA parameters (from Experiment 1) -- (gamma, alpha, t_variant)
BEST_FISTA = ( 1.5, 0.5, 1 )

#### FISTA-DEV configurations that pass the screening (from Experiment 2A) -- list of (gamma, alpha, t_variant)
FINALISTS_FISTA_DEV = [ ( 1.1, 0.25, 1 ), ( 1.5, 0.25, 1 ), ( 1.5, 0.5, 1 ) ]

#### Selected FISTA-DEV parameters (from Experiment 2B) -- (gamma, alpha, t_variant)
BEST_FISTA_DEV = ( 1.5, 0.7, 1 )

#### DeepOpt parameters that pass the screening (from Experiment 2A)
FINALISTS_DO = [ 0.2, 0.3, 0.4 ]

#### Selected DeepOpt parameter (from Experiment 2B)
BEST_ALPHA_DO = 0.5