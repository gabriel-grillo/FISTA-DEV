import os
#### For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf

#### GPU settings
TF_USE_GPU  = True
VRAM_MB = 13.0 * 1024
if TF_USE_GPU:
    # Listing visable GPUs and asserting that there is at least one
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU detected.")
    # Configuring TensorFlow GPU memory limit for the first GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [ tf.config.LogicalDeviceConfiguration( memory_limit = VRAM_MB) ]
    )
else:
    tf.config.set_visible_devices( [], 'GPU' )

#### Tensorflow settings
# Setting random seeds for 'random', 'numpy' and 'tensorflow'
tf.keras.utils.set_random_seed(42)
# Enabling operations determinism
tf.config.experimental.enable_op_determinism()

import time
from itertools import product
from traintest import test_untrained, train_fista_dev, test_fista_dev, train_deepopt, test_deepopt

from paper_config import *

#### Which experiments to run (they are meant to be run in order, editing the selected parameters in paper_config.py
#### after the analysis of each experiment with paper_results.py)
RUN_EXPERIMENT_1  = True    # Parameter selection for FISTA (untrained)
RUN_EXPERIMENT_2A = True    # Parameter selection for FISTA-DEV and DeepOpt: screening
RUN_EXPERIMENT_2B = True    # Parameter selection for FISTA-DEV: full budget for the finalists
RUN_EXPERIMENT_3  = True    # Ablation study for FISTA-DEV
RUN_EXPERIMENT_4  = True    # Comparison with the baselines on the test set

#######################################################################################################################
################################################## AUXILIARY ROUTINES #################################################
#######################################################################################################################

def print_header( title ):
    print('\n------------------------------------------------------------------------------------------------------')
    print(f'{title:^102}')
    print('------------------------------------------------------------------------------------------------------\n')

def run_untrained( mode, params_list, test_dataset_ratio, run_ista = True ):
    if run_ista:
        print(f'Testing ISTA ({mode}) with {100*test_dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, mode, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = test_dataset_ratio )
    for gamma, alpha, t_variant in params_list:
        print(f'Testing FISTA ({mode}): gamma={gamma}, alpha={alpha}, t_variant={t_variant}, {100*test_dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, mode, 'fista', TEST_BATCH_SIZE, TEST_ITERS,
                        gamma = gamma, alpha = alpha, t_variant = t_variant, dataset_ratio = test_dataset_ratio )

def run_fista_dev( mode, params_list, ablation_list, epochs_to_save, train_dataset_ratio, test_dataset_ratio, train = True ):
    for i, ( params, ablation ) in enumerate( product( params_list, ablation_list ) ):
        gamma, alpha, t_variant = params
        label  = f'FISTA-DEV {i+1}: N=[{MINITER},{K}], gamma={gamma}, alpha={alpha}, t_variant={t_variant}, '
        label += f'data_ratio={train_dataset_ratio}, ablation={ablation}'
        if train:
            print('Training ' + label)
            train_fista_dev( DATASET, PROBLEM, MINITER, K, gamma, alpha,
                             max(epochs_to_save), TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS, ablation, t_variant,
                             dataset_ratio = train_dataset_ratio, epochs_to_save = epochs_to_save )
        for epochs in epochs_to_save:
            print(f'Testing ({mode}) ' + label + f' trained with {epochs} epochs')
            test_fista_dev( DATASET, PROBLEM, mode, MINITER, K, gamma, alpha, epochs,
                            TEST_BATCH_SIZE, TEST_ITERS, DATE, ablation, t_variant,
                            dataset_ratio = ( train_dataset_ratio, test_dataset_ratio ) )

def run_deepopt( mode, alpha_list, epochs_to_save, train_dataset_ratio, test_dataset_ratio, train = True ):
    for i, alpha in enumerate( alpha_list ):
        label = f'DeepOpt {i+1}: N=[{MINITER},{K}], alpha={alpha}, data_ratio={train_dataset_ratio}'
        if train:
            print('Training ' + label)
            train_deepopt( DATASET, PROBLEM, MINITER, K, alpha,
                           max(epochs_to_save), TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                           dataset_ratio = train_dataset_ratio, epochs_to_save = epochs_to_save )
        for epochs in epochs_to_save:
            print(f'Testing ({mode}) ' + label + f' trained with {epochs} epochs')
            test_deepopt( DATASET, PROBLEM, mode, MINITER, K, alpha, epochs, DATE,
                          TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = ( train_dataset_ratio, test_dataset_ratio ) )

#### For time measurement
start_time = time.time()

#######################################################################################################################
#################################################### EXPERIMENT 1 #####################################################
######################################## Parameter selection for FISTA (untrained) ####################################
#######################################################################################################################
# ISTA and FISTA on the validation set, for a grid of (gamma, alpha).
# Analysis (paper_results.py): cost_to_tol_table -> choose BEST_FISTA.

if RUN_EXPERIMENT_1:
    print_header('EXPERIMENT 1 (Start)')

    run_untrained( 'val', list( product( GAMMA_FISTA, ALPHA_FISTA, T_VARIANT_FISTA ) ), VAL_DATASET_RATIO, run_ista = True )

    print_header('EXPERIMENT 1 (End)')

#######################################################################################################################
#################################################### EXPERIMENT 2 #####################################################
################################################ Parameter selection for FISTA-DEV ####################################
#######################################################################################################################
# Stage A (screening, flag RUN_EXPERIMENT_2A): grid of (gamma, alpha) with a fraction of the training data and few epochs.
#                      DeepOpt's parameter is screened with the same budget.
#                      Analysis (paper_results.py): validation loss + cost_to_tol_table -> FINALISTS_FISTA_DEV, FINALISTS_DO.
# Stage B (full budget, flag RUN_EXPERIMENT_2B): finalists retrained with all the training data and the full number of epochs.
#                      Analysis (paper_results.py): validation loss + cost_to_tol_table -> BEST_FISTA_DEV, BEST_ALPHA_DO.

if RUN_EXPERIMENT_2A:
    print_header('EXPERIMENT 2A (Start)')

    #### Stage A: screening
    run_fista_dev( 'val', list( product( GAMMA_DEV, ALPHA_DEV, T_VARIANT_DEV ) ), [ 'all' ],
                   SCREEN_EPOCHS_TO_SAVE, SCREEN_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO )
    run_deepopt( 'val', ALPHA_DO, SCREEN_EPOCHS_TO_SAVE, SCREEN_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO )

    print_header('EXPERIMENT 2A (End)')

if RUN_EXPERIMENT_2B:
    print_header('EXPERIMENT 2B (Start)')

    #### Stage B: full budget for the finalists of FISTA-DEV and of DeepOpt
    run_fista_dev( 'val', FINALISTS_FISTA_DEV, [ 'all' ],
                   FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO )
    run_deepopt( 'val', FINALISTS_DO, FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO )

    print_header('EXPERIMENT 2B (End)')

#######################################################################################################################
#################################################### EXPERIMENT 3 #####################################################
################################################## Ablation study for FISTA-DEV #######################################
#######################################################################################################################
# FISTA-DEV with the selected parameters, learning only some of the components, on the validation set.
# The full model ('all') was trained in Experiment 2 (stage B) and is only tested here.
# Analysis (paper_results.py): relative gap + step sizes + cost_to_tol_table for each ablation.

if RUN_EXPERIMENT_3:
    print_header('EXPERIMENT 3 (Start)')

    run_fista_dev( 'val', [ BEST_FISTA_DEV ], ABLATION,
                   [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO )
    run_fista_dev( 'val', [ BEST_FISTA_DEV ], [ 'all' ],
                   [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO, VAL_DATASET_RATIO, train = False )

    print_header('EXPERIMENT 3 (End)')

#######################################################################################################################
#################################################### EXPERIMENT 4 #####################################################
############################################ Comparison with the baselines (test set) #################################
#######################################################################################################################
# ISTA, FISTA (classical alpha = 0 and BEST_FISTA), FISTA-DEV (BEST_FISTA_DEV) and DeepOpt (BEST_ALPHA_DO) on the
# test set. The trained models come from Experiment 2 (stage B) and are only tested here.
# Analysis (paper_results.py): relative gap + cost_to_tol_table + wall-clock time.

if RUN_EXPERIMENT_4:
    print_header('EXPERIMENT 4 (Start)')

    gamma_best, _, t_variant_best = BEST_FISTA
    FISTA_TEST = [ BEST_FISTA, ( gamma_best, 0.0, t_variant_best ) ]

    run_untrained( 'test', FISTA_TEST, TEST_DATASET_RATIO, run_ista = True )
    run_fista_dev( 'test', [ BEST_FISTA_DEV ], [ 'all' ],
                   [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO, TEST_DATASET_RATIO, train = False )
    run_deepopt( 'test', [ BEST_ALPHA_DO ], [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO, TEST_DATASET_RATIO, train = False )

    print_header('EXPERIMENT 4 (End)')

#### For time measurement
end_time = time.time()
print(f"\nElapsed time: {end_time - start_time:.5e} seconds")