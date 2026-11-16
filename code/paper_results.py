import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from itertools import product

import plots_and_metrics as pm
from paper_config import *

#######################################################################################################################
################################################## GENERAL SETTINGS ###################################################
#######################################################################################################################

#### Which experiments to process (same flags as in paper_experiments.py)
RUN_EXPERIMENT_1  = True
RUN_EXPERIMENT_2A = True
RUN_EXPERIMENT_2B = True
RUN_EXPERIMENT_3  = True
RUN_EXPERIMENT_4  = True

#### Relative gap below which the methods stagnate (single precision); values below it are not shown
TAU = 1e-6
#### Relative gap tolerances for the tables (low, medium and high accuracy)
TOLS = [ 1e-2, 1e-4, 1e-5 ]

#### Root folder of the results
ROOT_PATH = os.path.join( os.getcwd(), 'paper_results_data' )

#######################################################################################################################
################################################## AUXILIARY ROUTINES #################################################
#######################################################################################################################

def print_header( title ):
    print('\n------------------------------------------------------------------------------------------------------')
    print(f'{title:^102}')
    print('------------------------------------------------------------------------------------------------------\n')

def make_path( name ):
    path = os.path.join( ROOT_PATH, name )
    if not os.path.exists( path ):
        os.makedirs( path )
    return path

def fista_dev_list( params_list, ablation_list, epochs_list, train_dataset_ratio ):
    # Same format used by plots_and_metrics: ( (gamma, alpha, t_variant), train ratio, epochs, (miniter, maxiter), ablation )
    return [ ( params, train_dataset_ratio, epochs, ( MINITER, K ), ablation )
             for params, ablation, epochs in product( params_list, ablation_list, epochs_list ) ]

def deepopt_list( alpha_list, epochs_list, train_dataset_ratio ):
    # Same format used by plots_and_metrics: ( alpha, train ratio, epochs, (miniter, maxiter) )
    return [ ( alpha, train_dataset_ratio, epochs, ( MINITER, K ) )
             for alpha, epochs in product( alpha_list, epochs_list ) ]

def load_rel_gap( mode, test_dataset_ratio, fista_list, fista_dev_list_, deepopt_list_ ):
    # Returns the dicts [ISTA, FISTA..., FISTA-DEV..., DeepOpt...] with 'data' = (F-F*)/F* and quantile lines,
    # and the dicts of inner iterations in the same order
    F_dicts = pm.load_data( 'Fhist', mode, DATASET, test_dataset_ratio, PROBLEM, DATE,
                            fista_list, fista_dev_list_, deepopt_list_ )
    pm.compute_quantiles_rel_opt_gap( F_dicts, DATASET, mode, PROBLEM, test_dataset_ratio )
    inner_dicts = pm.load_data( 'inner_iter_hist', mode, DATASET, test_dataset_ratio, PROBLEM, DATE,
                                fista_list, fista_dev_list_, deepopt_list_ )
    return F_dicts, inner_dicts

def plot_rel_gap( F_dicts, data_path, fig_name, last_iter = None ):
    # Median and interquartile range of (F-F*)/F*, with a reference C/k^2 curve and the stagnation level TAU
    dicts = [ d.copy() for d in F_dicts ]
    if last_iter is not None:
        for d in dicts:
            for key in [ 'center_line', 'upper_line', 'lower_line' ]:
                d[key] = d[key][:last_iter+1]
    iters = dicts[0].get('center_line').shape[0]
    k = 1 + np.arange( iters )
    #### Reference C/k^2, with C chosen so that the curve lies above all the median curves
    C = np.max( [ np.max( d.get('center_line') * k**2 ) for d in dicts ] )
    ref = C / k**2
    dicts = [ dict( center_line = ref, lower_line = ref, upper_line = ref, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + dicts
    #### x-coordinate
    dicts = [ dict( center_line = k, lower_line = k, upper_line = k, label = '$k+1$', ls = None, c = None ) ] + dicts
    #### Plotting
    fig = pm.plot_data( dicts, ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$',
                        fig_path = data_path, fig_name = fig_name )
    #### Stagnation level
    ax = fig.gca()
    ax.axhline( TAU, ls = '--', c = 'gray', lw = 0.8 )
    ax.set_ylim( bottom = TAU / 2 )
    fig.savefig( os.path.join( data_path, fig_name + pm.EXTENSION ) )
    return fig

def plot_stepsizes( mode, test_dataset_ratio, fista_list, fista_dev_list_, deepopt_list_, data_path, fig_name, labels = None ):
    step_dicts = pm.load_data( 'stepsize_hist', mode, DATASET, test_dataset_ratio, PROBLEM, DATE,
                               fista_list, fista_dev_list_, deepopt_list_ )
    pm.compute_quantiles( step_dicts )   # all problems share the same step sizes
    if labels is not None:
        for d, label in zip( step_dicts, labels ):
            d['label'] = label
    iters = step_dicts[0].get('center_line').shape[0]
    k = 1 + np.arange( iters )
    step_dicts = [ dict( center_line = k, lower_line = k, upper_line = k, label = '$k$', ls = None, c = None ) ] + step_dicts
    pm.plot_data( step_dicts, ylabel = r'$\rho_k$', fig_path = data_path, fig_name = fig_name, xscale = 'linear', yscale = 'linear' )

def iters_to_tol_by_label( rel_gap_dicts, inner_dicts, labels, tols, table_path = None, table_name = None ):
    # Methods in the rows, tolerances in the columns:
    #   'iters' : median outer iterations (median total inner iterations) to reach each relative gap tolerance
    #   'frac'  : percentage of problems that reached each tolerance
    iters_rows, frac_rows = [], []
    for d, d_inner in zip( rel_gap_dicts, inner_dicts ):
        rel_gap = d.get('data')
        max_iter = rel_gap.shape[1] - 1
        inner_cum = np.concatenate( [ np.zeros( ( rel_gap.shape[0], 1 ) ), np.cumsum( d_inner.get('data'), axis = 1 ) ], axis = 1 )
        iters_row, frac_row = [], []
        for tol in tols:
            reached = rel_gap <= tol
            ever = reached.any( axis = 1 )
            k_hit = np.where( ever, reached.argmax( axis = 1 ), np.inf )
            inner_hit = np.where( ever, inner_cum[ np.arange( len(k_hit) ), np.where( ever, k_hit, 0 ).astype(int) ], np.inf )
            med, med_inner = np.median( k_hit ), np.median( inner_hit )
            cell = f'{int(med)}' if np.isfinite( med ) else f'> {max_iter}'
            cell += f' ({int(med_inner)})' if np.isfinite( med_inner ) else ' (-)'
            iters_row.append( cell )
            frac_row.append( f'{100 * np.mean( ever ):.0f}%' )
        iters_rows.append( iters_row )
        frac_rows.append( frac_row )
    columns = [ f'{tol:.0e}' for tol in tols ]
    tables = {}
    for key, rows, title in [ ( 'iters', iters_rows, 'median outer iterations (median total inner iterations)' ),
                              ( 'frac', frac_rows, 'percentage of problems reaching the tolerance' ) ]:
        table = pd.DataFrame( rows, index = labels, columns = columns )
        table.index.name = 'method'
        table.columns.name = 'tolerance'
        tables[key] = table
        print( f'{title}:' )
        print( table.to_string(), '\n' )
        if table_path is not None and table_name is not None:
            with open( os.path.join( table_path, f'{table_name}_{key}.txt' ), 'w' ) as f:
                f.write( table.to_string() )
            with open( os.path.join( table_path, f'{table_name}_{key}.tex' ), 'w' ) as f:
                f.write( table.to_latex() )
    return tables

def metrics_grid_table( values, params_list, title, table_path, table_name ):
    # gamma in the rows, alpha in the columns (one table per t_variant)
    df = pd.DataFrame( [ dict( gamma = g, alpha = a, t_variant = tv, value = v ) for (g, a, tv), v in zip( params_list, values ) ] )
    with open( os.path.join( table_path, table_name + '.txt' ), 'w' ) as f:
        for tv in sorted( df['t_variant'].unique() ):
            table = df[ df['t_variant'] == tv ].pivot( index = 'gamma', columns = 'alpha', values = 'value' )
            print( f'{title} (T_k variant {tv}):' )
            print( table.to_string( float_format = '%.2e' ), '\n' )
            f.write( f'{title} (T_k variant {tv}):\n' + table.to_string( float_format = '%.2e' ) + '\n\n' )

#######################################################################################################################
#################################################### EXPERIMENT 1 #####################################################
######################################## Parameter selection for FISTA (untrained) ####################################
#######################################################################################################################

if RUN_EXPERIMENT_1:
    print_header('EXPERIMENT 1 (Start)')
    DATA_PATH = make_path( 'exp1' )

    PARAMS_FISTA = list( product( GAMMA_FISTA, ALPHA_FISTA, T_VARIANT_FISTA ) )
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, PARAMS_FISTA, [], [] )

    #### Tables: gamma x alpha for each tolerance (F_dicts[0] is ISTA)
    pm.iters_to_tol_table( F_dicts[1:], PARAMS_FISTA, TOLS, inner_dicts = inner_dicts[1:],
                           table_path = DATA_PATH, table_name = 'iters_to_tol_fista' )
    #### ISTA, for reference
    iters_to_tol_by_label( F_dicts[:1], inner_dicts[:1], [ 'ISTA' ], TOLS, table_path = DATA_PATH, table_name = 'iters_to_tol_ista' )

    #### Plots: one figure per (gamma, t_variant), with alpha varying, plus ISTA
    for gamma, t_variant in product( GAMMA_FISTA, T_VARIANT_FISTA ):
        idx = [ 1 + i for i, p in enumerate( PARAMS_FISTA ) if p[0] == gamma and p[2] == t_variant ]
        plot_rel_gap( [ F_dicts[0] ] + [ F_dicts[i] for i in idx ], DATA_PATH, f'optgap_gamma{gamma}_tvariant{t_variant}' )

    print_header('EXPERIMENT 1 (End)')

#######################################################################################################################
#################################################### EXPERIMENT 2 #####################################################
################################################ Parameter selection for FISTA-DEV ####################################
#######################################################################################################################

if RUN_EXPERIMENT_2A:
    print_header('EXPERIMENT 2A (Start)')
    DATA_PATH = make_path( 'exp2a' )

    PARAMS_DEV = list( product( GAMMA_DEV, ALPHA_DEV, T_VARIANT_DEV ) )
    FISTA_DEV_LIST = fista_dev_list( PARAMS_DEV, [ 'all' ], SCREEN_EPOCHS_TO_SAVE, SCREEN_TRAIN_DATASET_RATIO )
    DEEPOPT_LIST   = deepopt_list( ALPHA_DO, SCREEN_EPOCHS_TO_SAVE, SCREEN_TRAIN_DATASET_RATIO )

    #### Validation loss (same quantity minimized in training) and penalty after the training horizon
    val_loss, penalty, labels = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, DEEPOPT_LIST )
    n_dev = len( FISTA_DEV_LIST )
    for epochs in SCREEN_EPOCHS_TO_SAVE:
        idx = [ i for i, o in enumerate( FISTA_DEV_LIST ) if o[2] == epochs ]
        metrics_grid_table( val_loss[idx], [ FISTA_DEV_LIST[i][0] for i in idx ], f'FISTA-DEV validation loss, {epochs} epochs',
                            DATA_PATH, f'val_loss_fista_dev_ep{epochs}' )
        metrics_grid_table( penalty[idx], [ FISTA_DEV_LIST[i][0] for i in idx ], f'FISTA-DEV penalty, {epochs} epochs',
                            DATA_PATH, f'penalty_fista_dev_ep{epochs}' )
    print('DeepOpt:')
    pm.print_metrics_analysis( val_loss[n_dev:], penalty[n_dev:], labels[n_dev:], table_path = DATA_PATH, table_name = 'metrics_deepopt' )

    #### Iterations to reach each tolerance
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, [], FISTA_DEV_LIST, DEEPOPT_LIST )
    start = 1   # F_dicts[0] is ISTA
    for epochs in SCREEN_EPOCHS_TO_SAVE:
        idx = [ i for i, o in enumerate( FISTA_DEV_LIST ) if o[2] == epochs ]
        print(f'===== FISTA-DEV, {epochs} epochs =====')
        pm.iters_to_tol_table( [ F_dicts[start+i] for i in idx ], [ FISTA_DEV_LIST[i][0] for i in idx ], TOLS,
                               inner_dicts = [ inner_dicts[start+i] for i in idx ],
                               table_path = DATA_PATH, table_name = f'iters_to_tol_fista_dev_ep{epochs}' )
    print('===== DeepOpt =====')
    iters_to_tol_by_label( F_dicts[start+n_dev:], inner_dicts[start+n_dev:],
                           [ f'DeepOpt alpha={o[0]}, {o[2]} epochs' for o in DEEPOPT_LIST ], TOLS,
                           table_path = DATA_PATH, table_name = 'iters_to_tol_deepopt' )

    print_header('EXPERIMENT 2A (End)')

if RUN_EXPERIMENT_2B:
    print_header('EXPERIMENT 2B (Start)')
    DATA_PATH = make_path( 'exp2b' )

    FISTA_DEV_LIST = fista_dev_list( FINALISTS_FISTA_DEV, [ 'all' ], FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO )
    DEEPOPT_LIST   = deepopt_list( [ BEST_ALPHA_DO ], FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO )
    n_dev = len( FISTA_DEV_LIST )

    #### Validation loss and penalty
    val_loss, penalty, labels = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, DEEPOPT_LIST )
    pm.print_metrics_analysis( val_loss, penalty, labels, table_path = DATA_PATH, table_name = 'metrics_table' )

    #### Iterations to reach each tolerance and relative gap
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, [ BEST_FISTA ], FISTA_DEV_LIST, DEEPOPT_LIST )
    labels_table  = [ 'ISTA', f'FISTA gamma={BEST_FISTA[0]}, alpha={BEST_FISTA[1]}' ]
    labels_table += [ f'FISTA-DEV gamma={o[0][0]}, alpha={o[0][1]}, {o[2]} epochs' for o in FISTA_DEV_LIST ]
    labels_table += [ f'DeepOpt alpha={o[0]}, {o[2]} epochs' for o in DEEPOPT_LIST ]
    iters_to_tol_by_label( F_dicts, inner_dicts, labels_table, TOLS, table_path = DATA_PATH, table_name = 'iters_to_tol' )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap_zoom', last_iter = K + 10 )

    print_header('EXPERIMENT 2B (End)')

#######################################################################################################################
#################################################### EXPERIMENT 3 #####################################################
################################################## Ablation study for FISTA-DEV #######################################
#######################################################################################################################

if RUN_EXPERIMENT_3:
    print_header('EXPERIMENT 3 (Start)')
    DATA_PATH = make_path( 'exp3' )

    ABLATION_ALL    = ABLATION + [ 'all' ]
    ABLATION_LABELS = { 'd': 'only $d_k$', 'e': 'only $e_k$', 'v': 'only $v_k$',
                        'dev': 'deviations', 'theta': 'step sizes', 'all': 'deviations and step sizes' }
    FISTA_DEV_LIST = fista_dev_list( [ BEST_FISTA_DEV ], ABLATION_ALL, [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO )

    #### Relative gap: FISTA (same gamma, alpha, t_variant as FISTA-DEV, i.e., no learned components) and the ablations
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, [ BEST_FISTA_DEV ], FISTA_DEV_LIST, [] )
    F_dicts, inner_dicts = F_dicts[1:], inner_dicts[1:]   # ISTA is not shown in the ablation
    labels = [ 'FISTA (nothing learned)' ] + [ 'FISTA-DEV: ' + ABLATION_LABELS[a] for a in ABLATION_ALL ]
    for d, label in zip( F_dicts, labels ):
        d['label'] = label
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap_zoom', last_iter = K + 10 )
    iters_to_tol_by_label( F_dicts, inner_dicts, labels, TOLS, table_path = DATA_PATH, table_name = 'iters_to_tol' )

    #### Validation loss and penalty
    val_loss, penalty, _ = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, [] )
    pm.print_metrics_analysis( val_loss, penalty, labels[1:], table_path = DATA_PATH, table_name = 'metrics_table' )

    #### Learned step sizes (only the ablations that learn them, compared with mu/L)
    STEP_ABLATION = [ 'theta', 'all' ]
    plot_stepsizes( 'val', VAL_DATASET_RATIO, [ BEST_FISTA_DEV ],
                    fista_dev_list( [ BEST_FISTA_DEV ], STEP_ABLATION, [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO ), [],
                    DATA_PATH, 'stepsize',
                    labels = [ 'ISTA', 'FISTA' ] + [ 'FISTA-DEV: ' + ABLATION_LABELS[a] for a in STEP_ABLATION ] )

    print_header('EXPERIMENT 3 (End)')

#######################################################################################################################
#################################################### EXPERIMENT 4 #####################################################
############################################ Comparison with the baselines (test set) #################################
#######################################################################################################################

if RUN_EXPERIMENT_4:
    print_header('EXPERIMENT 4 (Start)')
    DATA_PATH = make_path( 'exp4' )

    gamma_best, alpha_best, t_variant_best = BEST_FISTA
    FISTA_TEST     = [ ( gamma_best, 0.0, t_variant_best ), BEST_FISTA ]
    FISTA_DEV_LIST = fista_dev_list( [ BEST_FISTA_DEV ], [ 'all' ], [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO )
    DEEPOPT_LIST   = deepopt_list( [ BEST_ALPHA_DO ], [ FINAL_EPOCHS ], FULL_TRAIN_DATASET_RATIO )

    F_dicts, inner_dicts = load_rel_gap( 'test', TEST_DATASET_RATIO, FISTA_TEST, FISTA_DEV_LIST, DEEPOPT_LIST )
    labels = [ 'ISTA', r'FISTA ($\alpha=0$)', rf'FISTA ($\alpha={alpha_best}$)', 'FISTA-DEV', 'DeepOpt' ]
    for d, label in zip( F_dicts, labels ):
        d['label'] = label

    #### Relative gap (all iterations and zoom on the learned iterations and the transition after K)
    fig = plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap_zoom', last_iter = K + 10 )
    #### Marks where the proximal solver starts to fail
    flag_dicts = pm.load_data( 'flag_hist', 'test', DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, FISTA_TEST, FISTA_DEV_LIST, DEEPOPT_LIST )
    for d, label in zip( flag_dicts, labels ):
        d['label'] = label
    pm.add_marks( flag_dicts, fig, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

    #### Iterations to reach each tolerance
    iters_to_tol_by_label( F_dicts, inner_dicts, labels, TOLS, table_path = DATA_PATH, table_name = 'iters_to_tol' )

    #### Step sizes
    plot_stepsizes( 'test', TEST_DATASET_RATIO, FISTA_TEST, FISTA_DEV_LIST, DEEPOPT_LIST, DATA_PATH, 'stepsize', labels = labels )

    print_header('EXPERIMENT 4 (End)')

print( '\nAll data was saved to:', ROOT_PATH )