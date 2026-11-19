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
RUN_EXPERIMENT_1  = False
RUN_EXPERIMENT_2A = False
RUN_EXPERIMENT_2B = True
RUN_EXPERIMENT_3  = False
RUN_EXPERIMENT_4  = False

#### Relative gap below which F* (from the 500 iterations of FISTA in F_ref) is not accurate; values below it are
#### not shown (see estimate_tau.py)
TAU = 1e-6
#### Cost of one inner iteration of proxTV relative to one gradient evaluation (A and A^T), measured with
#### measure_costs.py with batches of 32 problems. Cost of a run, in gradient evaluations: k + R_COST * (inner iterations)
R_COST = 0.022
#### Cost of one forward pass of each network relative to one gradient evaluation (measure_costs.py, batches of 32).
#### The networks are only applied in the first K iterations. FILL WITH THE VALUES MEASURED BY measure_costs.py.
R_NET    = { 'd': 0.191, 'e': 0.193, 'v': 0.199 }   # FISTA-DEV networks
R_NET_DO = 0.383                               # DeepOpt networks (dx1 + dx2)
if R_NET_DO == 0.0 or sum( R_NET.values() ) == 0.0:
    print( 'WARNING: the costs of the networks (R_NET, R_NET_DO) are zero; run measure_costs.py and fill them in.' )

def net_cost_fista_dev( ablation ):
    # Cost per learned iteration of the networks used by FISTA-DEV with the given ablation
    learned = { 'd': ['d'], 'e': ['e'], 'v': ['v'], 'dev': ['d','e','v'], 'theta': [], 'all': ['d','e','v'] }[ablation]
    return sum( R_NET[c] for c in learned )

def net_costs( n_untrained, fista_dev_list_, deepopt_list_ ):
    # Network costs aligned with the dicts returned by load_rel_gap: [ISTA, FISTA..., FISTA-DEV..., DeepOpt...]
    return [ 0.0 ] * ( 1 + n_untrained ) + [ net_cost_fista_dev( o[4] ) for o in fista_dev_list_ ] + [ R_NET_DO ] * len( deepopt_list_ )
#### Relative gap tolerances for the tables (low and high accuracy)
TOLS = [ 1e-2, 1e-5 ]

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

def cost_in_gradient_evaluations( inner_dict, r_cost = R_COST, r_net = 0.0 ):
    # Cost of iterations 0,...,MAXITER of each problem, in gradient evaluations:
    #   k + r_cost * (inner iterations up to k) + r_net * min(k, K)
    inner = inner_dict.get('data')
    k = np.arange( inner.shape[1] + 1 )[None,:]
    inner_cum = np.concatenate( [ np.zeros( ( inner.shape[0], 1 ) ), np.cumsum( inner, axis = 1 ) ], axis = 1 )
    return k + r_cost * inner_cum + r_net * np.minimum( k, K )

def plot_rel_gap_vs_cost( F_dicts, inner_dicts, data_path, fig_name, nets = None, r_cost = R_COST, n_grid = 400, max_cost = None ):
    # Median and interquartile range of (F-F*)/F* as a function of the cost, in gradient evaluations. For each problem,
    # the relative gap at cost c is the one of the last iterate whose cost does not exceed c (step function).
    nets = [ 0.0 ] * len( inner_dicts ) if nets is None else nets
    costs = [ cost_in_gradient_evaluations( d, r_cost, r_net ) for d, r_net in zip( inner_dicts, nets ) ]
    if max_cost is None:
        max_cost = max( np.max( c ) for c in costs )
    grid = np.geomspace( 1.0, max_cost, n_grid )
    dicts = []
    for d, c in zip( F_dicts, costs ):
        rel_gap = d.get('data')
        values = np.full( ( rel_gap.shape[0], n_grid ), np.nan )
        for p in range( rel_gap.shape[0] ):
            idx = np.searchsorted( c[p], grid, side = 'right' ) - 1
            available = grid <= c[p, -1]          # beyond the end of the run, the value is not available
            values[p, available] = rel_gap[p, idx[available]]
        # quantiles only where at least half of the problems are available
        enough = np.mean( np.isfinite( values ), axis = 0 ) >= 0.5
        q25, med, q75 = np.full( ( 3, n_grid ), np.nan )
        q25[enough], med[enough], q75[enough] = np.nanquantile( values[:, enough], [ 0.25, 0.5, 0.75 ], axis = 0 )
        dicts.append( dict( center_line = med, lower_line = q25, upper_line = q75,
                            label = d.get('label'), ls = d.get('ls'), c = d.get('c') ) )
    dicts = [ dict( center_line = grid, lower_line = grid, upper_line = grid,
                    label = 'cost (gradient evaluations)', ls = None, c = None ) ] + dicts
    fig = pm.plot_data( dicts, ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$', fig_path = data_path, fig_name = fig_name )
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

def cost_to_tol_by_label( rel_gap_dicts, inner_dicts, labels, tols, table_path = None, table_name = None, nets = None, r_cost = R_COST ):
    # Methods in the rows, tolerances in the columns, entries 'median cost (percentage of problems reaching the tolerance)',
    # with cost k + r_cost * (inner iterations up to k) + r_net * min(k, K) in gradient evaluations (see pm.cost_to_tol)
    nets = [ 0.0 ] * len( rel_gap_dicts ) if nets is None else nets
    rows = [ [ pm.cost_frac_cell( *pm.cost_to_tol( d.get('data'), tol, d_inner.get('data'), r_cost, r_net, K ) ) for tol in tols ]
             for d, d_inner, r_net in zip( rel_gap_dicts, inner_dicts, nets ) ]
    table = pd.DataFrame( rows, index = labels, columns = [ f'{tol:.0e}' for tol in tols ] )
    table.index.name = 'method'
    table.columns.name = 'tolerance'
    print( f'Median cost in gradient evaluations, r = {r_cost} (percentage of problems reaching the tolerance):' )
    print( table.to_string(), '\n' )
    if table_path is not None and table_name is not None:
        with open( os.path.join( table_path, f'{table_name}.txt' ), 'w' ) as f:
            f.write( table.to_string() )
        with open( os.path.join( table_path, f'{table_name}.tex' ), 'w' ) as f:
            f.write( table.to_latex() )
    return table

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
    pm.cost_to_tol_table( F_dicts[1:], PARAMS_FISTA, TOLS, inner_dicts = inner_dicts[1:],
                           table_path = DATA_PATH, table_name = 'cost_to_tol_fista', r_cost = R_COST )
    #### ISTA, for reference
    cost_to_tol_by_label( F_dicts[:1], inner_dicts[:1], [ 'ISTA' ], TOLS, table_path = DATA_PATH, table_name = 'cost_to_tol_ista' )

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

    #### Validation loss (same quantity minimized in training)
    val_loss, labels = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, DEEPOPT_LIST )
    n_dev = len( FISTA_DEV_LIST )
    for epochs in SCREEN_EPOCHS_TO_SAVE:
        idx = [ i for i, o in enumerate( FISTA_DEV_LIST ) if o[2] == epochs ]
        metrics_grid_table( val_loss[idx], [ FISTA_DEV_LIST[i][0] for i in idx ], f'FISTA-DEV validation loss, {epochs} epochs',
                            DATA_PATH, f'val_loss_fista_dev_ep{epochs}' )
    print('DeepOpt:')
    pm.print_metrics_analysis( val_loss[n_dev:], labels[n_dev:], table_path = DATA_PATH, table_name = 'val_loss_deepopt' )

    #### Iterations to reach each tolerance
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, [], FISTA_DEV_LIST, DEEPOPT_LIST )
    start = 1   # F_dicts[0] is ISTA
    for epochs in SCREEN_EPOCHS_TO_SAVE:
        idx = [ i for i, o in enumerate( FISTA_DEV_LIST ) if o[2] == epochs ]
        print(f'===== FISTA-DEV, {epochs} epochs =====')
        pm.cost_to_tol_table( [ F_dicts[start+i] for i in idx ], [ FISTA_DEV_LIST[i][0] for i in idx ], TOLS,
                               inner_dicts = [ inner_dicts[start+i] for i in idx ],
                               table_path = DATA_PATH, table_name = f'cost_to_tol_fista_dev_ep{epochs}', r_cost = R_COST,
                               r_net = net_cost_fista_dev( 'all' ), k_net = K )
    print('===== DeepOpt =====')
    cost_to_tol_by_label( F_dicts[start+n_dev:], inner_dicts[start+n_dev:],
                           [ f'DeepOpt alpha={o[0]}, {o[2]} epochs' for o in DEEPOPT_LIST ], TOLS,
                           table_path = DATA_PATH, table_name = 'cost_to_tol_deepopt', nets = [ R_NET_DO ] * len( DEEPOPT_LIST ) )

    print_header('EXPERIMENT 2A (End)')

if RUN_EXPERIMENT_2B:
    print_header('EXPERIMENT 2B (Start)')
    DATA_PATH = make_path( 'exp2b' )

    FISTA_DEV_LIST = fista_dev_list( FINALISTS_FISTA_DEV, [ 'all' ], FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO )
    DEEPOPT_LIST   = deepopt_list( FINALISTS_DO, FULL_EPOCHS_TO_SAVE, FULL_TRAIN_DATASET_RATIO )
    n_dev = len( FISTA_DEV_LIST )

    #### Validation loss
    val_loss, labels = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, DEEPOPT_LIST )
    pm.print_metrics_analysis( val_loss, labels, table_path = DATA_PATH, table_name = 'val_loss' )

    #### Iterations to reach each tolerance and relative gap
    F_dicts, inner_dicts = load_rel_gap( 'val', VAL_DATASET_RATIO, [ BEST_FISTA ], FISTA_DEV_LIST, DEEPOPT_LIST )
    labels_table  = [ 'ISTA', f'FISTA gamma={BEST_FISTA[0]}, alpha={BEST_FISTA[1]}' ]
    labels_table += [ f'FISTA-DEV gamma={o[0][0]}, alpha={o[0][1]}, {o[2]} epochs' for o in FISTA_DEV_LIST ]
    labels_table += [ f'DeepOpt alpha={o[0]}, {o[2]} epochs' for o in DEEPOPT_LIST ]
    NETS = net_costs( 1, FISTA_DEV_LIST, DEEPOPT_LIST )
    cost_to_tol_by_label( F_dicts, inner_dicts, labels_table, TOLS, table_path = DATA_PATH, table_name = 'cost_to_tol', nets = NETS )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap_vs_cost( F_dicts, inner_dicts, DATA_PATH, 'optgap_vs_cost', nets = NETS )
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
    NETS = [ 0.0 ] + [ net_cost_fista_dev( a ) for a in ABLATION_ALL ]
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap_vs_cost( F_dicts, inner_dicts, DATA_PATH, 'optgap_vs_cost', nets = NETS )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap_zoom', last_iter = K + 10 )
    cost_to_tol_by_label( F_dicts, inner_dicts, labels, TOLS, table_path = DATA_PATH, table_name = 'cost_to_tol', nets = NETS )

    #### Validation loss
    val_loss, _ = pm.compute_metrics( 'val', DATASET, VAL_DATASET_RATIO, PROBLEM, DATE, FISTA_DEV_LIST, [] )
    pm.print_metrics_analysis( val_loss, labels[1:], table_path = DATA_PATH, table_name = 'val_loss' )

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
    NETS = net_costs( len( FISTA_TEST ), FISTA_DEV_LIST, DEEPOPT_LIST )
    fig = plot_rel_gap( F_dicts, DATA_PATH, 'optgap' )
    plot_rel_gap_vs_cost( F_dicts, inner_dicts, DATA_PATH, 'optgap_vs_cost', nets = NETS )
    plot_rel_gap( F_dicts, DATA_PATH, 'optgap_zoom', last_iter = K + 10 )
    #### Marks where the proximal solver starts to fail
    flag_dicts = pm.load_data( 'flag_hist', 'test', DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, FISTA_TEST, FISTA_DEV_LIST, DEEPOPT_LIST )
    for d, label in zip( flag_dicts, labels ):
        d['label'] = label
    pm.add_marks( flag_dicts, fig, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

    #### Iterations to reach each tolerance
    cost_to_tol_by_label( F_dicts, inner_dicts, labels, TOLS, table_path = DATA_PATH, table_name = 'cost_to_tol', nets = NETS )

    #### Step sizes
    plot_stepsizes( 'test', TEST_DATASET_RATIO, FISTA_TEST, FISTA_DEV_LIST, DEEPOPT_LIST, DATA_PATH, 'stepsize', labels = labels )

    print_header('EXPERIMENT 4 (End)')

print( '\nAll data was saved to:', ROOT_PATH )