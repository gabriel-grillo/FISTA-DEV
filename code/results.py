import os
import numpy as np
from itertools import product
from datetime import datetime

from paths import plot_path
import plots_and_metrics as pm

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 0.05
PROBLEM             = 'lstv'

# TRANING PARAMETERS
DATE                = '2026-11-16'
# DATE                = datetime.today().strftime('%Y-%m-%d')
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [ 1 ]
MAXITER             = [ 10 ]
TRAIN_DATASET_RATIO = [ 0.05 ]

# FISTA PARAMETER
GAMMA_FISTA_UNTRAINED  = [ 1.1 ]
ALPHA_FISTA_UNTRAINED  = [ 0.5 ]
T_VARIANT_UNTRAINED    = [ 1 ]
PARAMS_FISTA_UNTRAINED = list( product( GAMMA_FISTA_UNTRAINED, ALPHA_FISTA_UNTRAINED, T_VARIANT_UNTRAINED ) )

# FISTA-DEV PARAMETERS
GAMMA     = [ 1.1 ]
ALPHA     = [ 0.5 ]
T_VARIANT = [ 1 ]
PARAMS    = list( product( GAMMA, ALPHA, T_VARIANT ) )
ABLATION = [ 'all' ]
FISTA_DEV_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ), ABLATION ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = [ 0.5 ]
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### DATA PATH ##################################
DATA_PATH = plot_path( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### TABLE: ITERATIONS TO REACH RELATIVE GAP (UNTRAINED FISTA) ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
#### COMPUTING QUANTILES OF (F-F*)/F*
pm.compute_quantiles_rel_opt_gap( F_dicts, DATASET, TEST_MODE, PROBLEM, TEST_DATASET_RATIO )
#### PRINTING TABLES
if len(PARAMS_FISTA_UNTRAINED) > 0:
    n_fista = len(PARAMS_FISTA_UNTRAINED)
    fista_rel_gap_dicts = F_dicts[1:1+n_fista]                       # F_dicts[0] is ISTA
    inner_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                                PARAMS_FISTA_UNTRAINED, [], [] )[1:1+n_fista]
    pm.iters_to_tol_table( fista_rel_gap_dicts, PARAMS_FISTA_UNTRAINED, tols = [ 1e-2, 1e-3, 1e-5 ],
                           inner_dicts = inner_dicts, table_path = DATA_PATH, table_name = 'iters_to_tol_fista' )

######################### TABLE: ITERATIONS TO REACH RELATIVE GAP (FISTA-DEV) ##################################
if len(FISTA_DEV_LIST) > 0:
    n_fista = len(PARAMS_FISTA_UNTRAINED)
    start = 1 + n_fista                                              # F_dicts: [ISTA, FISTA..., FISTA-DEV..., DeepOpt...]
    fista_dev_rel_gap_dicts = F_dicts[start:start+len(FISTA_DEV_LIST)]
    fista_dev_inner_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                                          PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, [] )[start:start+len(FISTA_DEV_LIST)]
    #### GROUPING BY (train ratio, epochs, (miniter, maxiter), ablation): ONE gamma x alpha TABLE PER GROUP
    groups = {}
    for options, d, d_inner in zip( FISTA_DEV_LIST, fista_dev_rel_gap_dicts, fista_dev_inner_dicts ):
        params, train_dataset_ratio, epochs, minmaxiter, ablation = options
        key = ( train_dataset_ratio, epochs, minmaxiter, ablation )
        groups.setdefault( key, ( [], [], [] ) )
        groups[key][0].append( params )
        groups[key][1].append( d )
        groups[key][2].append( d_inner )
    for key, ( params_list, rel_gap_dicts, inner_dicts ) in groups.items():
        train_dataset_ratio, epochs, minmaxiter, ablation = key
        miniter, maxiter = minmaxiter
        print( f'===== FISTA-DEV: train data={train_dataset_ratio}, epochs={epochs}, N=[{miniter},{maxiter}], ablation={ablation} =====' )
        table_name = f'iters_to_tol_fista_dev_data{train_dataset_ratio}_ep{epochs}_N{miniter}-{maxiter}_abl{ablation}'
        pm.iters_to_tol_table( rel_gap_dicts, params_list, tols = [ 1e-2, 1e-3, 1e-5 ],
                               inner_dicts = inner_dicts, table_path = DATA_PATH, table_name = table_name )

######################### PLOTTING MEDIAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
#### COMPUTING QUANTILES OF (F-F*)/F*
pm.compute_quantiles_rel_opt_gap( F_dicts, DATASET, TEST_MODE, PROBLEM, TEST_DATASET_RATIO )
#### STACKING O(1/k**2)
iters = F_dicts[0].get('data').shape[1]
F_dicts = [
    dict(
        center_line = 6000/(1 + np.arange(iters))**2,
        lower_line = 6000/(1 + np.arange(iters))**2,
        upper_line = 6000/(1 + np.arange(iters))**2,
        label = '$O(1/k^2)$',
        ls = ':',
        c = 'k'
    )
] + F_dicts
#### STACKING THE X-COORDINATE
F_dicts = [
    dict(
        center_line = 1 + np.arange(iters),
        lower_line = 1 + np.arange(iters),
        upper_line = 1 + np.arange(iters),
        label = '$k+1$',
        ls = None,
        c = None
    )
] + F_dicts
#### PLOTTING
optgap_fig = pm.plot_data( F_dicts,
                           ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$',
                           fig_path = DATA_PATH, fig_name = 'optgap' )
#### ADDING MARKS AT THE X-AXIS WHERE THE PROXIMAL OPERATOR SOLVER STARTS TO FAIL
flag_dicts = pm.load_data( 'flag_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                           PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
pm.add_marks( flag_dicts, optgap_fig, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

######################### PLOTTING MEDIAN (F-F*)/F* -- only trained iterations ##################################
#### COPYING (F-F*)/F* DICTS AND UPDATING DATA
F_dicts_zoom = [ d.copy() for d in F_dicts ]
for i,d in enumerate(F_dicts_zoom):
    d.update(
        {
            'center_line':F_dicts_zoom[i].get('center_line')[:max(MAXITER)+1],
            'upper_line':F_dicts_zoom[i].get('upper_line')[:max(MAXITER)+1],
            'lower_line':F_dicts_zoom[i].get('lower_line')[:max(MAXITER)+1]
        } 
    )
#### PLOTTING
pm.plot_data( F_dicts_zoom,
              ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$',
              fig_path = DATA_PATH, fig_name = 'optgapzoom' )

######################### TRAINED ALGORITHMS' MEDIAN VALIDATION LOSS AND PENALTY ##################################
if len(FISTA_DEV_LIST) > 0 or len(DEEPOPT_LIST) > 0:
    val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                                DATE, FISTA_DEV_LIST, DEEPOPT_LIST )
    pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

######################### PLOTTING EXTRA INFORMATION ##################################
if not os.path.exists(os.path.join( DATA_PATH, 'extra' )):
    os.makedirs(os.path.join( DATA_PATH, 'extra' ))

######################### PLOTTING MEDIAN STEPSIZE ##################################
#### LOADING STEPSIZE DATA
stepsize_dicts = pm.load_data( 'stepsize_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                                PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
if len(stepsize_dicts) > 0:
    #### COMPUTING QUANTILES OF STEPSIZE (currently all stepsizes in the batch are the same)
    pm.compute_quantiles( stepsize_dicts )
    #### STACKING THE X-COORDINATE
    iters = stepsize_dicts[0].get('data').shape[1]
    stepsize_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k+1$',
            ls = None,
            c = None
        )
    ] + stepsize_dicts
    #### PLOTTING
    pm.plot_data( stepsize_dicts,
                ylabel = r'$\rho_k$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'stepsize' )

######################### PLOTTING MEDIAN EPS ##################################
#### LOADING EPS DATA
eps_dicts = pm.load_data( 'eps_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, [] )
if len(eps_dicts) > 0:
    #### COMPUTING MEDIAN EPS
    pm.compute_quantiles( eps_dicts )
    #### STACKING O(1/k**3)
    iters = eps_dicts[0].get('data').shape[1]
    eps_dicts = [
        dict(
            center_line = 1e+3/(1 + np.arange(iters))**2,
            lower_line = 1e+3/(1 + np.arange(iters))**2,
            upper_line = 1e+3/(1 + np.arange(iters))**2,
            label = '$O(1/k^3)$',
            ls = ':',
            c = 'k'
        )
    ] + eps_dicts
    #### STACKING THE X-COORDINATE
    eps_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k+1$',
            ls = None,
            c = None
        )
    ] + eps_dicts
    #### PLOTTING
    pm.plot_data( eps_dicts,
                ylabel = r'$\epsilon_k$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'eps' )

######################### PLOTTING MEDIAN INNER ITERS ##################################
#### LOADING INNER ITERS DATA
inner_iter_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, 
                                 PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
if len(inner_iter_dicts) > 0:
    #### COMPUTING QUANTILES
    pm.compute_quantiles( inner_iter_dicts )
    #### STACKING O(k**2)
    iters = inner_iter_dicts[0].get('data').shape[1]
    inner_iter_dicts = [
        dict(
            center_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            lower_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            upper_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            label = '$O(k^2)$',
            ls = ':',
            c = 'k'
        )
    ] + inner_iter_dicts
    #### STACKING THE X-COORDINATE
    inner_iter_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k$',
            ls = None,
            c = None
        )
    ] + inner_iter_dicts
    #### PLOTTING
    pm.plot_data( inner_iter_dicts,
                ylabel = r'$\text{inner\_iters}_{k}$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'inner_iters' )
    
######################### PLOTTING MEDIAN CUMULATIVE INNER ITERS ##################################
#### LOADING INNER ITERS DATA
inner_iter_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, 
                                 PARAMS_FISTA_UNTRAINED, FISTA_DEV_LIST, DEEPOPT_LIST )
if len(inner_iter_dicts) > 0:
    #### COMPUTING THE COMULATIVE INNER ITERS
    cum_inner_iter_dicts = [ d.copy() for d in inner_iter_dicts ]
    for i,d in enumerate(cum_inner_iter_dicts):
        d.update( { 'data': np.cumsum( inner_iter_dicts[i].get('data'), axis = 1 ) } )
    #### COMPUTING MEDIAN VALUE
    pm.compute_quantiles( cum_inner_iter_dicts )
    #### STACKING THE X-COORDINATE
    iters = cum_inner_iter_dicts[0].get('data').shape[1]
    cum_inner_iter_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k$',
            ls = None,
            c = None
        )
    ] + cum_inner_iter_dicts
    #### PLOTTING
    pm.plot_data( cum_inner_iter_dicts,
                  ylabel = r'$\text{cum\_inner\_iters}_{k}$',
                  fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'cum_inner_iters' )

print( 'All data was saved to', DATA_PATH )
