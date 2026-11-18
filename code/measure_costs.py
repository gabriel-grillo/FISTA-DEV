"""Measures the relative cost of the operations of the outer and of the inner (proxTV) iterations.

The cost of an outer iteration of ISTA, FISTA, FISTA-DEV, and DeepOpt is dominated by one evaluation of the ray
transform A and one of its transpose A^T (the gradient of f(x) = 0.5*||Ax - b||^2), while each inner iteration of
proxTV evaluates the discrete gradient D and its transpose D^T, the projections onto the box (primal) and onto the
dual feasible set, and the duality gap. This script measures, on the GPU and for given batch sizes:

    1. the time of each of these operations separately;
    2. the time of one outer gradient evaluation, A^T(Ax - b);
    3. the time of one inner iteration of proxTV, measured with the actual implementation, as
           ( time(proxTV with N+1 inner iterations) - time(proxTV with 1 inner iteration) ) / N;
    4. the ratio r = (time of one inner iteration) / (time of one gradient evaluation);
    5. the time of one forward pass (inference) of each network of FISTA-DEV (d, e, v) and of DeepOpt (dx1, dx2),
       relative to one gradient evaluation. The networks are only applied in the first K iterations.

With these ratios, the cost of a run up to iteration k can be estimated, in units of gradient evaluations, as

    cost(k) = k + r * (total number of inner iterations up to k) + r_net * min(k, K),

in which r_net is the cost of the networks applied in each of the first K iterations (zero for ISTA and FISTA).

which only requires the histories already saved by the tests (inner_iter_hist), without measuring wall-clock time.
The evaluation of F used to save the history (one extra application of A per outer iteration) is reported
separately, since it is not part of the methods.

Run from the 'code' folder (as the other scripts):

    python3 measure_costs.py
    python3 measure_costs.py --batch-sizes 1 32 --repeats 50
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import time
from math import inf

import numpy as np
import pandas as pd
import tensorflow as tf

from operators import reconstruction_operators, torch_linear_op_as_tf2_layer
from proxTV import proxTV, reduce_TV, dual_projection
from traintest import network_base_arch

#### Networks of each method: name -> number of input images (see traintest.py and optimization.py)
NETWORKS = { 'FISTA-DEV d': 3, 'FISTA-DEV e': 4, 'FISTA-DEV v': 5, 'DeepOpt dx1': 3, 'DeepOpt dx2': 4 }


def timeit( fn, args, repeats, warmup ):
    """Median wall-clock time (in milliseconds) of fn(*args). fn must return a scalar tensor, whose conversion to
    numpy forces the synchronization with the GPU."""
    for _ in range( warmup ):
        fn( *args ).numpy()
    times = []
    for _ in range( repeats ):
        start = time.perf_counter()
        fn( *args ).numpy()
        times.append( time.perf_counter() - start )
    return 1e3 * np.median( times )


def build_operators( dataset, problem ):
    ray_trafo, ray_trafo_transp, img_shape, sino_shape, _, D, D_adj, D_norm, lam = reconstruction_operators( dataset, problem )
    A = torch_linear_op_as_tf2_layer( ray_trafo, ray_trafo_transp, img_shape, sino_shape, np.float32, name = 'ray_transform' )
    A_adj = torch_linear_op_as_tf2_layer( ray_trafo_transp, ray_trafo, sino_shape, img_shape, np.float32, name = 'ray_transform_adj' )
    return A, A_adj, D, D_adj, tf.constant( D_norm, dtype = tf.float32 ), tf.constant( lam, dtype = tf.float32 ), img_shape


def measure( batch_size, A, A_adj, D, D_adj, D_norm, lam, img_shape, n_inner, repeats, warmup ):
    rng = np.random.default_rng( 0 )
    x_shape = ( batch_size, ) + img_shape + ( 1, )
    x = tf.constant( rng.random( x_shape ), dtype = tf.float32 )
    b = A( tf.constant( rng.random( x_shape ), dtype = tf.float32 ) )
    u = tf.constant( rng.standard_normal( ( batch_size, 2 ) + img_shape + ( 1, ) ), dtype = tf.float32 )
    rho = tf.constant( 0.9, dtype = tf.float32 )
    eps = tf.fill( [ batch_size ], -1.0 )   # negative tolerance: the inner solver never stops before MAXITER

    #### Outer operations
    @tf.function
    def op_A( x ):
        return tf.reduce_sum( A( x ) )

    @tf.function
    def op_At( b ):
        return tf.reduce_sum( A_adj( b ) )

    @tf.function
    def op_grad( x, b ):
        return tf.reduce_sum( A_adj( A( x ) - b ) )

    @tf.function
    def op_F( x, b ):
        r = A( x ) - b
        return tf.reduce_sum( 0.5 * tf.reduce_sum( r**2, axis = [1,2,3] ) + lam * reduce_TV( D( x ) ) )

    #### Inner operations
    @tf.function
    def op_D( x ):
        return tf.reduce_sum( D( x ) )

    @tf.function
    def op_Dt( u ):
        return tf.reduce_sum( D_adj( u ) )

    @tf.function
    def op_box( x ):
        return tf.reduce_sum( tf.math.maximum( x, 0.0 ) )

    @tf.function
    def op_dual_proj( u ):
        return tf.reduce_sum( lam * dual_projection( u / lam ) )

    @tf.function
    def op_gap( x, u ):
        grad = D( x )
        return tf.reduce_sum( lam * reduce_TV( grad ) - tf.reduce_sum( x * D_adj( u ), axis = [1,2,3] ) )

    #### One inner iteration of the actual proxTV implementation (by difference)
    def make_prox( maxiter ):
        @tf.function
        def run_prox( z ):
            prox = proxTV( D, D_adj, D_norm, lam, lower = 0.0, upper = inf, MAXITER = maxiter )
            out, _, iters = prox( z, rho, eps )
            return tf.reduce_sum( out ) + 0.0 * tf.cast( tf.reduce_sum( iters ), tf.float32 )
        return run_prox

    t = {}
    t['A']                     = timeit( op_A,  ( x, ), repeats, warmup )
    t['A^T']                   = timeit( op_At, ( b, ), repeats, warmup )
    t['gradient A^T(Ax-b)']    = timeit( op_grad, ( x, b ), repeats, warmup )
    t['F (history only)']      = timeit( op_F, ( x, b ), repeats, warmup )
    t['D']                     = timeit( op_D,  ( x, ), repeats, warmup )
    t['D^T']                   = timeit( op_Dt, ( u, ), repeats, warmup )
    t['box projection']        = timeit( op_box, ( x, ), repeats, warmup )
    t['dual projection']       = timeit( op_dual_proj, ( u, ), repeats, warmup )
    t['duality gap (D, D^T)']  = timeit( op_gap, ( x, u ), repeats, warmup )
    t_prox_1 = timeit( make_prox( 1 ), ( x, ), repeats, warmup )
    t_prox_n = timeit( make_prox( n_inner + 1 ), ( x, ), max( 5, repeats // 10 ), max( 1, warmup // 5 ) )
    t['inner iteration (proxTV)'] = ( t_prox_n - t_prox_1 ) / n_inner

    #### Networks (forward pass, as in the tests)
    for name, n_inputs in NETWORKS.items():
        inputs = [ tf.keras.Input( shape = img_shape + ( 1, ) ) for _ in range( n_inputs ) ]
        model = network_base_arch( inputs, name.replace( ' ', '_' ) )
        imgs = [ tf.constant( rng.standard_normal( x_shape ), dtype = tf.float32 ) for _ in range( n_inputs ) ]
        @tf.function
        def op_net( *imgs ):
            return tf.reduce_sum( model( list( imgs ), training = False ) )
        t[f'network {name}'] = timeit( op_net, imgs, repeats, warmup )
    return t


def main():
    parser = argparse.ArgumentParser( description = 'Measures the relative cost of outer and inner iterations.' )
    parser.add_argument( '--dataset', default = 'mayo_clinic_512' )
    parser.add_argument( '--problem', default = 'lstv' )
    parser.add_argument( '--batch-sizes', type = int, nargs = '+', default = [ 32 ] )
    parser.add_argument( '--n-inner', type = int, default = 100, help = 'inner iterations used to measure one inner iteration' )
    parser.add_argument( '--repeats', type = int, default = 30 )
    parser.add_argument( '--warmup', type = int, default = 5 )
    args = parser.parse_args()

    A, A_adj, D, D_adj, D_norm, lam, img_shape = build_operators( args.dataset, args.problem )

    results = {}
    for batch_size in args.batch_sizes:
        print( f'Measuring with batch size {batch_size}...' )
        results[f'batch {batch_size} (ms)'] = measure( batch_size, A, A_adj, D, D_adj, D_norm, lam, img_shape,
                                                      args.n_inner, args.repeats, args.warmup )
    table = pd.DataFrame( results )
    print( f'\nMedian time per operation ({args.dataset}, {args.problem}):' )
    print( table.to_string( float_format = '%.3f' ), '\n' )

    print( 'Ratios (per batch size):' )
    for col in table.columns:
        t = table[col]
        r = t['inner iteration (proxTV)'] / t['gradient A^T(Ax-b)']
        print( f'  {col.split(" (")[0]}:  r = inner iteration / gradient evaluation = {r:.3f}   '
               f'(1 gradient evaluation = {1/r:.1f} inner iterations);  '
               f'(D + D^T) / (A + A^T) = {( t["D"] + t["D^T"] ) / ( t["A"] + t["A^T"] ):.3f}' )
        grad = t['gradient A^T(Ax-b)']
        nets = { name: t[f'network {name}'] / grad for name in NETWORKS }
        print( '      networks / gradient evaluation: ' + ',  '.join( f'{name}: {v:.4f}' for name, v in nets.items() ) )
        print( f'      r_net FISTA-DEV (d + e + v) = {nets["FISTA-DEV d"] + nets["FISTA-DEV e"] + nets["FISTA-DEV v"]:.4f},  '
               f'r_net DeepOpt (dx1 + dx2) = {nets["DeepOpt dx1"] + nets["DeepOpt dx2"]:.4f}' )
    print( '\nCost model, in units of gradient evaluations:  cost(k) = k + r * (inner iterations up to k) + r_net * min(k, K)' )

    with open( os.path.join( os.getcwd(), f'measure_costs_{args.dataset}_{args.problem}.txt' ), 'w' ) as f:
        f.write( table.to_string( float_format = '%.3f' ) + '\n' )


if __name__ == '__main__':
    main()