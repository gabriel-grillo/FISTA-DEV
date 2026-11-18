"""Estimates the accuracy of the approximation F* of the optimal value, and the threshold TAU below which the relative
optimality gaps are not shown.

In paper_results.py, F_b^* is the smallest value attained by all the methods and by the reference run saved in
data/<dataset>/F_ref/<problem>/<mode>, namely FISTA with alpha = 0 and gamma = 2 after 500 iterations (see
traintest.py). Since this run is still decreasing at its last iterations, F_b^* overestimates the optimal value
F_b^opt, and the plotted relative gaps (F_b(x_k) - F_b^*)/F_b^* underestimate the true ones by

    eps_b = (F_b^* - F_b^opt) / F_b^*.

A relative gap g is then measured with relative error eps_b / g, and gaps of the order of eps_b are not meaningful.

For each problem, this script computes the relative gap of the reference run with respect to its own smallest value,

    g_b(k) = ( F_b(k) - F_b^* ) / F_b^*,

and prints:
    1. the quantiles of g(k) over the problems at selected iterations, which show that the run still decreases at a
       regular rate at its last iterations;
    2. the decrease factor of g over windows of W iterations, q(k) = g(k) / g(k + W);
    3. an estimate of eps_b. In the last iterations, the gaps are only a few units in the last place of F, which is
       computed in single precision, so that they cannot be used directly (the decrease factors in item 2 become
       ratios of small integers). Thus, the decrease rate rho of the gap is fitted (least squares on log g) in a window
       [K1, K2] in which the gaps are well resolved, and, since F_b^* is approximately F_b(end), the true gap at the
       end of the run is extrapolated as
           eps_b ~ g(K2) * rho^(end - K2);
    4. the resolution of F in single precision, about eps_float32 relative to F, which also limits the accuracy of
       the plotted gaps;
    5. the threshold TAU for which the relative error (eps_b + eps_float32)/g of every plotted gap g >= TAU is at
       most 10%, 25%, and 50% for 90% of the problems.

Run from the 'code' folder (as the other scripts):

    python3 estimate_tau.py
    python3 estimate_tau.py --mode test --k1 200 --k2 300
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import pandas as pd


def load_F_ref( dataset, problem, mode ):
    """Returns the saved reference values as an array of shape [N, MAXITER+1] (one row per problem)."""
    import tensorflow as tf
    path = os.path.join( os.getcwd(), 'data', dataset, 'F_ref', problem, mode )
    if not os.path.exists( path ):
        raise FileNotFoundError( f'{path} does not exist. Run the experiments once to generate F_ref.' )
    ds = tf.data.Dataset.load( path )
    return np.stack( [ h.numpy() for h in ds ] ).astype( np.float64 )


def relative_gap( F ):
    """(F - F*) / F*, with F* the smallest value of each row (computed in double precision)."""
    F_star = F.min( axis = 1, keepdims = True )
    return ( F - F_star ) / F_star


def quantiles( values, q = ( 0.25, 0.5, 0.75, 0.9 ) ):
    return pd.Series( np.nanquantile( values, q ), index = [ f'q{int(100*p)}' for p in q ] )


def estimate_eps( rel_gap, k1, k2 ):
    """Estimate of eps_b = (F_b^* - F_b^opt)/F_b^* for each problem: the decrease rate rho of the gap is fitted by
    least squares on log g(k), k in [k1, k2], and the gap is extrapolated from k2 to the end of the run."""
    last = rel_gap.shape[1] - 1
    k = np.arange( k1, k2 + 1 )
    eps = np.zeros( rel_gap.shape[0] )
    # fixed-point iterations: the true gap is g + eps, so the fit is repeated with the corrected gaps
    for _ in range( 20 ):
        log_g = np.log( np.maximum( rel_gap[:, k1:k2+1] + eps[:, None], np.finfo( np.float64 ).tiny ) )
        slope = np.polyfit( k, log_g.T, 1 )[0]            # log(rho) for each problem
        rho = np.exp( slope )
        eps = np.where( rho < 1, ( rel_gap[:, k2] + eps ) * rho ** ( last - k2 ), 0.0 )
    return np.where( rho < 1, eps, np.nan ), rho


def main():
    parser = argparse.ArgumentParser( description = 'Estimates the accuracy of F* and the threshold TAU.' )
    parser.add_argument( '--dataset', default = 'mayo_clinic_512' )
    parser.add_argument( '--problem', default = 'lstv' )
    parser.add_argument( '--mode',    default = 'val', choices = [ 'train', 'val', 'test' ] )
    parser.add_argument( '--window',  type = int, default = 50, help = 'number of iterations of the windows W' )
    parser.add_argument( '--k1',      type = int, default = 200, help = 'start of the window used to fit the decrease rate' )
    parser.add_argument( '--k2',      type = int, default = 300, help = 'end of the window used to fit the decrease rate' )
    args = parser.parse_args()

    F = load_F_ref( args.dataset, args.problem, args.mode )
    rel_gap = relative_gap( F )
    n_problems, n_values = rel_gap.shape
    last, W = n_values - 1, args.window
    print( f'\n{args.dataset} / {args.problem} / {args.mode}: {n_problems} problems, reference run with {last} iterations\n' )

    #### 1. Relative gap with respect to the smallest value of the run, at selected iterations
    iterations = [ k for k in [ 1, 10, 50, 100, 200, 300, 400 ] if k < last ] + list( range( last - 2 * W, last + 1, W // 2 ) )
    table = pd.DataFrame( { k: quantiles( rel_gap[:, k] ) for k in sorted( set( iterations ) ) } ).T
    table.index.name = 'k'
    print( '1. Relative gap of the reference run with respect to its smallest value:' )
    print( table.to_string( float_format = '%.2e' ), '\n' )

    #### 2. Decrease factor over windows of W iterations
    starts = list( range( max( 0, last - 6 * W ), last - W + 1, W ) )
    with np.errstate( divide = 'ignore', invalid = 'ignore' ):
        table = pd.DataFrame( { f'{k}->{k+W}': quantiles( np.where( rel_gap[:, k + W] > 0, rel_gap[:, k] / rel_gap[:, k + W], np.nan ) )
                                for k in starts } ).T
    print( f'2. Decrease factor of the relative gap over windows of {W} iterations (the last ones are affected by F*):' )
    print( table.to_string( float_format = '%.2f' ), '\n' )

    #### 3. Estimate of the relative error of F*
    eps, rho = estimate_eps( rel_gap, args.k1, args.k2 )
    print( f'3. Estimated relative error of F*, extrapolating the decrease rate fitted in [{args.k1}, {args.k2}] to iteration {last}:' )
    print( f'   decrease factor over {W} iterations:  ' + ',  '.join( f'{i} = {v:.2f}' for i, v in quantiles( rho ** -W ).items() ) )
    print( '   eps: ' + ',  '.join( f'{i} = {v:.2e}' for i, v in quantiles( eps ).items() ) )
    print( f'   problems in which the gap is not decreasing in the window: {np.sum( np.isnan( eps ) )}\n' )

    #### 4. Resolution of F in single precision
    eps32 = np.finfo( np.float32 ).eps
    print( f'4. Relative resolution of F in single precision: {eps32:.2e} (gaps of a few times this value are not resolved)\n' )

    #### 5. Threshold TAU for a given relative error of the plotted gaps
    err_90 = np.nanquantile( eps, 0.9 ) + eps32
    print( '5. Threshold TAU such that every plotted gap g >= TAU has relative error (eps + eps_float32)/g at most:' )
    for err in [ 0.10, 0.25, 0.50 ]:
        print( f'   {100*err:.0f}% (for 90% of the problems):  TAU = {err_90 / err:.1e}' )
    print()


if __name__ == '__main__':
    main()