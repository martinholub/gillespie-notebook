from logger_utils import initialize_logger, teardown_logger, has_handlers
logger = initialize_logger()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numba
from decorate import func_args
import ssa_utils

ssa_utils.create_result_folders()
    
#@numba.jitclass(mp_spec)
# Numba does not provide support for array of strings.
# This limits usage of `nopython` mode to functions that dont use the custom class
class modelParameters:
    """Model Parameters for the Biological system
    
    Parameters
    ---------------
    k - stochastic reaction constants,
    t_end - length of simulation, 
    q0 - initial state, 
    nu - matrix of state change vectors
    psi - matrix of molecularity indices
    dist - list of char vectors, which copy number distribution is expected 
    
    Notes
    ------------
    """
    
    def __init__(self,
                k, nu, psi = np.empty(0), q0 = np.empty(0), t_end = 200,
                names = [], dist = []):
        # Stochastic reaction constants
        self.k = k.astype(np.float32)
        # State change matrix
        self.nu = nu.astype(np.int32)
        # Reactand molecularity matrix
        if not psi.size:
            psi = self.nu;
        self.psi = psi.astype(np.int32)
        # Initial state (vector of number of molecules)
        if not q0.size:
            q0 = np.zeros(nu.shape[1], np.int32)
        self.q0 = q0
        # Total tick time
        self.t_end = np.int32(t_end)
        # Species names
        if len(names) == 0:
            names = ["Species {}".format(n) for n in range(q0.shape[0])]
        self.names = names
        # Distributions to plot
        if len(dist) == 0:
            dist = ['none'] * np.int(q0.shape[0])
        self.dist = dist
        
        assert self.nu.shape == self.psi.shape
        #assert self.k.shape[0] == self.nu.shape[0]
        assert (self.k > 0).all()
        assert self.t_end > 0
        assert (self.q0 >= 0).all()
        assert len(dist) == len(names)

@numba.jit(nopython = True)
def get_propensities(rates, populations, psi):
    """Return propensities given rates and populations
    
    Computes the value of propensity for all reactions. Generalizes to various systems
    
    Parameters
    ----------
    rates: array, vector of stochastic reaction constants
    populations: array, vector of number of molecules of each species 
    
    Returns
    ---------------
    propensities: array, vector of propensiites for all reactions
    total_propensity: float, sum of all propensiites
    """
    
    # See model description  
    propensities = np.empty(psi.shape[0], dtype = np.float32)
    for r in range(psi.shape[0]):
        temp = populations ** psi[r,:]
        propensities[r] = temp.prod() * rates[r] 
        
    total_propensity = np.float32(propensities.sum())
    return propensities, total_propensity

@numba.jit(nopython=True)
def draw_ssa(propensities, total_prop):
    """Draw random samples according to prescriptions of SSA
    
    Parameters
    -------
    propensities: array, propensities of all reactions
    total_prop: float, total propensity
    
    Returns
    ---------
    idx: int, index into the state change matrix to determine updated state
    tau: float, time step
    """
    
    xi1 = np.random.uniform(0,1)
    prop_rank = np.sort(np.cumsum(propensities)/total_prop)
    #idx = np.searchsorted(prop_rank, xi1) # works the same way
    idx = np.int32(np.argmax(prop_rank > xi1))
    
    xi2 =  np.random.uniform(0,1)
    tau = np.float32(1/total_prop * np.log(1/(1-xi2)))
    return idx, tau

@func_args(logger)
def ssa_routine(k, t_end, q0, nu, psi):
    """ Stochastic Simulation Algorithm

    Performs sampling from distribution governed by chemical master equation.

    Parameters
    --------------
    k: array, vector stochastic reaction constants
    t_end: int, number of steps for the simulation
    q0: array, initial number of molecules for each species
    nu: ndarray, stochiometric matrix, with state change vectors in columns
    psi: ndarray, reactant molecularity matrix
    
    Returns
    --------------
    Q: array, all states q at times t
    time: array, vector of times t
    tot_props: array, total propensities at times t
    is_success: bool, flag indicating if simulation ran to completion
    """
    is_success = 1
    time_int_tracker = (t_end/5)
    time_int_incr = (t_end/5)
    counter = 0
    keep_every = 1 #int(len(k)*t_end / 3)
    
    #Initialize arrays to store states, tick times and total_prop values  
    time = []
    t = 0
    time.append(t)
    
    Q = []
    state = q0
    Q.append(state)
    
    tot_props = []
    tot_props.append(get_propensities(k, state, psi)[1])
    
    while t < t_end:
        # Calculate propensities for current state
        propensities, total_prop = get_propensities(k, state, psi)
        
        if total_prop <= 0:
            ## DEBUG
            logger.error("Propensities negative, path will be skipped.")
            is_success = 0
            break
        
        # Draw random samples for update of state and time step
        idx, tau = draw_ssa(propensities, total_prop)
        if tau <= 0:
            ## DEBUG
            logger.error("Backward stepping in time now allowed. Skipping path.")
            is_success = 0
            break
            
        ## Find random change in state - see theory
        state, is_valid = validate_reaction(state, nu[idx, :])
        if not is_valid:
            ## DEBUG
            logger.error("Encountered invalid reaction. Skipping path.\n state:{}\n nu:{}\n t:{}".format(state, nu[idx,:], t))
            is_success = 0
            break
            
        ## Find random time step size - see theory
        t = t + tau
        if t > time_int_tracker:
            ## DEBUG
            # TODO: Running average of tau to predict time till end.
            logger.info("Relative time: {:.2f}.".format(t/t_end))
            time_int_tracker = t + time_int_incr
        
        # Append values to arrays
        if ((counter % keep_every) == 0):
            time.append(t)
            Q.append(state)
            tot_props.append(total_prop)
        
        counter += 1 
        
    # Add singleton dimension for easier concatenation later
    Q = np.asarray(Q)
    Q = np.reshape(Q, Q.shape+(1,))
    time = np.asarray(time)
    time = np.reshape(time, time.shape+(1,))
    tot_props = np.asarray(tot_props)
    tot_props = np.reshape(tot_props, tot_props.shape+(1,))
    
    return Q, time, tot_props, is_success

def validate_reaction(old_state, state_change):
    """ Acept/reject reaction
    
    Reaction is rejected if it leads to fractional or negative number of species, otherwise is accepted. The main use is for debugging.
    
    Parameters:
    -----------
    old_state: array-like, vector of copy numbers at time t
    state_change: array-like, state(t+1)=state(t) + state_change
    """
    
    temp_state = old_state + state_change
    is_bad = (temp_state < 0).any() or (temp_state % 1 != 0).any()
    if is_bad:
        # Reject reaction
        new_state = old_state
        ret_val = 0
    else:
        new_state = temp_state
        ret_val = 1
        
    return new_state, ret_val

def generate_paths(params, num_paths, do_save = 1, adj = "pad"):
    """Generate sample state trajectories
    
    Iteratively runs Stochastic Simulation Algorithm to obtain `num_paths` samples.
    
    Parameters
    -----------
    params: class modelParameters defining  k - stochastic reaction constants,
                                            t_end - length of simulation, 
                                            q0 - initial state, 
                                            nu - state change vector
                                            psi- reactand molecularity matrix
    num_paths: int, number of paths to generate
    
    Returns
    -----------
    paths_adj: ndarray, paths adjusted (trim/pad) to common length
    times_adj ndarray, ticks adjsuted (trim/pad) to common length
    paths: list, arrays of paths for all species
    times: list, arrays of ticks for all paths
    """

    paths = []
    times = []
    props = []
                                
    for i in range(num_paths):
        logger.info("Start of simulation for path # {} out of {}.".format(i+1, num_paths))
        Q, time, tot_props, isOK = ssa_routine( params.k, params.t_end, params.q0,
                                                params.nu, params.psi)
        if not isOK:
            continue
        paths.append(Q)
        times.append(time)
        props.append(tot_props)
        
    
    # Either trim or nan-pad to common length
    if adj.startswith("p"):
        paths_adj, times_adj, props_adj = ssa_utils.pad_paths(paths, times, props)
    elif adj.startswith("t"):
        paths_adj, times_adj, props_adj = \
                                        ssa_utils.trim_paths(paths, times, props)
    else:
        logger.warning("Invalid `adj`: `{}`, using `pad`.".format(adj))
        paths_adj, times_adj, props_adj = pad_paths(paths, times, props)
            
    if do_save:
        ssa_utils.save_data({"Qs": paths_adj, "times": times_adj,
                            "tot_props": props_adj})
    
    return paths_adj, times_adj, props_adj
      
def describe_paths(Qs, times, names = []):
    """Calculates 1st and 2nd order statistics
    
    Computes mean and variance, aggregating information from all realizations of the process. Prepares mean evolution array over all processes for plotting.
    
    Parameters
    ------------
    Qs: ndarray, shape (min_ticks x species x paths)
    times: ndarray, shape (min_ticks x paths)
    names: list, species names as character vectors
    
    Returns
    ----------
    df_stats: DataFrame, mean, std and CV for all species
    df_plots: DataFrame, mean evolution and corresponding ticktimes for all species
    """
    n_species = Qs.shape[1]
    is_list = 0
    
    if not names:
        names = ["Species {}".format(n) for n in range(n_species)]
    
    logger.info("Calculating statistics for all species.")
    stats = {}; plots = {};

    for i, name in enumerate(names):
        Q = Qs[:, i, :]
        idx = ssa_utils.get_transient_idx(Q)
        
        samp_mean = np.nanmean(Q[idx:,:])
        samp_std = np.nanstd(Q[idx:, :])
                
        stats[name] = { "mean": samp_mean,
                        "var": samp_std**2,
                        "CV": samp_std/samp_mean if samp_mean != 0 else np.nan }
        
        Q_mean, ticks = ssa_utils.calculate_rolling_mean([Q, times])
        plots[name] = { "mean": Q_mean,
                        "ticks": ticks,
                        }
    
    df_stats = pd.DataFrame.from_dict(stats, orient = "index").reindex(names)
    df_plots = pd.DataFrame.from_dict(plots, orient = "index").reindex(names)
    return df_stats, df_plots

def plot_paths(Qs, times, means, ticks):
    """Plots sample paths, their means and marginal distribution estimates
    
    Plots all realizations of the process and mean evolution in time. Additionally
    calculates and plots histogram as estimate of marginal PDF w.r.t. time.
    
    Parameters
    ---------------
    Q: ndarray, states for all species over all ticks for all paths
    time: ndarray, tick times for all paths
    means: array, mean evolution for all species
    ticks: array, tick times corresponding to values in `means`
    
    Returns
    ---------------
    fig, ax : this figure and array of axes
    """
    plt.close()
    
    names = means.index.tolist()
    n_species = Qs.shape[1]
    n_paths = Qs.shape[2]
    
    # Naively find index beyond which stationarity can be assumed
    min_idx = ssa_utils.get_transient_idx(Qs)
    
    fig, ax = plt.subplots( n_species, 2, figsize=(5*2, 4*n_species), 
                            gridspec_kw = {'width_ratios':[5, 2]},
                            squeeze = False)
                
    for jj, name in zip(range(n_species), names):
        for pt in range(n_paths): # Plot all paths
            mask_good = np.isfinite(times[:, pt]) # be explicit about nan
            ax[jj, 0].plot( times[mask_good, pt], Qs[mask_good, jj, pt],
                            "b-", lw = 1, alpha = 0.25)
        
        if n_paths > 1:
        # Plot mean evolution
            ax[jj, 0].plot( ticks[jj], means[jj], "r-", lw = 3,
                    alpha = 0.8, label = "mean evolution, n={}".format(n_paths))
            ax[jj, 0].legend()
        
        ax[jj, 0].set_ylabel(r'# of {}s'.format(name))
        # Plot location where transient trimming happens
        ax[jj, 0].vlines(ticks[jj][min_idx], *ax[jj,0].get_ylim(),
                        colors = "g", alpha = 0.2, lw = 1)
        ax[jj, 0].set_xlabel(r'Time (ticks)')
        
        # Plot histogram of marginal distribution
        data = Qs[min_idx:, jj,:].flatten()
        
        # get rid of nans
        data = data[np.isfinite(data)]
        nbins = np.int(min(ssa_utils.get_num_bins(data), data.max()-data.min()))
        if nbins > 5:
            y,_,_ = ax[jj, 1].hist( data, bins = nbins, normed = True, rwidth = 1,
                                    color = 'b', alpha = 0.3, lw = 0,
                                    label = r"{} bins".format(nbins),
                                    orientation = "horizontal")
            
            # Plot CDF
            #ssa_utils.add_empirical_cdf(Qs[min_idx:, jj,:].flatten(),  ax[jj, 1])
            ssa_utils.add_empirical_cdf(data,  ax[jj, 1])
            
            ax[jj, 1].legend()
            ax[jj, 1].set_ylim(ax[jj,0].get_ylim())
            ax[jj, 1].set_xlabel(r'Density', color = "b")
            ax[jj, 1].tick_params('x', colors='b')
        
        else:
            #ax[jj, 1].get_xaxis().set_visible(False)
            #ax[jj, 1].get_yaxis().set_visible(False)
            ax[jj, 1].axis("off")
                        
    fig.tight_layout() # get nice spacing between plots
    #plt.show()
    
    return fig, ax
################################################################################    
def plot_stats(times, tot_props):
    """Plots simulation statistics
    
    Plots distribution of $\tau$ and evolution of total propensity over time for
    all paths. The former should be chi2 (k=1) the latter should reach steady state.
    Additionally, saves the figure to file.
    
    Parameters
    -------------
    times: ndarray, ticks x paths
    tot_props: ndarray, ticks x paths
    
    """
    # Naively find index beyond which stationarity can be assumed
    min_idx = ssa_utils.get_transient_idx(tot_props)
    times = times[min_idx:,:]
    tot_props = tot_props[min_idx:,:]
    n_paths = times.shape[1]
    
    # Get deltas in time, flatten, remove nans (due to adjust. to common length)
    taus = np.diff(times, axis = 0)
    taus = taus.flatten()
    taus = taus[np.isfinite(taus)]
        
    fig, ax = plt.subplots( 2, 1, figsize=(3.5*2, 3*2))
    
    nbins = ssa_utils.get_num_bins(taus)
    y,x,_ = ax[0].hist( taus, bins = nbins, normed = True, 
                                color = 'b', alpha = 0.3, lw = 0,
                                label = r"Histogram, {} bins".format(nbins),
                                rwidth = 1)
                                
    ax[0].set_xlabel(r'$\Delta t = \tau$')
    ax[0].set_ylabel(r'Density')
    ax[0].legend()
    
    # Plot total propensity evolutions
    ## Remove nans (due to common length adjustment)
    for pt in range(n_paths):    
        mask_good = np.isfinite(times[:, pt])
        ax[1].plot( times[mask_good,pt], tot_props[mask_good, pt], 
                    "b-", lw = 1, alpha = 0.25)
    
    # Calculate rolling mean over flattened and sorted propensities
    if n_paths > 1:
        tot_props_mean, ticks = ssa_utils.calculate_rolling_mean([tot_props, times])
        
        ax[1].plot( ticks, tot_props_mean, "r-", lw = 3, alpha = 0.8,
                    label = "Mean total propensity, n={}".format(n_paths))
        ax[1].set_xlabel(r'Time (ticks)')
        ax[1].set_ylabel(r'Total propensity')
        ax[1].legend()
    
    fig.tight_layout()
    ssa_utils.save_figure(fig, name = "stats")
    
    
################################################################################ 
def simulate(params, n_paths, do_load = 0, do_save = 1):
    """ Wrapper for simulation
    """    
    if do_load:
        Qs, times, tot_props = ssa_utils.load_data()
    else:
        logger.info("Starting simulation with n_paths={}".format(n_paths))
        Qs, times, tot_props = generate_paths(params, n_paths, do_save)
        
        
    df_stats, df_plots = describe_paths(Qs, times, params.names)
    display(df_stats)
    logger.info("Simulation finished.")
    return Qs, times, df_stats, df_plots, tot_props

def visualize(Qs, times, df_stats, df_plots, tot_props, params):
    """ Wrapper for visualization
    """
    # Downsample results --- speed up plotting
    Qs, times, df_plots, tot_props = \
                ssa_utils.downsample_results([Qs, times, df_plots, tot_props])
    # Don't plot if constant throughout
    idx = ssa_utils.filter_constant(df_stats["var"].values)
    
    # Plot individual paths and their means for all species
    fig, ax = plot_paths(   Qs[:,idx,:], times,
                            df_plots["mean"][idx], df_plots["ticks"][idx])
    
    # Add summary plots on side, if requested
    pid = 0
    for jj, dst in enumerate(params.dist):
        if idx[jj] == False or dst.startswith("no"):
            continue
        elif dst.startswith("po"):
            ssa_utils.add_poisson(ax[pid, 1], params.k[jj], Qs[:,jj,:])
            pid += 1 
        elif dst.startswith("ga"):
            ssa_utils.add_gaussian(ax[pid, 1],  df_stats["mean"][params.names[jj]],
                        df_stats["var"][params.names[jj]], Qs[:,jj,:])
             # Fit Kernel density estimate
            ssa_utils.add_kde_estimate(ax[pid, 1], Qs[:, jj,:])
            pid += 1
       
    ssa_utils.save_figure(fig)
    
    # Create and save plot of additional info on taus and total props.
    plot_stats(times, tot_props)
   
def main(params, n_paths, do_load = 0, do_save = 1):
    """ Wrapper for whole simulation
    
    Note:
    -----------
    if do_load=1, do_save is not effective
    """
    logger = initialize_logger()
    try:
        Qs, times, df_stats, df_plots, tot_props = \
                                    simulate(params, n_paths, do_load, do_save)
        visualize(Qs, times, df_stats, df_plots, tot_props, params)
        logger.info("Program execution finished.")
    finally:
        teardown_logger(logger)
    