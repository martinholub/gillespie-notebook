import numpy as np
from datetime import datetime
from math import factorial
import os
import glob
import logging
import seaborn as sns
import pandas as pd
import warnings

# Share logger between modules - TODO: Create another logger
## Import this module only after creating logger from `ssa_routine`
logger = logging.getLogger("ssa_routine")
    
## Helper functions
def get_transient_idx(data):
    """Naive method to find some index beyond which stationarity assumed.
    
    TODO: Fit parabola to noisy data and find elbow.
    """
    min_idx = np.int(max([10, data.shape[0]/5]))
    return min_idx

def downsample_dataFrame(df, freq = 1000):
    """ Returns DataFrame with values downsampled by freq
    
    Assumes that each entry in dataframe is array that can be adressed by [column, index]. Values are downsampled along 0th dimension.
    """
    freq =np.random.choice([0, 1], df.values[0][0].shape,
                            p = [1-1/freq, 1/freq]).astype(bool)
                            
    df_out = pd.DataFrame(columns = df.columns, index = df.index)
    for c in df.columns:
        for i in df.index:
            if type(freq) == int:
                df_out[c][i] = df[c][i][::freq]
            else:
                df_out[c][i] = df[c][i][freq]
    return df_out
            
def downsample_results(in_list, freq = None):
    """Downsamples arrays and/or dataframes in in_list by freq
    """
    if not freq:
        freq = max(np.int(in_list[0].shape[0]/2500), 1)
    
    idx =np.random.choice([0, 1], in_list[0].shape[0],
                            p = [1-1/freq, 1/freq]).astype(bool)
    out_list = []
    
    freq_df = freq*in_list[0].shape[-1]
    
    for el in in_list:
        if type(el) == pd.DataFrame:
            out_list.append(downsample_dataFrame(el, freq_df))
        else:
            if type(idx) == int:
                out_list.append(el[::idx,:])
            else:
                 out_list.append(el[idx,:])
    
    assert len(in_list) == len(out_list)
    
    return out_list
    
def get_num_bins(data):
    """ Compute optimal number of bins according to Freedman-Diaconis rule
    
    The rule is applied for at least moderate copy numbers, else, number of
    unique values is selected as number of bins.
    
    Notes
    ------------
      https://stats.stackexchange.com/a/862/194926
    """
    
    IQR = np.subtract.reduce(np.percentile(data, [75, 25]))
    num_points = len(data)
    bwidth = 2 * IQR * (num_points**(-1/3))
    unique_vals = np.unique(data)
    if unique_vals.shape[0] > 20:
        # For moderate and high copy numbers
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            np.seterr(divide="warn")
            try:
                num_bins=min(60, 
                            np.arange(data.min(), data.max(), bwidth).shape[0])
            except (ValueError, Warning) as e:
                logger.warning(e)
                num_bins = min(60, unique_vals.shape[0])
                logger.info("Setting number of bins to {} (default).".format(num_bins))
            
    else:
        # For low copy numbers
        num_bins = unique_vals.shape[0]
    return np.int(num_bins)
    
def moving_average(data, win_size = None):
    """ Calculates rolling mean on tuple/list of two arrays: values and times
    
    Parameters
    --------------
    data: array, values to compute moving average on
    win_size: size of window for moving average, default = len(array)/500
    
    Returns
    --------------
    arrays_tuple: 2 ndarrays, rolling mean of flattened and masked inputs
    """
    if win_size is None:
        win_size = max(np.int(len(data)/500), 1)
    
    ret = np.cumsum(data, dtype=np.float)
    ret[win_size:] = ret[win_size:] - ret[:-win_size]
    return ret[win_size - 1:] / win_size

def calculate_rolling_mean(arrays_tuple, win_size = None):
    """ Calculates rolling mean on tuple/list of two arrays: values and times
    
    Parameters
    --------------
    arrays_tuple: tuple or list of exactly two arrays
            Qs: 2D array, values x evolutions
            times: 2D array, values x evolutions
    win_size: size of window for moving average, default = len(array)/500
    
    Returns
    --------------
    arrays_tuple: 2 ndarrays, rolling mean of flattened, masked and sorted inputs
    
    Notes:
    ---------
    Thanks to Stan for tip.
    """
    try:
        Qs, times = arrays_tuple
    except ValueError as e:
        logger.error("Calculate mean expects exactly two arrays, values and times")
        logger.error(e)
        return None
    
    # Flatten, remove nans, sort in time, calculate moving average
    times = times.flatten()
    mask_good = np.isfinite(times)
    times = times[mask_good]
    idxs = np.argsort(times)
    times = np.sort(times)
    times_out = moving_average(times, win_size)
    
    # Flatten, remove nans, sort by time, calculate moving average
    Qs = Qs.flatten()[mask_good]
    Qs = Qs[idxs]
    Qs_out = moving_average(Qs, win_size)
    
    assert Qs_out.shape == times_out.shape
        
    return Qs_out, times_out
        
def empirical_cdf(data, yscale = 1.):
    """Calculates scaled emprical cumulative distribution
    
    The function is optionally scaled down to maximum of corresponding pdf/histogram.
    """
    x = np.sort(data)
    y = 1. * np.arange(len(data)) / (len(data) - 1) * yscale
    return x, y

def add_empirical_cdf(data, ax, yscale = 1.):
    """Adds empirical CDF to existing axes.
    
    Horizontal orientation of axes is assumed
    """
    x, y = empirical_cdf(data, yscale)
    bx = ax.twiny()
    bx.plot(y, x, 'g--', alpha = 0.5, lw = 1,
            label = "emp. CDF")
    bx.tick_params('x', colors='g')
    bx.set_xlabel('CDF', color='g')
    bx.grid(False)
    
def add_poisson(ax, lmbd, Qs):
    """Adds Poisson PDF to existing axes.
    
    Horizontal orientation of axes is assumed
    
    @param lambda: int,  Event rate parameter
    @param ax: plt axis to plot on (assume horzontal)
    @Qs: ndarray, data that is supposed to obey Poisson distr.
    """
    # Naively trim off transient evolution at the beginning of the simulation
    min_idx = get_transient_idx(Qs)
    
    xx = np.arange(0, np.nanmax(Qs[min_idx:,:].flatten())+1)
    poiss =  [(np.exp(-lmbd) * lmbd ** x) / factorial(x) for x in xx]
    
    ax.plot(poiss, xx, 'bo-', alpha = 0.5, lw = 2,
            label = "Poiss. PDF")
    ax.legend()

def add_gaussian(ax, mn, vr, Qs, orientation = "horizontal"):
    """Adds Gaussian PDF to existing axes.
    
    Horizontal orientation of axes is assumed
    
    Parameters
    -----------
    ax: axis, horizontal orientation assumed
    mn: float, mean
    vr: float, variance
    Qs: ndarray: 2d array of species numbers over time
    """
    # Naively trim off transient evolution at the beginning of the simulation
    min_idx = get_transient_idx(Qs)
    
    max_val = np.nanmax(Qs[min_idx:,:])+1
    x = np.linspace(0, max_val, num = 100)
    gauss = np.exp(-(x-mn)**2/(2*vr))/np.sqrt(2*np.pi*vr)
    
    if orientation == "horizontal":
        ax.plot(gauss, x, 'b-',alpha=0.5, lw = 2, label='Gauss. PDF')
    else:
        ax.plot(x, gauss, 'b-',alpha=0.5, lw = 2, label='Gauss. PDF')
    ax.legend()

def add_kde_estimate(ax, data):
    """Estimates distribution of the data using kernel density estimate
    
    Gaussian kernel with `scott` method of bandwith determination is used.
    
    Parameters
    ----------
    ax: axis, axis object to plot on
    data: array, array of copy numbers for single species over all times
    
    Notes
    ------------
    # TODO: make it computationally cheaper
    # TODO: call directly statsmodels.(...).fit and use `adjust` to change bw
    """
    min_idx = get_transient_idx(data)
    data = data[min_idx:, :].flatten()
    data = data[np.isfinite(data)]
    
    kde_pars = {"kernel": "gau", "bw": "scott", 
            "cumulative": False}  
    nbins = min(get_num_bins(data), data.max()-data.min())
    kde_pars.update({"gridsize": nbins})
    
    sns.distplot(   data, ax = ax,
                    hist = False, kde = True, rug = False,
                    vertical = True, kde_kws = kde_pars, label = "KDE")
    ax.legend()
    
def filter_constant(vars):
    """Create T/F index for data with variance vars, F if variance constant.
    """
    return vars != 0
            
def save_figure(fig, name = "ssa"):
    """ Save figure to PDF
    """
    fname = "results~/plots/{}_{}.pdf".format(name, 
                                datetime.now().strftime('%Y%m%d-%H%M%S'))
    kwargs = {  "dpi": 400,
                "orientation": "portrait",
                "papertype": "a2",
                "format": "pdf",
                #"frameon": False,
                #"bbox_inches": "tight"
                }
    try:
        fig.savefig(fname, **kwargs)
    except Exception as e:
        try:
            logger.error(e)
        except NameError: # if logger not defined
            print(e)
        raise e
        
def save_data(kwds, suffix=""):
    """Saves data to numpy .npz format 
    
    Parameters
    --------
    kwds: dict, data as key-value pairs ('name': [values])
    suffix: string, text to append to filename
    """
    time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    parent_dir = "results~/data/"
    fname = parent_dir + "out_" + time_str + suffix
    try:
        np.savez_compressed(fname, **kwds)
        logger.info("Data saved to file {}.".format(fname))
    except Exception as e:
        logger.error(e)
        raise e
        
def load_data(  files = ["Qs", "times", "tot_props"], 
                dir = "results~/data/*.npz"):
    """Loads the newest *.npz file in directory
    
    Parameters
    -----------
    files: list of char vectors, names of variables we expect to load
    dir: string, s`./path/to/files/.*npz`
    """
    file_list = glob.glob(dir)
    latest_file = max(file_list, key=os.path.getctime)
    try:
        retObj = np.load(latest_file)
        logger.info("Loaded data from {}.".format(latest_file))
    except Exception as e:
        logger.error(e)
        raise e
    assert retObj.files == files
    retVals = tuple([retObj[f] for f in files])
        
    return retVals
    
def create_result_folders(parent = "./results~",
                        dirs = ["data", "logs", "plots"]):
    """Creates folder structure to store simulation results (data, plots, logs)
    """
    if not os.path.isdir(parent):
        os.mkdir(parent)
    for di in dirs:
        dir_path = os.path.join(parent, di)
        if os.path.isdir(dir_path):
            continue
        else:
            logger.info("Creating {} in {}".format(di, parent))
            os.mkdir(dir_path)

def pad_paths(Qs, times, props):
    """Pad paths to common maxima; length and reshape
    
    Pad paths to same length given by the longest path. Fill other with nan.
    Additionally, stack to a 3D array of shape (time x n_species x n_paths). Acts only along first dimension.
    
    Parameters
    -----------
    Qs: list of arrays (ticks x species x 1), sample paths
    times: list of arrays (ticks x 1), tick times
    how: str, one of "index" or "time" - how to find common length for later concatenation
         Currently only "index" implemented
    
    Returns
    -----------
    Qs: ndarray, shape (min_ticks x species x paths)
    times: ndarray, shape (min_ticks x paths)
    """

    if type(Qs) == list:
        n_species = Qs[0].shape[1]
        n_paths = len(Qs)
    else:
        logger.error(  "Can trim paths only on list of arrays."+
                        "The provided is {}.".format(type(Qs)))
        return Qs, times, props
        
    max_length = max([q.shape[0] for q in Qs])
    out_list = []
    for i, var in enumerate([Qs, times, props]):
        ax = var[0].ndim - 1
        var=[np.pad(v.astype(np.float64),((0, l-v.shape[0]),)+((0, 0),)*(v.ndim-1),
            mode="constant", constant_values = np.nan) 
            for v,l in zip(var, [max_length]*len(var))]
            
        out_list.append(np.concatenate(var, axis = ax))
        
    return tuple(out_list)
   
def trim_paths(Qs, times, props):
    """Trim paths to common minimal length and reshape
    
    Trims paths to same length given by the shortest path. Discards values beyond.
    Additionally, stack to a 3D array of shape (time x n_species x n_paths). Acts only along first dimension.
    
    Parameters
    -----------
    Qs: list of arrays (ticks x species x 1), sample paths
    times: list of arrays (ticks x 1), tick times
    
    Returns
    -----------
    Qs: ndarray, shape (min_ticks x species x paths)
    times: ndarray, shape (min_ticks x paths)
    
    Notes
    ----------
    By trimming paths in length in indices, you obtain paths in different
    length on time. This may impact your statistics afterwards (e.g. \tau 
    distribution) because you effectively discard periods with different
    behaviour. Padding with nans is prefered.
    """
    if type(Qs) == list:
        n_species = Qs[0].shape[1]
        n_paths = len(Qs)
    else:
        logger.error(  "Can trim paths only on list of arrays."+
                        "The provided is {}.".format(type(Qs)))
        return Qs, times, props
    
    min_length = min([q.shape[0] for q in Qs])
    #min_length = np.int(min_length / 20)
    Qs = [q[-min_length:, :, :] for q in Qs]
    times = [t[-min_length:, :] for t in times]
    props = [p[-min_length:, :] for p in props]
    
    Qs = np.dstack(Qs)
    times = np.squeeze(np.dstack(times))
    props = np.squeeze(np.dstack(props))
    
    return Qs, times, props            
    