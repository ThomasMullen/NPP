"""
You can use scipy.ndimage.convolve1d which allows you to specify an axis argument.
import numpy as np
import scipy
img = np.random.rand(64, 64, 54) #three dimensional image
k1 = np.array([0.114, 0.141, 0.161, 0.168, 0.161, 0.141, 0.114]) #the kernel along the 1st dimension
k2 = k1 #the kernel along the 2nd dimension
k3 = k1 #the kernel along the 3nd dimension
# Convolve over all three axes in a for loop
out = img.copy()
for i, k in enumerate((k1, k2, k3)):
    out = scipy.ndimage.convolve1d(out, k, axis=i)
Work is mostly base upon this method: https://www.biorxiv.org/content/10.1101/061507v2.full
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter, convolve, convolve1d

import ray
import psutil
import time

import utils.hdf5_tools as h5_tools
import src.rigid_motion_correction as rmc

n_cpu = psutil.cpu_count(logical=False)

ray.init(num_cpus=n_cpu)


def create_detrend_filter(filter_size):
    """
    create an inverted hat filter. Middle value has a single value of 1 and all other values have a negative value
    which sums to -1.
    :param filter_size: size of filter length. length will be extended by one if an even length is given.
    :return: 1D numpy array of the hat filter.
    """
    # check filter length is odd
    if filter_size % 2 != 1:
        # otherwise add extra element
        filter_size += 1
    # normalise negative values removing middle value
    hat_filter = np.ones(filter_size) / (1 - filter_size)
    # insert 1 for the middle value
    hat_filter.put(filter_size // 2, 1)
    return hat_filter


def define_sliding_z_thickness(frames, z_thickness=5, axis=1):
    """
    Reshape the data to have an additional dimension so there is a sliding window through each frame. For example, if
    you want to analyse a sub-vol (volume a few planes thick), you can have a dimension equal to the vol thickness and
    the other dimension indexing each roll element.
    :param frames: entire 4D data should be of shape (t, x, y, z)
    :param z_thickness: the number of planes you want in to analyse in your sliding window. Default 5 planes.
    :param axis: the axis you will be rolling through. Default z-axis.
    :return: reshaped numpy array. Example frames = (t=140, x=262, y=710, z=127), thickness = 5, along z-axis: return
    array would be (t=140, x=262, y=710, roll_z=123, ∂z=5)
    """
    return sliding_window_view(frames, z_thickness, axis)


def center_data(Array, **kwargs):
    """
    centre the data with respect to the mean data.
    :param Array: array to apply transformation
    :param Kwargs: arguments in the mean function e.g., specific the axis e.g. axis=0
    :return: transformed array
    Examples
    --------
    >>>> center_data(sub_stack, axis=0)
    """
    return np.subtract(Array, np.mean(Array, **kwargs))


def time_detrend_filter(Array, **kwargs):
    """
    Filter along the time axis to correct for drift.
    :param Array: substack array
    :param kwargs: weights=detrend_filter, axis=0 the axis you would like to apply the filter over
    :return: filter array along a specific axis
    Examples
    --------
    >>> from scipy.ndimage import convolve1d
    >>> detrend_filter = create_detrend_filter(detrend_length=15)
    >>> time_detrend_filter(sub_stack, weights=detrend_filter, axis=0)
    """
    return convolve1d(Array, **kwargs)


def multi_dimensional_smoothing(Array, **kwargs):
    """
    Apply a gaussian filter along multi-dimensioanl array.
    :param Array: multi-dimensional array that wants to be smoothed
    :param kwargs: keyword arguments from gaussian_filter function
    :return: smoothed array
    Example
    -------
    >>>> # specify smoothing along each axis
    >>>> multi_dimensional_smoothing(sub_stack, sigma=[0, 1, 1, 1])
    or
    >>>> # smooth equally along all axis
    >>>> multi_dimensional_smoothing(sub_stack, sigma=1)
    """
    return gaussian_filter(Array, **kwargs)


def frob_norm_squared(Array, **kwargs):
    """
    Find the :math:`||x||^2` frobenius for each element in array olong a defined axis
    :param Array: want to normalise
    :param kwargs: keywor arguments e.g. axis of normalisation
    :return: normalised array
    Example
    -------
    >>>> # specify smoothing along each axis
    >>>> frob_norm_squared(sub_stack, axis=0)
    """
    return np.power(norm(Array, **kwargs), 2)


@ray.remote
def compute_correlation_map(a_sub_stack, gauss_filter_width, detrend_length):
    """
    To find the locations of new sources we compute the “multi-dimensional” correlation of each pixel with the
    surrounding pixels. We define the multi-dimensional correlation between a set of N traces fi (all with mean 0):
    c(f1, f2, .., fn) = ||sum(fi)||^2 / N*sum(||fi||)^2 .
    The weighted correlation map can be calculated efficiently at each pixel by smoothing with a Gaussian kernel
    each dimension of fi.
    :param sub_stack: np array of a sub-volume of frames with dimensions (t, x, y, ∂z)
    :param gauss_filter_width: standard deviation of you gaussian kernel across all dimensions.
    :param detrend_length: frame length for the detrend inverted top-hat filter. Used to account for drift in temporal
    axis and correct for systematic correlations.
    :return: 2D correlation map of a single x-y- slice corresponding to the central slice in ∂z.
    """
    # get thickness of sub-stack
    z_thickness = a_sub_stack.shape[-1]

    # gaussian width 0 along time axis - prevent smoothing in time direction
    # filtered_sub_stack = a_sub_stack
    filtered_sub_stack = gaussian_filter(a_sub_stack, [0, gauss_filter_width, gauss_filter_width, gauss_filter_width])

    # center the mean about zero to account for weighted correlation map
    a_sub_stack = center_data(a_sub_stack, axis=0)
    filtered_sub_stack = center_data(filtered_sub_stack, axis=0)

    # build de-trend filter to correct for long-term drift
    detrend_filter = create_detrend_filter(detrend_length)

    # convolve filter along the time axis
    a_sub_stack = convolve1d(a_sub_stack, detrend_filter, axis=0)
    filtered_sub_stack = convolve1d(filtered_sub_stack, detrend_filter, axis=0)

    # calculate the correlation map

    # calculate the frob norm along time axis
    individual_norm = norm(a_sub_stack, axis=0)
    filtered_individual_norm = gaussian_filter(individual_norm, 1)
    filtered_individual_norm_sqrd = filtered_individual_norm ** 2

    #
    mean_norm_squared = frob_norm_squared(filtered_sub_stack, axis=0)

    # find correlation map for the middle plane
    # select only middle slices and change type to float to handle 0 division
    corr_map = mean_norm_squared[:, :, z_thickness // 2].astype('float64') / \
               filtered_individual_norm_sqrd[:, :, z_thickness // 2].astype('float64')

    return corr_map


def apply_motion_correction(
        parent_path=f'F:\\OptovinScape\\20210628\\tiff_stacks\\20210628_6dpf_elavl3_H2B_fish3_run1'):
    """
    snippect of reduced volume data to create a motion corrected timeseries
    :return:
    """
    #
    # data_name = "vol"
    data_name = "reducedVol"

    # path_dir = f"{parent_path}\\volumes"
    path_dir = f"{parent_path}\\reducedVolumes"

    # n_timesteps = 500
    n_timesteps = h5_tools.compute_total_frames(path_dir, 4, True)

    timeseries = np.stack(
        [np.array(h5_tools.extract_dataset(f'{path_dir}/{file_name:06}.mat', data_name)[:]) for file_name in
         range(n_timesteps)])

    frames_path = f"{parent_path}\\removed_frames.npy"
    frames_exc = np.load(frames_path)
    frames_exc = frames_exc[frames_exc <= n_timesteps]
    timeseries = np.delete(timeseries, frames_exc, axis=0)

    shift_path = f"{parent_path}\\1_motion_shifts.npy"
    shifts = np.load(shift_path)
    shifts = shifts[:-1, :timeseries.shape[0]].T

    start_time = time.time()

    # ray_align_frame=ray.remote(rmc.align_frame)

    vol_ids = [rmc.align_frame.remote(timeseries[i], shifts[i], output=None, order=3, mode='constant', cval=0.0,
                                      prefilter=True) for i in range(timeseries.shape[0])]

    timeseries = np.array(ray.get(vol_ids))

    duration = time.time() - start_time
    print(duration)
    return timeseries


if __name__ == '__main__':
    parent_path = f'F:\\OptovinScape\\20210628\\tiff_stacks\\20210628_6dpf_elavl3_H2B_fish3_run1'
    timeseries = apply_motion_correction(parent_path)
    print("read in volumes")
    # make mask
    timeseries = np.where(timeseries == 0, np.nan, timeseries)

    # build sub-stacks
    # reshape the timeseries array to have a rolling window of 5, along the 3rd axis, in a new 4th axis
    sub_stacks = define_sliding_z_thickness(timeseries, z_thickness=5, axis=1)
    # sub_stacks = define_sliding_z_thickness(timeseries, z_thickness=5, axis=3)
    n_windows = sub_stacks.shape[-2]

    del timeseries
    print("built substacks")
    # start_time = time.time()
    # parallelize function
    corr_map_ids = [compute_correlation_map.remote(sub_stacks[:, :, :, dz_ix, :], 1, 40) for dz_ix in range(n_windows)]
    # get data from pointed memory
    correlation_vol = np.array(ray.get(corr_map_ids)).transpose(1, 2, 0)
    # duration1 = time.time() - start_time
    # print(duration1)

    # squeeze dimensions
    # correlation_vol = correlation_vol[:-5, :, 2:-1]

    np.save(f"{parent_path}\\correlation_vol.npy", correlation_vol)
