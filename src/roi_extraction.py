import numpy as np
from scipy.stats import pearsonr
from skimage import morphology
import matplotlib.pyplot as plt

from src.correlation_map import apply_motion_correction
from utils.io_functs import create_directory, export_np_array
from visualisation.depth_projection import depth_volume_plot

"""
ROI extraction
------
We will now generate a function that extracts ROIs (regions of interest). The function should take 4 inputs: a
correlation map, the final stack of aligned frames, a correlation threshold value and a max size of an ROI in pixels.
The function should:
* Identify the pixel with the highest correlation value in the correlation map (the ROI seed). Initialize the list of
pixels in this ROI with this **seed.**
* Compute the individual correlations of the fluorescence time-series in this pixel with its 8 neighbours. A convenient
way of idetifying the neighbours of a region is by dilating the reion by one piexl. You can do this by importing
morphology from skimage and using `morphology.binary_dilation`.
* Add the neighbours whose correlation exceeds the threshold to the ROI.
* Repeat by looking at the neighbouring pixels of the new ROI and computing the correlation of their fluorescence with
the total fluorescence of the ROI. Add the pixels to the ROI if they exceed the threshold.
* Stop the function if no neighbouring pixels exceed the threshold or if the size of the ROI exceeds the maximum size
(the last input to the function).
* The function should return the pixels within the ROI, the fluoresence trace of the ROI, the size of the ROI and the
correlation map with the values of the pixels in the extracted ROI set to zero.
Mike's notes for finding the regions:
1. use the threshold and find the maximum correlation value in the map as a seed
2. take indices of nearest neighbours
3. assign seed as the template trace and extract traces of neighbours
4. find correlation with template trace and that of each neighbouring trace
5. join neighbours beyond a given threshold and reasign the total average as the new trace
"""


def next_roi(correlation_volume, frames, correlation_threshold, max_volume):
    # find maximum existing correlation point
    this_max = np.nanmax(correlation_volume)
    # print(this_max)
    # get the index coordinates max correlation value (seed)
    result = np.where(correlation_volume == this_max)
    coords = list(zip(result[0], result[1], result[2]))
    # store each index for each dimension
    I, J, K = coords[0][:]
    # extract timeseries for the seed
    this_roi_trace = np.squeeze(frames[:, I, J, K])
    # create a correlation map for the ROI
    this_roi = np.zeros_like(correlation_volume)
    # assign correlation value to seed as 1
    this_roi[I, J, K] = 1

    this_correlation_map = np.copy(correlation_volume)
    this_correlation_map[I, J, K] = 0;

    added = 1

    while np.sum(this_roi != 0) < max_volume and added == 1:
        added = 0
        # dilated = morphology.binary_dilation(this_roi, np.ones((3, 3, 3))).astype(np.uint8)
        dilated = morphology.binary_dilation(this_roi, morphology.cube(width=3)).astype(np.uint8)# converta boolean type
        new_pixels = dilated - this_roi
        # locate index of neighbouring pixels and stack as coordinated
        coords = np.stack(np.where(new_pixels == 1), axis=1).astype(dtype=np.int32)
        # iterate through each neighbour and calc correlation with ROI
        for a in range(coords.shape[0]):
            # extract index for coordinate
            I, J, K = coords[a]
            # check it hasn't already been used otherwise it would've been set to zero
            """
            map_corr_func = lambda a, b, c: pearsonr(a[:,b], c)[0] 
            list(map(map_corr_func, frames, coords, Y))
            """
            if this_correlation_map[I, J, K] != 0:
                # extract time series
                Y = np.squeeze(frames[:, I, J, K])
                # calculate timeseries correlation with the roi timeseries
                C, _ = pearsonr(this_roi_trace, Y)
                # if they are highly correlated join, and update the masks
                if C > correlation_threshold:
                    # mark as part of the ROI
                    this_roi[I, J, K] = 1
                    # set correlation value to 0 on the map
                    this_correlation_map[I, J, K] = 0
                    # add to ROI timeseries
                    this_roi_trace = this_roi_trace + Y # check if you need to average rather than add
                    # keep in while loop
                    added = 1

    return this_roi, this_roi_trace, np.sum(this_roi), this_correlation_map


if __name__ == '__main__':
    parent_path = f'F:\\OptovinScape\\20210628\\tiff_stacks\\20210628_6dpf_elavl3_H2B_fish3_run1'
    # read in aligned frames
    aligned_frames = apply_motion_correction(parent_path=parent_path)
    # read in correlation map
    original_correlation_map = np.load(f"{parent_path}\\correlation_vol_reduced.npy")
    # original_correlation_map = np.load(f"{parent_path}\\correlation_vol.npy")

    correlation_map = np.copy(original_correlation_map)

    n_rois = 200
    # create a timeseries array of all ROIs
    all_traces = np.zeros((n_rois, aligned_frames.shape[0]))
    # create an ROI volume
    all_rois = np.zeros_like(original_correlation_map)
    # create an used pixel volume
    used_pixels = np.zeros_like(original_correlation_map)

    for i in range(n_rois):
        this_roi3, this_roi_trace, N, this_correlation_map = next_roi(correlation_map, aligned_frames, 0.3, 600)
        all_traces[i, :] = this_roi_trace
        all_rois = all_rois + (i + 1) * this_roi3
        used_pixels = used_pixels + this_roi3
        correlation_map[all_rois > 0] = 0

    # export data
    create_directory(parent_path=parent_path, dir_name="roiData")
    export_np_array(all_traces, f"{parent_path}\\roiData", "all_traces")
    export_np_array(all_rois, f"{parent_path}\\roiData", "all_rois")
    export_np_array(used_pixels, f"{parent_path}\\roiData", "used_pixels")


    # view plots
    depth_volume_plot(original_correlation_map, show_fig=True)
    correlation_map[correlation_map==0]=200
    depth_volume_plot(correlation_map,color_scaling=190, show_fig=True)

    norm_trace = []
    # fig, ax = plt.subplots(figsize=(10,10))
    for i in range(all_traces.shape[0]):
        x = all_traces[i].copy()
        # normalise traces
        x *= (1/x.max())
        norm_trace.append(x)
        # ax.plot(x + (i * 5), lw=.5)
    # plt.show()
    norm_trace = np.array(norm_trace)

    plt.figure()
    plt.imshow(norm_trace, aspect='auto', vmax=1)
    plt.show()
