import numpy as np
import numpy.ma as ma
import os
import numpy.ma as ma

import ray
from numpy.linalg import norm
from numpy.random import choice
from skimage.registration import phase_cross_correlation
import scipy.ndimage as ndi
from tqdm import tqdm

import utils.hdf5_tools as h5_tools
from utils.io_functs import export_np_array, create_directory


class MotionCorrection:
    """
    ## Properties:
    :param max_recursions: number of iterations of template used.
    :param n_samples: number of volumes used to build the mean template.
    :param n_time_steps: number of timesteps in neural recording data.
    :param frames: neural recording data - dimensions (t, x, y, z).
    :param bggray: nonzero background offset
    :param original_template: reference template build from non-aligned frames.
    :param aligned_template: latest updated template volume build from aligned frames.
    :param shifts: latest motion correction for each frame in each x, y, z dimension.
    :param total_motion_log: absolute distance for each voxel over each iteration.
    :param aligned_frames: latest aligned frames.
    """

    def __init__(self, frames, n_max_recursions, n_samples, log_filepath='', shifts=None):
        """
        :param frames: imaging data used to calculate rigid alignment, should be in the form (t, x, y, z)
        :param n_max_recursions: number of recursive iterations used to calculate the shifts in data
        :param n_samples: number of volumes used to make the template volume.
        """
        self.max_recursions = n_max_recursions
        self.n_samples = n_samples
        self.n_time_steps = frames.shape[0]
        assert self.n_time_steps > self.n_samples
        self.frames = frames
        self.bggray = self.compute_background()
        self.correct_background()
        self.aligned_template = self.create_template()
        self.shifts = shifts
        self.log_filepath = log_filepath

    def sample(self):
        """
        samples n indices from n time steps without replacement
        :return: an array of indices
        """
        return choice(self.n_time_steps, self.n_samples, replace=False)

    def compute_background(self):
        """
        use all the frames to calculate the background offset
        :return: the lowest non-zero value
        """
        # flatten n sample frames
        temp = self.frames[self.sample()].flatten()
        # drop all zero values and find the minimum value
        bggray = min(temp[temp != 0])
        return bggray

    def correct_background(self):
        """
        remove background offset from voxel values.
        :return: updated the voxel values
        """
        self.frames -= self.bggray
        return

    def create_template(self):
        """
        creates a reference volume by randomly chooses `n_samples` from `n_time_steps` frames and then calculating a
        mean value for each voxel. The mean voxel value should reduce noise.
        :return: a starting template to calculate the shift.
        """
        return np.mean(self.frames[self.sample()], 0)

    def update_template(self, **kwargs):
        """
        Uses the aligned frames to calculate the updated template volume.
        :return: updated aligned template
        Example:
        --------
        >>>> update_template(self, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        """

        # sample from frames to be aligned and used for the new template
        samples = self.sample()

        # parallise shift fn using ray
        ray_shift = ray.remote(ndi.shift)

        # shift and align image
        aligned_frames_ids = [ray_shift.remote(frame, dr, **kwargs) for frame, dr in
                              zip(self.frames[samples], self.shifts.T[samples])]
        # pull aligned frame
        aligned_frames = np.array(ray.get(aligned_frames_ids))

        # reference template from aligned frames
        self.aligned_template = np.mean(aligned_frames, axis=0)

    def compute_phase_correlation(self, **kwargs):
        """
        :param frames: x-y slices over time, dimensions should be ordered (t, x, y, z)
        :param anatomy: the reference anatomy each frame is compared with. If empty, the MIP of all frames will be used.
        :return: phase shift in the X, Y and total shift for each frame.
        Example
        -------
        >>>> from skimage.registration import phase_cross_correlation
        >>>> ray_phase_cross_correlation = ray.remote(phase_cross_correlation)
        >>>> ray_phase_cross_correlation.remote(self.aligned_template, self.frames[i, :, :, :], upsample_factor=10, space='real', return_error=False)
        """

        # not parallelize iterate monatonically and find shift values
        shifts = np.zeros((3, self.n_time_steps))
        for i, frame in enumerate(self.frames):
            shifts[:, i] = phase_cross_correlation(self.aligned_template, frame, **kwargs)

        # # parallelize phase shift function
        # ray_phase_cross_correlation = ray.remote(phase_cross_correlation)
        # # iterate through each volume frame and compare phase alignment to the reference template volume
        # shift_ids = [ray_phase_cross_correlation.remote(self.aligned_template, frame, **kwargs) for frame in self.frames]
        # # concatenate shift values from memory id
        # shifts = np.stack(ray.get(shift_ids), axis=1)

        # calculate absolute euclidean distance and save shifts to a log file
        export_np_array(np.concatenate([shifts, norm([shifts], axis=1)]), self.log_filepath,
                        filename=f'{self.max_recursions}_motion_shifts')

        # update shifts using the latest template
        self.shifts = shifts
        return

    def align_frames(self, **kwargs):
        """
        Shift all frames in the x-y-z direction depending on phase correlation w.r.t. to the reference template volume.
        :param shifts: the alignments in each dimension of the original volume frames w.r.t. to the latest updated
        template volume.
        :param aligned_frames: the corrected volume frames using the shifts - dimensions (t, x, y, z)
        :param frames: the original volume frames - dimensions (t, x, y, z)
        :return: aligned frames
        Example:
        --------
        >>>> align_frames.remote(self, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        """
        # parallise shift fn using ray
        ray_shift = ray.remote(ndi.shift)

        # shift and align image
        aligned_frames_ids = [ray_shift.remote(frame, dr, **kwargs) for frame, dr in zip(self.frames, self.shifts.T)]
        # pull aligned frame
        return np.array(ray.get(aligned_frames_ids))

    def rigid_alignment(self):
        """
        Recursive function that iterately updates the reference template and motion correction shifts w.r.t. the
        original volume frames. Number of iterations define by the `max_recursion` variable in the constructor.
        :return: updated `aligned_frames` and `aligned_template`, the latest `shifts`.
        """
        if self.max_recursions == 0:
            return

        self.compute_phase_correlation(upsample_factor=10, space='real', return_error=False)
        self.update_template(output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        self.max_recursions -= 1
        print("Remaining iterations", self.max_recursions)
        return self.rigid_alignment()


def create_mask(data, value, filepath=None):
    """
    Create a mask for the volume/time varying volume
    :param data: numpy array to build mask over
    :param value: want to mask
    :param filepath: filepath to export mask
    :return: numpy boolean mask
    """
    mask = ma.masked_equal(data, value)
    if filepath is not None:
        np.save(filepath, mask.mask)
    return mask


@ray.remote
def align_frame(frame, alignment, **kwargs):
    """
    Shift frames in the x-y-z direction depending on phase correlation w.r.t. to the reference template volume.
    :param alignment: the alignments in each dimension of the original volume frames w.r.t. to the latest updated
    template volume.
    :param frame: the original volume frame - dimensions (x, y, z)
    :return: updated aligned frames
    
    Example:
    --------
    >>>> align_frame(frame, alignment, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    
    """
    # unwrap the tuples of shifts
    X, Y, Z = alignment
    # shift and align image
    aligned_frame = ndi.shift(frame, (X, Y, Z), **kwargs)
    return aligned_frame




def update_full_volumes(
        parent_path=f'F:\\OptovinScape\\20210628\\tiff_stacks\\20210628_6dpf_elavl3_H2B_fish3_run1', copy=True):
    """
    Read in the shifts calculated from the reduced volumes and apply to the full volumes.
    :param parent_path: str directory which contain the `reducedVolume` and `volume` directories.
    :param copy: True if you want to create a copy  of the full volumes with the aligment, or false if you want to over-
    write the original volume files.
    :return: void
    """
    # path directory of full volumes
    # path_dir = f"{parent_path}\\volumes"
    path_dir = f"{parent_path}\\reducedVolumes"

    # var name
    # data_name = "vol"
    data_name = "reducedVol"

    # get total number of volumes
    n_timesteps = h5_tools.compute_total_frames(path_dir, 4, True)

    # load shifts
    shift_path = f"{parent_path}\\1_motion_shifts.npy"
    shifts = np.load(shift_path)
    shifts = shifts[:-1, :n_timesteps]

    frames_path = f"{parent_path}\\removed_frames.npy"

    # if removed frames exist
    if os.path.exists(shift_path):
        frames_exc = np.load(frames_path)
        frames_exc = frames_exc[frames_exc <= n_timesteps]
        # insert 0 for index of removed frames in the shifts array
        shifts = np.insert(shifts.T, frames_exc, [0, 0, 0], axis=0)

    # if copy true make new alignedVolume directory
    if copy:
        create_directory(parent_path, "alignedVolume")

    timeseries = np.stack(
        [np.array(h5_tools.extract_dataset(f'{path_dir}/{file_name:06}.mat', data_name)[:]) for file_name in
         range(n_timesteps)])

    vol_ids = [align_frame.remote(timeseries[i], shifts[i], output=None, order=3, mode='constant', cval=0.0,
                                  prefilter=True) for i in range(timeseries.shape[0])]

    timeseries = np.array(ray.get(vol_ids)).astype(np.uint8)

    # export depending on copy
    if copy:
        [h5_tools.export_numpy_2_h5(volume, f"{parent_path}\\alignedVolume\\{i:06}.mat", to_compress=True) for
         i, volume in enumerate(timeseries)]

    else:
        [h5_tools.export_numpy_2_h5(volume, f'{path_dir}/{i:06}.mat', to_compress=True) for i, volume in
         enumerate(timeseries)]

    return


if __name__ == '__main__':
    parent_path = f'F:\\OptovinScape\\20210628\\tiff_stacks\\20210628_6dpf_elavl3_H2B_fish3_run1'
    path_dir = f'{parent_path}\\reducedVolumes'
    # path_dir = '/Users/thomasmullen/Desktop/SCAPE/full_volumes_matlab'

    n_timesteps = h5_tools.compute_total_frames(path_dir, 4, True)
    # only take a sample for testing
    # n_timesteps = 100

    # read in the reduced volumes
    timeseries = np.stack(
        [np.array(h5_tools.extract_dataset(f'{path_dir}/{file_name:06}.mat', "reducedVol")[:]) for file_name in
         tqdm(range(n_timesteps))])

    # remove optovin flashes
    optovin_frames = np.unique(np.where(np.mean(timeseries, axis=(1, 2)) > 116)[0])
    if optovin_frames.size != 0:
        timeseries = np.delete(timeseries, optovin_frames, axis=0)
        export_np_array(optovin_frames, log_filepath=f"{path_dir}\\..\\", filename="removed_frames")

    # get motion shift
    motion_corrected = MotionCorrection(frames=timeseries, n_max_recursions=3, n_samples=20,
                                        log_filepath=f"{path_dir}\\..\\")
    motion_corrected.rigid_alignment()

    # implement on full volumes
    update_full_volumes(parent_path=parent_path, copy=True)
