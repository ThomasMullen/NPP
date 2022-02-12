import os
import h5py
import mat73
import numpy as np

"""
View HDF5 data structure
------------------------
"""


def traverse_datasets(hdf_file):
    """
    Peak into matlab file and print the Key, Shape, Data type.
    :param hdf_file:
    :return:
    """

    def h5py_dataset_iterator(g, prefix=''):
        """
        iterate through the HDF5 file and search through the nested datasets
        :param g: .mat filepath
        :param prefix:
        :return: prints out the directory/subdirectory, shape, and dtype in the HDF5 file
        """
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield path, item
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def view_hdf_structure(filepath, print_labels=False):
    """
    Looks through the structure and prints information about the structure.
    :param filepath: filepath of .mat
    :return:
    """
    vol_labels = []
    with h5py.File(filepath, 'r') as f:
        for dataset in traverse_datasets(f):
            if print_labels:
                print(f'Path: {dataset}\tShape: {f[dataset].shape}\tData type: {f[dataset].dtype}')
            vol_labels.append(dataset)
    return vol_labels[:-1]


def list_experiment_directories(experiment_parent_directory):
    """
    :param experiment_parent_directory: the directory which contains the folders of fish experiments.
    :return: a list of directories, each directory containing 3 .mat file i.e. log files that need preprocessing.
    """
    list_experiment_directories = next(os.walk(experiment_parent_directory))[1]
    print('Experiments:')
    [print(i) for i in list_experiment_directories]
    return list_experiment_directories


def extract_dataset(filepath, dataset_name=''):
    """
    extracts the dataset of the dataset you are interested in
    :param filepath: the .mat filepath
    :param dataset_name: the name of the dataset you are interested in
    :return: a n-dimensional array for the dataset.
    """
    # print(dataset_name)
    with h5py.File(filepath, 'r') as f:
        data = np.array(f[dataset_name][:])
    return data


def pull_frames(mat_filepath, plane_number, frame_range, start_frame=0, sample_rate=10):
    labels = view_hdf_structure(mat_filepath)
    # extract shape of plane
    plane_shape = extract_dataset(mat_filepath, labels[0]).shape[:2]
    frames_shape = (int(frame_range / sample_rate),) + plane_shape
    frames = np.zeros(frames_shape)

    for i, frame in enumerate(range(start_frame, start_frame + frame_range, sample_rate)):
        frames[i, :, :] = extract_dataset(mat_filepath, labels[frame])[:, :, plane_number, 0, 0]
    return frames


def compute_total_frames(volumes_dir, number_chars=3, is_mat_file=False):
    vol_list = os.listdir(volumes_dir)

    try:
        last_frame = max([int(name[:-number_chars]) for name in vol_list])
    except:
        raise print("Ensure only integer named .h5 file volumes are in directory:", volumes_dir)
    if is_mat_file:
        return last_frame
    return last_frame


def list_frames_numbers(volumes_dir, number_chars=3):
    vol_list = os.listdir(volumes_dir)

    try:
        frames = np.array([int(name[:-number_chars]) for name in vol_list])
    except:
        raise print("Ensure only integer named .h5 file volumes are in directory:", volumes_dir)
    return np.sort(frames)


def export_numpy_2_h5(array, filepath, to_compress=True):
    # store original volume shape
    vol_shape = array.shape

    # reshape volume into 2D array
    array = array.reshape(vol_shape[0], -1)

    # export as hdf5 file
    file = h5py.File(filepath, 'w')
    if to_compress:
        file.create_dataset("vol", shape=vol_shape, data=array, compression="gzip", compression_opts=9)
    else:
        file.create_dataset("vol", shape=vol_shape, data=array)
    file.close()
    return


if __name__ == '__main__':
    path_dir = "/Volumes/LSM4/tomData/01062021/Fish4/tiff_stacks/20210601_7dpf_HUC_H2B_fish4_run1/"
    file_name = "dataSkewCorrected.mat"

    labels = view_hdf_structure(path_dir + file_name)

    # # extract plane
    # plane_160 = np.zeros((100, 262, 710))
    # for i, frame in enumerate(range(0, 1000, 10)):
    #     plane_160[i, :, :] = extract_dataset(path_dir + file_name, labels[frame])[:, :, 160, 0, 0]

    # extract volume
    full_vol = extract_dataset(path_dir + file_name, labels[100])[:, :, :, 0, 0]
