"""
This script reduces the size of the volumes. Each volume will be saved as a separate file and number w.r.t. the time it
was recorded. the exported volumes will be reshaped as a 2D array but the HD5F file will contain information about the
original shape so it will be able to reproduce the 3D volume.
"""
from tqdm import tqdm

import utils.hdf5_tools as h5_tools

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.simple_script(nargout=0)
# eng.quit()
from utils.io_functs import create_directory


def export_to_single_volumes(filepath, volume_labels, export_path, reduce_volume=False, reduced_dims=None):
    if reduce_volume:
        xmin, xmax, ymin, ymax, zmin, zmax = reduced_dims
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = (None, None, None, None, None, None)

    # define buffer
    x = 0
    for i, label in enumerate(tqdm(volume_labels)):
        # read in full volume and truncate using defined params
        try:
            vol = h5_tools.extract_dataset(filepath, label)[ymin:ymax, xmin:xmax, zmin:zmax, 0, 0]
        except:
            x += 1
            continue

        # reshape volume into 2D array & export as hdf5 file
        h5_tools.export_numpy_2_h5(array=vol, filepath=f"{export_path}/{(i - x):06}.h5")
    return


if __name__ == '__main__':
    path_dir = "/Volumes/TomOrgerLab/experimentSCAPE/20210630_fish1"
    file_name = "data.mat"

    # make export directory for reduced volumes for motion correction
    reduced_export_path = create_directory(parent_path=path_dir, dir_name='reduced_volumes')
    full_export_path = create_directory(parent_path=path_dir, dir_name='full_volumes')

    # get a list of all the dataset names to open
    labels = h5_tools.view_hdf_structure(f"{path_dir}/{file_name}")

    # TODO: run for complete timeseries (for now only run for a sample of ~100)
    # labels = labels[1310:1450]

    # define reduced volume dimensions
    # [z = 143, y = 513, x = 299]

    dims = (40, 130, 365, 550, 120, 150)

    export_to_single_volumes(filepath=path_dir + file_name, volume_labels=labels, export_path=reduced_export_path,
                             reduce_volume=True, reduced_dims=dims)

    # calculate volumes
    export_to_single_volumes(filepath=path_dir + file_name, volume_labels=labels, export_path=full_export_path,
                             reduce_volume=False)
