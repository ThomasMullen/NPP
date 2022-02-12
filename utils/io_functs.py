import os

import numpy as np


def create_directory(parent_path, dir_name=''):
    # make export directory
    dir_path = os.path.join(parent_path, dir_name)
    # if already exists return directory path file
    if os.path.isdir(dir_path):
        print(f"Directory '{dir_path}' already exists.")
        return dir_path
    # otherwise make directory
    os.mkdir(dir_path)
    print(f"Directory '{dir_path}' created.")
    return dir_path


def convert_pngs_to_mp4(png_dir, filename, png_name="", number_units=6, fps=60, to_gif=True):
    """
    convert all .png files in directory to a mp4 video
    :param png_dir: directory where pngs are stored
    :param filename: saved mp4 file name
    :param png_name: prefix of tag pngs
    :param number_units: number of units labelled
    :param fps: frame rate of mp4
    :param to_gif: if want mp4 to be converted as a gif
    :return:
    """
    # cd directory to save space
    os.chdir(png_dir)
    # convert to video
    os.system(
        f"ffmpeg -r {fps} -f image2 -s 1920x1080 -i {png_name}%0{number_units}d.png -vcodec libx264 -crf 25  -pix_fmt "
        f"yuv420p {filename}.mp4")
    # convert to gif
    if to_gif:
        os.system(f"ffmpeg -i {filename}.mp4 -pix_fmt rgb24 -s qcif  {filename}.gif")
    return


def export_np_array(array, log_filepath, filename=''):
    """
    save the motion alignment values to then be able to correct for entire volumes,
    :param array: numpy array
    :param log_filepath: string of directory path (make sure it ends with '/')
    :param filename: file name
    :return: save npy files.
    """
    if os.path.isdir(log_filepath) or os.path.isdir(log_filepath[:-1]):
        np.save(f'{log_filepath}/{filename}.npy', array)
    return
© 2022 GitHub, Inc.
Terms
Privacy
