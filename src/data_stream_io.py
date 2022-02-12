import numpy as np
import zarr
import mat73
# import pyini
import os
import configparser
import numcodecs
import tifffile
r"""
Explanation
----------
Image data is acquired from the camera as a continuous stream of bytes. These bytes as stored into .dat files. To 
reconstruct the 1D array into a formatted structure requires a .ini configure file to describe the original image shape
and also an info.mat file.
Each .dat file will encode M number of planes, and there will be N number of planes that completes 1 volume (N > M). You
will need multiple .dat files to recover a full volume, and these .dat files may rollover to the next volume.
Key parameters:
----------
:acquisition_config.planes_per_datfile:         planes per a .dat file
:info.info.daq.pixelsPerLine:                   planes per a volume
:info.info.daq.numberOfScans:                   number of volumes
:n_scans * planes_per_vol:                      total number of planes saved
:round(total_planes / planes_per_datfile):      total number of .dat file
:planes_per_vol/planes_per_datfile:             number of .dat files for a volume
"""

def check_filepath(filepath):
    assert os.path.isfile(filepath) == True


class Struct(object):
    """
    Converts nested dictionary (e.g. .mat file) into an object-like structure. Pass in dictionary
    """

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Struct(b) if isinstance(b, dict) else b)


class LoadConfig:
    """
    load parameters to structure from the data acquisition `.ini` file.
    """

    def __init__(self, filepath):
        read_config = configparser.ConfigParser()
        read_config.read(filepath, encoding='utf-8-sig')
        self.path = filepath
        self.aoi_height = int(read_config.get('data', 'aoiheight'))
        self.aoi_width = int(read_config.get('data', 'aoiwidth'))
        self.aoi_stride = int(read_config.get('data', 'aoistride'))
        self.image_size_bytes = int(read_config.get('data', 'imagesizebytes'))
        self.pixel_encoding = read_config.get('data', 'pixelencoding')
        self.is_16_bit = (self.pixel_encoding.lower() == 'mono16')
        self.planes_per_datfile = int(read_config.get('multiimage', 'imagesperfile'))
        self.calculate_expected_stride()

    def print_data(self):
        """Print out the list of feilds in the .ini file"""
        read_config = configparser.ConfigParser()
        read_config.read(self.path, encoding='utf-8-sig')
        [print(i) for i in read_config.items('data')]

    def calculate_expected_stride(self):
        """Calculate the expected stride depending on the bit encoding"""
        if self.is_16_bit:
            self.expected_stride = (self.aoi_width * 2)
        else:
            self.expected_stride = (self.aoi_width * 3 / 2)

    def calculate_image_byte_dimension(self, extra_rows):
        """
        finds the actual number of rows and columns in the data saved by the cameera.
        :param extra_rows:
        :param image_size_bytes: parameter found in the image acqDataStruct.data.imagesizebytes
        :return: (number of rows, number of columns)
        """
        # self.planes_per_datfile
        n_rows = self.aoi_height + extra_rows
        n_cols = int(self.image_size_bytes / (2 * n_rows))
        if n_cols != self.aoi_stride / 2:
            RuntimeError("Something wrong in numberColumns calculation")
        return n_rows, n_cols

    def calculate_padded_rows(self):
        """2 rows always padded at the end."""
        return self.aoi_stride - self.expected_stride

    def calculate_extra_rows(self):
        extra_rows = int(self.image_size_bytes / self.aoi_stride) - self.aoi_height
        return extra_rows

    def calculate_image_crop(self):
        """
        As the camera pads the images, this gives you the region to crop for the real data.
        :return: x_crop, y_crop
        """
        row_pad_bytes = self.calculate_padded_rows()
        extra_rows = self.calculate_extra_rows()
        n_row, n_col = self.calculate_image_byte_dimension(extra_rows)

        x_crop = n_col - row_pad_bytes / 2
        y_crop = n_row - extra_rows - 1
        return int(x_crop), int(y_crop)


"""Functions to include"""

# read in dark volume function

# subtract dark vol include a 12/16 bit parameter

# extract parameters for skew correction
"""
scan_angle = info.info.daq.scanAngle
x_width = info.info.GUIcalFactors.xK_umPerVolt*scan_angle/planes_per_vol
conversionFactors = [info.info.GUIcalFactors.y_umPerPix, info.info.GUIcalFactors.z_umPerPix, x_width]
"""


# get sorted file names
def sort_spool_filenames(total_planes, images_per_file_ini):
    r"""
    The format of the filenames are numerically labelled in order but the id is reversed. This function returns the
    labels in a sorted array
    Note: need to generalise this for more than 10 digits, for now this will work.
    max_digits is set to 10 `{i:010}`.
    # \mathbf{A} = \left[\begin{array}{ccc} 1 & 3 & 5\\ 2 & 5 & 1\\ 2 & 3 & 8\end{array}\right].
    [1, 0, 0, 0, 0, 0, 0, 0, 0]
    [2, 0, 0, 0, 0, 0, 0, 0, 0]
    [3, 0, 0, 0, 0, 0, 0, 0, 0]
    [4, 0, 0, 0, 0, 0, 0, 0, 0]
    [5, 0, 0, 0, 0, 0, 0, 0, 0]
    [6, 0, 0, 0, 0, 0, 0, 0, 0]
    [7, 0, 0, 0, 0, 0, 0, 0, 0]
    [8, 0, 0, 0, 0, 0, 0, 0, 0]
    [9, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 0, 0, 0, 0, 0, 0, 0]
    [1, 1, 0, 0, 0, 0, 0, 0, 0]
    [2, 1, 0, 0, 0, 0, 0, 0, 0]
    [3, 1, 0, 0, 0, 0, 0, 0, 0]
    :return:
    """
    n_spool_files = int(np.ceil(total_planes / images_per_file_ini))
    file_names = [str(f'{i:010}')[::-1] + "spool.dat" for i in range(n_spool_files)]
    return file_names


def read_dat_file(filepath):
    """
    read in data file into a numpy array.
    :param filepath: of the .dat
    :return: 1d raw numpy array of the dat file
    """
    with open(f"{filepath}", "r") as f:
        return np.fromfile(f, dtype=np.uint16)


def reshape_dat(dat_arr, n_cols, n_rows, images_per_file):
    """
    :param dat_arr:
    :param n_cols:
    :param n_rows:
    :param images_per_file:
    :return:
    Example:
    --------
    >>>> ini_filepath = f"{data_dir}/acquisitionmetadata.ini"
    >>>> acquisition_config = LoadConfig(ini_filepath)
    >>>> planes_per_datfile, n_rows, n_cols = acquisition_config.planes_per_datfile,
    >>>>                                   acquisition_config.calculate_image_byte_dimension(acquisition_config.extra_rows())
    >>>> reshape_dat(arr, n_cols, n_rows, planes_per_datfile)
    """

    # note: change index order, permute i.e. x.transpose(3,1,2)

    return dat_arr.reshape((n_cols, n_rows, images_per_file))


def crop_dat_vol(dat_arr, x_crop, y_crop):
    """
    Crops the data array defined by x_crop and y_crop, in order to remove padding from the image files.
    :return: cropped 3D data array (x, y, z)
    **Note:** data array dimension is by default defined as (x, y, z). It will only follow the pipeline standard array
    order of (z, y, x) after the .zarr file has been exported.
    tempvol(:,:,find(spoolsNeeded==file2load))=SCAPE_rawdata(Xcrop(1):Xcrop(2),Ycrop(1):Ycrop(2) ,imagesNeeded(find(spoolsNeeded==file2load)));
    """
    return dat_arr[:x_crop, :y_crop]


def save_to_tiff(dat_arr, filepath):
    for img in dat_arr.transpose(2, 1, 0):
        tifffile.imsave(f"{filepath}/frame.tiif", img, append=True, bigtiff=True, imagej=True)


def save_to_zarr(zarr_filepath):
    compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
    data = np.random.normal(1, 180, (5000, 183, 560, 200)).astype(np.int8)
    z = zarr.array(data, chunks=(1000, 1000), compressor=compressor)
    zarr.save(f'{dump_dir}/example.zarr', z)
    z.compressor

    # with zarr.LMDBStore(...) as store:
    # array = zarr.create(..., chunks=(1000,1000), store=store, ...)
    # for line_num, line in enumerate(text_file):
    #     array[line_num, :] = process_data(line)
    z.info
    pass


"""
If acquisition stoped before the end correct the SpoolToLoad number
    %this should be outside the loop
    d = dir([whichExp, '\*.dat']);
    n=length(d);
    if n < size(filesToLoad,2)
        framesavailable=(n-1)*imagesperfile;
        %how many images in the last spool file?
        loadfile = filesToLoad{n};
        filePath = fullfile(whichExp, loadfile);
        FID = fopen(filePath, 'r');
        rawData = fread(FID, 'uint16=>uint16');
        fclose(FID);
        %count all images before it goes to zeros
        endpoint=find(rawData>0,1,'last');
        if (~isempty(endpoint))
        framesavailable=framesavailable+ceil(endpoint/(fileimagesizebytes/2));
        else
        framesavailable=framesavailable+imagesperfile;
        end
        numScans = floor(framesavailable/framesPerScan); % length of whole experiment in volumes
        volumesToLoad = 1:numScans; 
    end
"""


def export_dat_files(filename, x_max, y_max, planes_per_scan):
    pass


""""
Read in the data files
%where we will load camera raw data
    SCAPE_rawdata=zeros(numberCols,numberRows , imagesperfile,'uint16');
    %where we will load the true volume data extracted from the raw camera data
    tempvol=zeros(Xcrop(2)-Xcrop(1)+1,Ycrop(2)-Ycrop(1)+1 , framesPerScan,'uint16');
    
    %TEMP
    if (vols>0)
     numScans = vols; % length of whole experiment in volumes
        volumesToLoad = 1:numScans;
    end
    
    %load one volume at a time
    for currvol=volumesToLoad
        desiredFrames=[1:framesPerScan]+(currvol-1)*framesPerScan;
        %which spool (.dat) file is each frame in and what number in that file
        spoolsNeeded=ceil(desiredFrames/imagesperfile);
        imagesNeeded=mod(desiredFrames-1,imagesperfile)+1;
        
        %load the data from the spool files and concatenate into tempvol
        for file2load=unique(spoolsNeeded)
            if (currentfileloaded~=file2load)
               %load a new file 
                loadfile = filesToLoad{file2load};
                filePath = fullfile(whichExp, loadfile);
                FID = fopen(filePath, 'r');
                rawData = fread(FID, 'uint16=>uint16');
                fclose(FID);
                currentfileloaded=file2load;
                SCAPE_rawdata=reshape(rawData,[numberCols numberRows   imagesperfile]);
            end
            tempvol(:,:,find(spoolsNeeded==file2load))=SCAPE_rawdata(Xcrop(1):Xcrop(2),Ycrop(1):Ycrop(2) ,imagesNeeded(find(spoolsNeeded==file2load)));
        end
        
    t = permute(tempvol, [3 1 2]); % lab convention ZXY
        imageDimensions = size(t);
        changDim = ones(1, 1,imageDimensions(1),imageDimensions(2),imageDimensions(3), 'uint16');
        changDim(1,chan,:,:,:) = t;
        clear t
        
                if currvol == 1
                    imagedata=cell([numScans 1]);
                    save(filename,'imagedata','-v7.3');
                    saveFile=matfile(filename);
                    saveFile.Properties.Writable=true;
                    saveFile.imagedata(currvol,1)={changDim};
                else
                    saveFile.imagedata(currvol,1)={changDim};
                end
"""

"""
Note for stripe correction: when you subtract volumetosubtract, make sure that the volumetosubtract has not be trimmed 
on the negatvie values this is what causing the systematic trim 
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dump_dir = f"/Users/thomasmullen/Desktop/dump"
    parent_dir = f"/Volumes/TomOrgerLab/SCAPE_Raw_Data_test"
    data_dir = f"{parent_dir}/20210805_ECLAV3_6dpf_Fish2_run1"

    # get ini parameters
    ini_filepath = f"{data_dir}/acquisitionmetadata.ini"
    acquisition_config = LoadConfig(ini_filepath)

    # get info
    info_file = f"{parent_dir}/20210805_ECLAV3_6dpf_Fish2_run1_info.mat"
    info = Struct(mat73.loadmat(info_file))

    scane_name = info.info.scanName
    planes_per_vol = info.info.daq.pixelsPerLine.astype(int)  # framesPerScan
    n_scans = info.info.daq.numberOfScans.astype(int)  # length of experiment in volumes
    # camera parameters - stripe correction
    x_left = info.info.camera.x_left.astype(int)
    x_roi = info.info.camera.xROI.astype(int)
    y_roi = info.info.camera.yROI.astype(int)

    # skew correction parameters
    scan_angle = info.info.daq.scanAngle
    x_width = info.info.GUIcalFactors.xK_umPerVolt * scan_angle / planes_per_vol
    conversionFactors = [info.info.GUIcalFactors.y_umPerPix, info.info.GUIcalFactors.z_umPerPix, x_width]

    # read in files
    planes_per_datfile = acquisition_config.planes_per_datfile
    n_dat_per_vol = int(np.ceil(planes_per_vol/planes_per_datfile))
    x_crop, y_crop = acquisition_config.calculate_image_crop()
    total_planes = n_scans * planes_per_vol
    spool_file_needed = round(total_planes / planes_per_datfile)

    file_names = sort_spool_filenames(total_planes, planes_per_datfile)

    # define volume array
    temp_vol = np.zeros((x_crop, y_crop, planes_per_vol), dtype=np.uint8)


    # test reading in .dat files
    dat_filepath = f"{data_dir}/{file_names[100]}"
    dat_arr = read_dat_file(dat_filepath)
    (n_rows, n_cols) = acquisition_config.calculate_image_byte_dimension(acquisition_config.calculate_extra_rows())

    dat_arr = reshape_dat(dat_arr, n_cols, n_rows, planes_per_datfile)
    dat_arr = crop_dat_vol(dat_arr, x_crop, y_crop)

    # for img in dat_arr.transpose(2, 1, 0):
    #     tifffile.imsave(f"{dump_dir}/frame.tiff", img, append=True)

    for vol_ix in range(n_scans):
        # list plane indices for the defined volume
        desired_plane_list = np.arange(planes_per_vol) + (vol_ix * planes_per_vol)
        # assign a dat file number to each plane
        spool_file_needed_for_plane = np.floor(desired_plane_list / planes_per_datfile).astype(int)
        # assign plane number for each dat file
        planes_needed = desired_plane_list % planes_per_datfile

        for dat_ix in np.unique(spool_file_needed_for_plane):
            # print(file_names[dat_ix])
            dat_arr = read_dat_file(f"{data_dir}/{file_names[dat_ix]}")
            dat_arr = reshape_dat(dat_arr, n_cols, n_rows, planes_per_datfile)
            dat_arr = crop_dat_vol(dat_arr, x_crop, y_crop)

            plane_ix = planes_needed[np.where(spool_file_needed_for_plane == dat_ix)[0]]
            dat_arr[..., plane_ix]





    tifffile.imsave(f"{dump_dir}/frame.tiff", dat_arr, append=True, bigtiff=True, imagej=True)

    # f = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        print(filenames)
        # f.append(int(filenames[:-9]))

def interpolate_files(planes_per_vol, dat_ix):
    # list the number of planes references for a given .dat file index
    desired_plane_list = np.arange(planes_per_vol) + (dat_ix * planes_per_vol)
    spool_file_needed_for_plane = np.floor(desired_plane_list/planes_per_datfile)
    planes_needed = desired_plane_list % planes_per_datfile

    # loaded_dat
    dat_files_needed = np.unique(spool_file_needed_for_plane)
    first_dat_file = dat_files_needed[0]
    len_first_dat_file = sum(spool_file_needed_for_plane==first_dat_file)
    last_dat_file = dat_files_needed[-1]
    len_last_dat_file = sum(spool_file_needed_for_plane==last_dat_file)
