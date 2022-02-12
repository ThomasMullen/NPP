import os
import sys
import argparse

sys.path.append('./src/')
sys.path.append('./utils')

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define the parameters used for pre-processing pipeline')
    parser.add_argument('-m', help="First argument number of iterations in rigid motion algorithm, second "
                                   "argument number of samples used to make template in rigid motion "
                                   "correction", nargs=2, metavar=('Iter', 'Samples'),
                        default=[3, 20], type=int)
    parser.add_argument('-d', '--Duplicate', help="Make a copy of volumes that have had rigid motion algorithm applied",
                        default=True, type=bool)
    parser.add_argument('-pD', '--PathData', help="path to data", default='./data/', type=str)
    parser.add_argument('-dZ', '--ZThickness', help="Define the volume z-thickness when searching for correlationss ",
                        default=5, type=int)
    parser.add_argument('-r', '--ROI', help="Define minimum correlation threshold, maximum ROI voxel size, "
                                            "maximum number of ROIs in trace extraction.", default=[0.3, 600, 200],
                        nargs=3, metavar=('CorrThresh', 'RegionSize', 'NRegion'), type=float)
    parser.add_argument('-dN', '--DataName', help="name of the dataset in raw file e.g. 'vol' or 'reducedVol'.",
                        default='toy', type=str)
    parser.add_argument('-o', '--Out', help="path to save shifts, correlation volume, and traces results, i.e. not "
                                            "volume data.", default='./data/', type=str)
    args = parser.parse_args()
    main(args)
