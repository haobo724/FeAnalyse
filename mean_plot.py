import argparse
import glob
import os

import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def coord(PixelSpacing,point_location=[]):
    x,y =np.array(point_location)//np.array(PixelSpacing)
    return int(x),int(y)
def messure(dicom, windows_width=20, point_location=[86.351,76.596], offset=None,show=True):
    if offset is None:
        offset = []
    else:
        point_location = np.array(point_location)
        offset = np.array(offset)
        point_location += offset
        print('point_location',point_location)
    if point_location is None:
        point_location = []
    width_50N =91.9303
    height_50N =149.6168
    head = pydicom.read_file(dicom)
    dicom_volumen = pydicom.dcmread(dicom)

    dicom_array = dicom_volumen.pixel_array
    dicom_array = np.moveaxis(dicom_array, 0, 2)

    PixelSpacing = head[0x0028, 0x0030][:]
    slice_nr = dicom_array.shape[2]//2
    PixelSpacing= list(map(float,PixelSpacing))
    print('PixelSpacing',PixelSpacing)





    half_y = height_50N / 2
    point_location[0] = width_50N - point_location[0]
    point_location[1] = half_y - point_location[1]
    x,y = coord(PixelSpacing,point_location)
    x = dicom_array.shape[1]-x-1
    rough_y = dicom_array.shape[0]//2-y

    slice_line = dicom_array[:,x,slice_nr]
    # print(np.max(slice_line),np.argmax(slice_line))
    # y = np.argmax(slice_line)
    # plt.plot(slice_line)
    # plt.show()
    #
    peaks, _ = find_peaks(slice_line, distance=20,prominence=1,width=2)
    diff = abs(peaks-rough_y)
    y = peaks[np.argmin(diff)]



    print('x,y',x,y)
    sampled_line = dicom_array[y,x-windows_width:x+windows_width,:].squeeze()
    x_axis = np.arange(-windows_width,windows_width)
    y_axis = sampled_line[...,slice_nr]
    print('-'*20)
    if show:
        plt.plot(x_axis, y_axis)
        plt.title("Connected Scatterplot points with line")
        plt.xlabel("x")
        plt.ylabel("sinx")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_path', type=str, help='', default='F:\Siemens\pachong\mean')
    parser.add_argument('--pl', type=list, help='', default=[[56.33,26.56],[86.351,76.596],[16.362,76.607]])
    parser.add_argument('--offset', type=list, help='', default=[0,0])
    parser.add_argument('--windows', type=int, help='', default=30)
    args = parser.parse_args()
    # Folder PATH!
    point_location=args.pl
    dicom_path = glob.glob(os.path.join(args.dicom_path,'*.dcm'))
    for dicom in dicom_path:
        print('Now:',dicom)
        for p in point_location:
            messure(dicom,windows_width=args.windows,point_location=p,offset=args.offset)


    