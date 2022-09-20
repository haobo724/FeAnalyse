import argparse
import glob
import os

import numpy as np
import pydicom
from matplotlib import pyplot as plt


def coord(PixelSpacing, point_location):
    x, y = np.array(point_location) // np.array(PixelSpacing)
    return int(x), int(y)


def save(name, maxium, line):
    text_name = os.path.basename(name).split('.')[0]
    save_path = os.path.join(os.path.dirname(name), text_name+ '.txt')
    side_points = np.concatenate([line[:10] , line[-10:]])
    avg = np.mean(side_points)

    with open(save_path, 'w') as f:
        f.write('maxium: '+ str(maxium))
        f.write('\n')
        f.write('avg: '+ str(avg))


def messure(dicom, windows_width=30, point_location=[86.351, 76.596], offset=None, show=True):
    suchbereich_radius = 20

    head = pydicom.read_file(dicom)
    dicom_volumen = pydicom.dcmread(dicom)

    dicom_array = dicom_volumen.pixel_array
    dicom_array = np.moveaxis(dicom_array, 0, 2)

    PixelSpacing = head[0x0028, 0x0030][:]

    if offset is None:
        offset = []
    else:
        offset = np.array(offset)
        x, y = coord(PixelSpacing, offset)
        point_location += np.array([x, y])
        print('point_location', point_location)
    if point_location is None:
        point_location = []

    slice_nr = dicom_array.shape[2] // 2
    PixelSpacing = list(map(float, PixelSpacing))
    print('PixelSpacing', PixelSpacing)
    x, y = point_location

    #
    # half_y = height_50N / 2
    # point_location[0] = width_50N - point_location[0]
    # point_location[1] = half_y - point_location[1]
    # x,y = coord(PixelSpacing,point_location)
    # x = dicom_array.shape[1]-x-1
    # rough_y = dicom_array.shape[0]//2-y
    # slice_line = dicom_array[:,x,slice_nr]
    # # print(np.max(slice_line),np.argmax(slice_line))
    # # y = np.argmax(slice_line)
    # # plt.plot(slice_line)
    # # plt.show()
    # #
    # peaks, _ = find_peaks(slice_line, distance=20,prominence=1,width=2)
    # diff = abs(peaks-rough_y)
    # y = peaks[np.argmin(diff)]
    suchbereich = dicom_array[y - suchbereich_radius:y + suchbereich_radius,
                  x - suchbereich_radius:x + suchbereich_radius, slice_nr].squeeze()
    offset_y, offset_x = np.unravel_index(suchbereich.argmax(), suchbereich.shape)
    offset_x -= suchbereich_radius
    offset_y -= suchbereich_radius
    y += offset_y
    x += offset_x
    maxium = np.max(suchbereich)
    print('x,y', x, y)
    print('maxium', maxium)

    sampled_line = dicom_array[y, x - windows_width:x + windows_width, slice_nr].squeeze()

    x_axis = np.arange(-windows_width, windows_width)
    y_axis = sampled_line
    save(dicom, np.max(suchbereich), y_axis)

    print('-' * 20)
    if show:
        plt.plot(x_axis, y_axis)
        plt.title("Connected Scatterplot points with line")
        plt.xlabel("x")
        plt.ylabel("sinx")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_path', type=str, help='', default='F:\Siemens\pachong\mean')
    parser.add_argument('--pl', type=list, help='', default=[[295, 1030], [795, 405], [1170, 1030]])
    parser.add_argument('--offset', type=list, help='einheit mm', default=[0, 0])
    parser.add_argument('--windows', type=int, help='', default=30)
    args = parser.parse_args()
    # Folder PATH!
    point_location = args.pl
    dicom_path = glob.glob(os.path.join(args.dicom_path, '*.dcm'))
    for dicom in dicom_path:
        print('Now:', dicom)
        for p in point_location:
            messure(dicom, windows_width=args.windows, point_location=p, offset=args.offset)
