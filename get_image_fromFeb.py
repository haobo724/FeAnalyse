import argparse
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from visualization import imgetoshow3DFast
from read import FEmapping

SCALE = 1


def do_cluster(im_flatten, num_bins):
    # cluster = KMeans(n_clusters=num_bins)
    cluster = MiniBatchKMeans(n_clusters=num_bins, batch_size=2048)
    cluster.fit(im_flatten)
    return cluster


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def test(save_path='recon', part='Fat', gray_value=255, dicom_path=''):
    reader = FEmapping()
    reader.read_dicom(dicom_path)
    PixelSpacing = reader.get_info('PixelSpacing')
    SliceThickness = reader.get_info('SliceThickness')

    NumberofFrames = reader.get_info('NumberofFrames')
    Rows = reader.get_info('Rows')
    Columns = reader.get_info('Columns')

    PixelSpacing.append(SliceThickness)
    dicom_array =  reader.get_info('dicom_array')
    range = np.where(dicom_array > 0)
    x = np.max(range[0]) - np.min(range[0])
    y = np.max(range[1]) - np.min(range[1])
    z = np.max(range[2]) - np.min(range[2])
    volumen_size = [x, y, z] #-----------x
    print('NumberofFrames:', NumberofFrames)
    print('volumen_size:', volumen_size)
    Rows,Columns,NumberofFrames = y, x, z
    with open('Fat_new.pkl', 'rb') as f:
        fat = pickle.load(f)

    with open('Tissue_new.pkl', 'rb') as f:
        tissue = pickle.load(f)
    a = []

    if part == 'Fat':
        for k in fat:
            k = list(float(i) for i in k)
            a.append(k)

    else:
        for j in tissue:
            j = list(float(i) for i in j)
            a.append(j)

    pointcloud_as_array = np.array(a)  # shape is (7072, 3)
    print(pointcloud_as_array.shape)
    # imgetoshow3DFast(pointcloud_as_array)
    cluster = do_cluster(pointcloud_as_array[:, 2].reshape(-1, 1), num_bins=NumberofFrames)
    # print(cluster.cluster_centers_)
    # print(center.shape)
    center = sorted(cluster.cluster_centers_)
    center = np.array(center).reshape(-1)

    print(center.shape)
    d = np.gradient(center).mean()
    print('d:',d)
    # hist = centroid_histogram(cluster)
    # print(hist)
    # _ = plt.hist(hist, bins='auto')  # arguments are passed to np.histogram
    # plt.show()
    # pointcloud_as_array=np.around(pointcloud_as_array,3)

    # hist, bin_edges=np.histogram(pointcloud_as_array[:,2])
    # distance = np.max(pointcloud_as_array, axis=0) - np.min(pointcloud_as_array, axis=0)
    #
    # print('distance1:', distance)
    print('max ,min :', np.max(pointcloud_as_array, axis=0), np.min(pointcloud_as_array, axis=0))

    maxium = np.max(pointcloud_as_array, axis=0)
    minium = np.min(pointcloud_as_array, axis=0)
    abstand = maxium - minium
    space_spacing = abstand/PixelSpacing
    print('Volume space:',space_spacing)
    # xyz_d  = distance/np.array([Rows,Columns,NumberofFrames])
    # print('xyz_d',xyz_d)
    # res, num = np.unique(pointcloud_as_array[:, 2], return_counts=True)
    # print(num)
    #
    # thresh = np.mean(num)
    # print('group thresh:',thresh)
    # index = np.where(num > thresh)
    # x_y_most = res[index]
    # print("x_y_most",x_y_most)

    # print()
    # pointcloud_as_array = pointcloud_as_array/(pointcloud_as_array.max(axis=0))
    # pointcloud_as_array = pointcloud_as_array+abs(np.min(pointcloud_as_array,axis=0))
    # pointcloud_as_array = pointcloud_as_array*[1.08435303, 0.88868057, 0.57180586]
    # X_codinate = pointcloud_as_array[:,0]/ pointcloud_as_array[:,0].max()
    # y_codinate = pointcloud_as_array[:,1]
    # z_codinate = pointcloud_as_array[:,2]

    # Z = np.min(pointcloud_as_array[:,2])
    # gradient = np.gradient(x_y_most).mean()
    # print(gradient)
    # d = gradient*0.8
    # print(Rows*SCALE,Columns*SCALE)
    slice_nr = 0
    arry = np.array([ Rows * SCALE-1,Columns * SCALE-1, 0], dtype=np.uint8)
    print(arry)
    for sl_z in center:
        final_pointcloud_array = []
        blank = np.zeros(( Columns * SCALE,Rows * SCALE))
        for point in pointcloud_as_array:

            if sl_z - d  <= point[2] and point[2] <= sl_z + d :
                final_pointcloud_array.append(point)

        final_pointcloud_array = np.array(final_pointcloud_array)  # shape is (7072, 3)
        final_pointcloud_array = final_pointcloud_array - np.array(minium)
        final_pointcloud_array = final_pointcloud_array / np.array(space_spacing)
        final_pointcloud_array = final_pointcloud_array / np.max(final_pointcloud_array, axis=0)
        final_pointcloud_array = final_pointcloud_array * arry
        final_pointcloud_array_xy =final_pointcloud_array[:, :2]
        # final_pointcloud_array_xy = sorted(final_pointcloud_array_xy,key=lambda k:[k[1]])

        # imgetoshow3DFast(final_pointcloud_array)
        # index = np.around(final_pointcloud_array_xy).astype(np.uint8)
        for i in final_pointcloud_array_xy.astype(np.uint8):
            x, y = i
            x +=random.randint(-3,3)
            x = min(x,147)
            blank[int(y), int(x)] = gray_value
        blank = cv2.GaussianBlur(blank,(3,3),1)

        # blank = cv2.dilate(blank,kernel=(7,7))
        blank = np.where(blank>0,0,gray_value)
        blank = blank.astype(np.uint8)

        # print(blank.shape)
        # plt.imshow(blank)
        # plt.show()
        name = os.path.join(save_path, str(slice_nr) + '.raw')
        with open(name, 'wb') as f:
            f.write(blank)
        slice_nr+=1

        # plt.imshow(blank)
        # plt.show()
        # if not len(final_pointcloud_array):
        #     print(final_pointcloud_array.shape)
        #     sl_z += d
        #     continue

        #
        # blank = np.zeros((Rows*SCALE,Columns*SCALE))
        # y = np.arange(minium[1],maxium[1],xyz_d[1]/SCALE)
        # x = np.arange(minium[0],maxium[0],xyz_d[0]/SCALE)
        #
        # for index1,i in enumerate(x[:Columns*SCALE]):
        #     for index2,j in  enumerate(y[:Rows*SCALE]):
        #         pos  = (i,j,Z)
        #         distances = np.sqrt(np.sum(np.asarray(final_pointcloud_array-pos) ** 2, axis=1))
        #         if np.min(distances) <= math.sqrt( xyz_d[0]**2+ xyz_d[1]**2+xyz_d[2]**2):
        #         # if np.min(distances) <= xyz_d[0]*2:
        #             blank[index2, index1] = gray_value
        #
        # blank = blank.astype(np.uint8)
        #
        # name = os.path.join(save_path,str(slice_nr)+'.raw')
        # with open(name, 'wb') as f:
        #     f.write(blank)
        # Z+=xyz_d[2]


def setup(args):
    # step 1 : initialize all name parameters
    febfile_name = args.febfile_name
    dcm_name = args.dcm_name
    Node_name = args.Node_name

    with open(febfile_name, "rb") as f:
        feb_data = f.read()
    fe = FEmapping()
    fe.read_dicom(dcm_name)

    # step 3 : analyse feb file
    '''
    workflow :
    according to matname in  'MeshDomains - SolidDomain' to know the element name ,i.e 'Part1'
    according to element name to get element infos
    according to args.Node_name to get node infos

    '''

    Fat_element = fe.get_Ele(feb_data, 'Part2')
    Tissue_element = fe.get_Ele(feb_data, 'Part1')
    Fat_element_dic, Fat_element_dic_full = fe.get_node_single_ele(Fat_element)
    Tissue_element_dic, Tissue_element_dic_full = fe.get_node_single_ele(Tissue_element)
    node_dic = fe.get_node_dic(feb_data, f'{Node_name}')
    fe.get_eckePoint_from_element(Fat_element_dic, node_dic)
    fe.get_eckePoint_from_element(Tissue_element_dic, node_dic, cls='Tissue')

    fat_center, tissue_center = fe.get_ecke()
    with open("fat_new.pkl", 'wb') as f:
        pickle.dump(fat_center, f)
    with open(f"tissue_new.pkl", 'wb') as f:
        pickle.dump(tissue_center, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'error722/breast04_30_sf_001.feb')
    parser.add_argument('--dcm_name', type=str, help='', default='error722/Breast04.dcm')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='fat')

    args = parser.parse_args()
    print(args)
    # setup(args)
    save_path_Fat = 'recon_Fat'
    save_path_Tissue = 'recon_Tissue'
    if not os.path.exists(save_path_Fat):
        os.mkdir(save_path_Fat)
    if not os.path.exists(save_path_Tissue):
        os.mkdir(save_path_Tissue)
    test(save_path=save_path_Fat, part='Fat', gray_value=255, dicom_path=args.dcm_name)
    test(save_path=save_path_Tissue, part='Tissue', gray_value=128, dicom_path=args.dcm_name)
