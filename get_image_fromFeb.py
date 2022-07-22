import argparse
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from read import FEmapping
from visualization import imgetoshow3DFast
SCALE = 1

def test(save_path = 'recon',part = 'Fat',gray_value=255):
    reader = FEmapping()
    reader.read_dicom(r'Breast06_left.dcm')
    PixelSpacing = reader.get_info('PixelSpacing')
    SliceThickness = reader.get_info('SliceThickness')
    NumberofFrames=reader.get_info('NumberofFrames')
    Rows=reader.get_info('Rows')
    Columns=reader.get_info('Columns' )
    PixelSpacing.append(SliceThickness)

    print('NumberofFrames:',NumberofFrames)

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

    pointcloud_as_array = np.array(a) #shape is (7072, 3)
    print(pointcloud_as_array.shape)
    imgetoshow3DFast(pointcloud_as_array)


    pointcloud_as_array=np.around(pointcloud_as_array, 3)
    # hist, bin_edges=np.histogram(pointcloud_as_array[:,2])
    distance = np.max(pointcloud_as_array, axis=0) - np.min(pointcloud_as_array, axis=0)

    print('distance1:', distance)
    print('max ,min :', np.max(pointcloud_as_array, axis=0),np.min(pointcloud_as_array, axis=0))
    maxium = np.max(pointcloud_as_array, axis=0)
    minium = np.min(pointcloud_as_array, axis=0)

    xyz_d  = distance/np.array([Rows,Columns,NumberofFrames])
    print('xyz_d',xyz_d)
    res, num = np.unique(pointcloud_as_array[:, 2], return_counts=True)
    print(num)
    _ = plt.hist(num, bins='auto')  # arguments are passed to np.histogram
    plt.show()
    thresh = np.mean(num)
    print('group thresh:',thresh)
    index = np.where(num > thresh)
    x_y_most = res[index]
    print("x_y_most",x_y_most)



    # print()
    # pointcloud_as_array = pointcloud_as_array/(pointcloud_as_array.max(axis=0))
    # pointcloud_as_array = pointcloud_as_array+abs(np.min(pointcloud_as_array,axis=0))
    # pointcloud_as_array = pointcloud_as_array*[1.08435303, 0.88868057, 0.57180586]
    # X_codinate = pointcloud_as_array[:,0]/ pointcloud_as_array[:,0].max()
    # y_codinate = pointcloud_as_array[:,1]
    # z_codinate = pointcloud_as_array[:,2]

    Z = np.min(pointcloud_as_array[:,2])
    gradient = np.gradient(x_y_most).mean()
    print(gradient)
    # d = abs(x_y_most[0]-x_y_most[1])
    d = gradient*0.8
    # d = 0.008
    print(Rows*SCALE,Columns*SCALE)
    slice_nr = 0


    while Z<maxium[2]:
        final_pointcloud_array = []
        for point in pointcloud_as_array:

            if Z - d <= point[2] and point[2] <= Z + d:
                final_pointcloud_array.append(point)
        final_pointcloud_array = np.array(final_pointcloud_array) #shape is (7072, 3)
        if not len(final_pointcloud_array):
            print(final_pointcloud_array.shape)
            Z += d
            continue

        # X_codinate = (((final_pointcloud_array[:,0]+1)/2)*Rows)
        #
        # y_codinate = (((final_pointcloud_array[:,1]+1)/2)*Columns)
        # xy_codinate = final_pointcloud_array[:,:2]
        blank = np.zeros((Rows*SCALE,Columns*SCALE))
        y = np.arange(minium[1],maxium[1],xyz_d[1]/SCALE)
        x = np.arange(minium[0],maxium[0],xyz_d[0]/SCALE)

        for index1,i in enumerate(x[:Columns*SCALE]):
            for index2,j in  enumerate(y[:Rows*SCALE]):
                pos  = (i,j,Z)
                distances = np.sqrt(np.sum(np.asarray(final_pointcloud_array-pos) ** 2, axis=1))
                if np.min(distances) <= math.sqrt( xyz_d[0]**2+ xyz_d[1]**2+xyz_d[2]**2):
                # if np.min(distances) <= xyz_d[0]*2:
                    blank[index2, index1] = gray_value
        # plt.pause(0.1)
        # plt.imshow(blank)
        # plt.show()
        # cv2.imwrite('recon2/'+str(Z)+'.jpg',blank)
        blank = blank.astype(np.uint8)
        # blank.tobytes()
        # blank.tofile('recon2/'+str(Z)+'.raw')
        # imageio.imsave('recon2/'+str(Z)+'.raw',blank)
        name = os.path.join(save_path,str(slice_nr)+'.raw')
        with open(name, 'wb') as f:
            f.write(blank)
        Z+=xyz_d[2]
        slice_nr+=1


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
    fe.get_center_from_element(Fat_element_dic, node_dic)
    fe.get_center_from_element(Tissue_element_dic, node_dic, cls='Tissue')

    fat_center, tissue_center = fe.get_center()
    with open("fat_new.pkl", 'wb') as f:
        pickle.dump(fat_center, f)
    with open(f"tissue_new.pkl", 'wb') as f:
        pickle.dump(tissue_center, f)


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'error722/breast04_30_sf_133.feb')
    parser.add_argument('--dcm_name', type=str, help='', default='error722/Breast04.dcm')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='fat')

    args = parser.parse_args()
    print(args)
    setup(args)
    save_path_Fat = 'recon_Fat'
    save_path_Tissue = 'recon_Tissue'
    if not os.path.exists(save_path_Fat):
        os.mkdir(save_path_Fat)
    if not os.path.exists(save_path_Tissue):
        os.mkdir(save_path_Tissue)
    test(save_path=save_path_Fat,part='Fat',gray_value =255)
    test(save_path=save_path_Tissue,part='Tissue',gray_value =128)
