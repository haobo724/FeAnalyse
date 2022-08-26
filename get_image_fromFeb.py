import argparse
import cv2
import os
import pickle
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from read import FEmapping
from visualization import imgetoshow3DFast

SCALE = 1


def do_cluster(im_flatten, num_bins):
    cluster = KMeans(n_clusters=num_bins, random_state=0)
    # cluster = MiniBatchKMeans(n_clusters=num_bins, batch_size=2048)
    cluster.fit(im_flatten)
    return cluster


def find_best_k(data):
    intera = []
    value = data.reshape(-1, 1)
    value_s = StandardScaler().fit_transform(value)
    for k in range(120, 250):
        cluster_z = do_cluster(value_s, num_bins=k)
        intera.append(np.sqrt(cluster_z.inertia_))
    plt.plot(range(120, 250), intera, 'o-')
    plt.show()


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


def test(save_path='recon', save_path_img='',febname=''):
    fat_gray_value = 255
    with open(febname+'_fat_new.pkl', 'rb') as f:
        fat = pickle.load(f)

    with open(febname+'_tissue_new.pkl', 'rb') as f:
        tissue = pickle.load(f)
    a = []
    for k in fat:
        k = list(float(i) for i in k)
        a.append(k)
    thresh_idx = len(a) + 1
    for j in tissue:
        j = list(float(i) for i in j)
        a.append(j)

    pointcloud_as_array = np.array(a)  # shape is (7072, 3)

    print(pointcloud_as_array.shape)
    minium = np.min(pointcloud_as_array, axis=0)
    pointcloud_as_array -= minium
    maxium = np.max(pointcloud_as_array, axis=0)
    minium = np.min(pointcloud_as_array, axis=0)

    print('max ,min :', maxium, minium)


    scale = 400
    grad_shape = (np.around(maxium, 3) + 1 / scale) * scale
    grid_3d = np.zeros(grad_shape.astype(int))
    print(grid_3d.shape)
    pointcloud_as_array *= scale
    pointcloud_as_array = np.around(pointcloud_as_array, 3)
    nr = len(np.unique(pointcloud_as_array))//3
    print('{} points will be used in volume'.format(nr))
    tissue_value = 64
    for p in tqdm.tqdm(pointcloud_as_array[thresh_idx:]):
        x, y, z = p.astype(np.uint8)
        if grid_3d[x, y, z] == tissue_value:
            x_min = max(x - 1, 0)
            y_min = max(y - 1, 0)
            z_min = max(z - 1, 0)

            x_max = min(x + 1, grid_3d.shape[0])
            y_max = min(y + 1, grid_3d.shape[1])
            z_max = min(z + 1, grid_3d.shape[2])
            grid_3d[x_min:x_max, y_min:y_max, z_min:z_max] = tissue_value
        else:
            grid_3d[x, y, z] = tissue_value

    for p in tqdm.tqdm(pointcloud_as_array[:thresh_idx]):
        x, y, z = p.astype(np.uint8)
        if grid_3d[x, y, z] == fat_gray_value:
            x_min = max(x - 1, 0)
            y_min = max(y - 1, 0)
            z_min = max(z - 1, 0)

            x_max = min(x + 1, grid_3d.shape[0])
            y_max = min(y + 1, grid_3d.shape[1])
            z_max = min(z + 1, grid_3d.shape[2])
            grid_3d[x_min:x_max, y_min:y_max, z_min:z_max] = fat_gray_value
        else:
            grid_3d[x, y, z] = fat_gray_value

    grid_3d = grid_3d.astype(np.uint8)
    slice_nr = 0
    final_result = post(grid_3d,fat_gray_value)
    final_result = np.asarray(final_result, dtype=np.uint8)

    scale_factor= (4, 4, 4)
    final_result = zoom(final_result, scale_factor,order=0).astype(np.uint8)
    print(final_result.shape)
    name = os.path.join(save_path, febname+str(final_result.shape) + '.raw')
    with open(name, 'wb') as f:
        f.write(final_result)
    # for i in tqdm.tqdm(final_result):
    #     # print(np.unique(i))
    #     # i = cv2.resize(i, (420, 580), interpolation=cv2.INTER_NEAREST)
    #     name = os.path.join(save_path, str(slice_nr) + '.raw')
    #     with open(name, 'wb') as f:
    #         f.write(i)
    #     name_image = os.path.join(save_path_img, str(slice_nr) + '.jpg')
    #     img_resize = cv2.resize(i,(200,200),cv2.INTER_NEAREST)
    #     cv2.imwrite(name_image, img_resize)
    #     slice_nr += 1
    print('h,w:', final_result.shape[1:])
    return



def post(img,fat_gray_value=255):
    result = []
    for i in img:
        ret, binary = cv2.threshold(i, 1, 255, cv2.THRESH_BINARY_INV)
        # plt.imshow(binary)
        # plt.show()

        cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(cnts)
        # sorted_cnts = sorted(cnts,key=cv2.contourArea)
        blank = np.zeros_like(binary)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 10:
                # print(area)
                cv2.drawContours(blank, [c], 0, (fat_gray_value, fat_gray_value, fat_gray_value), -1)
        # plt.imshow(blank)
        # plt.show()

        img_post = blank + i
        result.append(img_post)
    return result

def setup(args):
    # step 1 : initialize all name parameters
    febfile_name = args.febfile_name
    dcm_name = args.dcm_name
    Node_name = args.Node_name

    with open(febfile_name, "rb") as f:
        feb_data = f.read()
    febname = os.path.basename(febfile_name).split('.')[0]
    if os.path.exists(febname+'_fat_new.pkl'):

        print(febname+'_fat_new.pkl is existed')
        return
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
    with open(febname+"_fat_new.pkl", 'wb') as f:
        pickle.dump(fat_center, f)
    with open(febname+f"_tissue_new.pkl", 'wb') as f:
        pickle.dump(tissue_center, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'error722/breast05_35_sf_067.feb')
    parser.add_argument('--dcm_name', type=str, help='', default='error722/Breast05.dcm')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='fat')

    args = parser.parse_args()
    print(args)
    setup(args)
    if not os.path.exists('recon_raw'):
        os.mkdir('recon_raw')
    if not os.path.exists('recon_img'):
        os.mkdir('recon_img')



    febname = os.path.basename(args.febfile_name).split('.')[0]
    save_path_raw = os.path.join('recon_raw',febname)
    save_path_img = os.path.join('recon_img',febname)
    if not os.path.exists(save_path_raw):
        os.mkdir(save_path_raw)
    if not os.path.exists(save_path_img):
        os.mkdir(save_path_img)
    test(save_path=save_path_raw, save_path_img=save_path_img, febname=febname)
