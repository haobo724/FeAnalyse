import argparse
import cv2
import os
import pickle

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


def test(save_path='recon', save_path_img='', fat_gray_value=255, dicom_path=''):
    with open('Fat_new.pkl', 'rb') as f:
        fat = pickle.load(f)

    with open('Tissue_new.pkl', 'rb') as f:
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
    # temp = pointcloud_as_array.copy()
    # imgetoshow3DFast(pointcloud_as_array)
    # data = StandardScaler().fit_transform(temp[:, 2].reshape(-1, 1)).reshape(-1)
    #
    # pointcloud_as_array[:, 0] = data
    # pointcloud_as_array[:, 1] = StandardScaler().fit_transform(temp[:, 1].reshape(-1, 1)).reshape(-1)
    # pointcloud_as_array[:, 2] = StandardScaler().fit_transform(temp[:, 0].reshape(-1, 1)).reshape(-1)

    # volume = np.where(dicom_array > 0)
    # x = np.max(volume[0]) - np.min(volume[0]) + 1
    # y = np.max(volume[1]) - np.min(volume[1]) + 1
    # z = np.max(volume[2]) - np.min(volume[2]) + 1
    # print('volume :"', x, y, z)
    print(pointcloud_as_array.shape)
    minium = np.min(pointcloud_as_array, axis=0)
    pointcloud_as_array -= minium
    maxium = np.max(pointcloud_as_array, axis=0)
    minium = np.min(pointcloud_as_array, axis=0)

    print('max ,min :', maxium, minium)

    # abstand_thresh = 0.001
    # x=y=z =np.arange(0,np.around(max(maxium),2),abstand_thresh)
    # print(x)
    # mesh_x,mesh_y,mesh_z = np.meshgrid(x,y,z)
    # ratio = 1/len(x)
    # input()
    # # grid_3d = np.zeros((1/abstand_thresh,1/abstand_thresh,1/abstand_thresh))
    # grid_3d = np.zeros((150,150,150))
    # for i in tqdm.tqdm(range(150)):
    #     for j in range(150):
    #         for k in range(150):
    #             coord = np.array([i,j,k])*ratio
    #             distances = np.sqrt(np.sum(np.asarray(pointcloud_as_array - coord) ** 2, axis=1))
    #             if np.min(distances)<abstand_thresh*5:
    #                 idx = np.argmin(distances)
    #                 if idx <thresh_idx:
    #                     grid_3d[i,j,k]=255
    #                 else:
    #                     grid_3d[i,j,k]=128
    #
    #             else:
    #                 grid_3d[i,j,k]=0

    scale = 400
    grad_shape = (np.around(maxium, 3) + 1 / scale) * scale
    grid_3d = np.zeros(grad_shape.astype(int))
    print(grid_3d.shape)
    pointcloud_as_array *= scale
    pointcloud_as_array = np.around(pointcloud_as_array, 3)
    nr = len(np.unique(pointcloud_as_array))//3
    print('{} points will be used in volume'.format(nr))
    for p in tqdm.tqdm(pointcloud_as_array[thresh_idx:]):
        x, y, z = p.astype(np.uint8)
        if grid_3d[x, y, z] == 64:
            x_min = max(x - 1, 0)
            y_min = max(y - 1, 0)
            z_min = max(z - 1, 0)

            x_max = min(x + 1, grid_3d.shape[0])
            y_max = min(y + 1, grid_3d.shape[1])
            z_max = min(z + 1, grid_3d.shape[2])
            grid_3d[x_min:x_max, y_min:y_max, z_min:z_max] = 64
        else:
            grid_3d[x, y, z] = 64

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
    for i in tqdm.tqdm(final_result):
        name = os.path.join(save_path, str(slice_nr) + '.raw')
        with open(name, 'wb') as f:
            f.write(i)
        name_image = os.path.join(save_path_img, str(slice_nr) + '.jpg')
        img_resize = cv2.resize(i,(200,200),cv2.INTER_NEAREST)
        cv2.imwrite(name_image, img_resize)
        slice_nr += 1
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
    # plt.imshow(img_post)
    # plt.show()

    #
    # xyz_d = distance / np.array([Rows, Columns, NumberofFrames])
    # print('xyz_d', xyz_d)
    #
    # cluster_z = do_cluster(pointcloud_as_array[:, 2].reshape(-1, 1), num_bins=NumberofFrames)
    # # cluster_z_rank = cluster_z.cluster_centers_.argsort(axis=0)
    # center_z = sorted(cluster_z.cluster_centers_)
    # center_z = np.array(center_z).reshape(-1)
    #
    # cluster_y = do_cluster(pointcloud_as_array[:, 1].reshape(-1, 1), num_bins=Rows)
    # # cluster_y_rank = cluster_y.cluster_centers_.argsort(axis=0)
    # center_y = sorted(cluster_y.cluster_centers_)
    # center_y = np.array(center_y).reshape(-1)
    #
    # cluster_x = do_cluster(pointcloud_as_array[:, 0].reshape(-1, 1), num_bins=Columns)
    # cluster_x_rank = np.argsort(cluster_x.cluster_centers_, axis=0)
    # center_x = sorted(cluster_x.cluster_centers_)
    # center_x = np.array(center_x).reshape(-1)
    #
    # new_point = []
    # if not os.path.exists('new_poin-t.pkl'):
    #
    #     for point in tqdm.tqdm(pointcloud_as_array):
    #         x, y, z = point
    #         result1 = cluster_x.predict(x.reshape(1, -1))[0]
    #         # result1= cluster_x_rank[result1]
    #         p1 = cluster_x.cluster_centers_[result1][0]
    #         print(x, p1)
    #         result2 = cluster_y.predict(y.reshape(1, -1))[0]
    #         # result2= cluster_y_rank[result2]
    #
    #         p2 = cluster_y.cluster_centers_[result2][0]
    #
    #         result3 = cluster_z.predict(z.reshape(1, -1))[0]
    #         # result3= cluster_z_rank[result3]
    #
    #         p3 = cluster_z.cluster_centers_[result3][0]
    #         # print(p1,p2,p3)
    #         # print(result1,result2,result3)
    #         # new_point.append([result1,result2,result3])
    #         new_point.append([p1, p2, p3])
    #
    #     new_point = np.array(new_point)
    #     print('done')
    #     with open("new_point.pkl", 'wb') as f:
    #         pickle.dump(new_point, f)
    # else:
    #     with open('new_point.pkl', 'rb') as f:
    #         new_point = pickle.load(f)
    #     print('new_point loaded')
    #
    # # index 方法
    # # bb = np.zeros([int(x)+1,int(y)+1,int(z)+1])
    # #
    # # for i in tqdm.tqdm(new_point):
    # #     x1,y1,z1 = i
    # #     bb[x1,y1,z1] =gray_value
    #
    # # for idx in range(bb.shape[2]):
    # #
    # #     name = os.path.join(save_path, str(slice_nr) + '.raw')
    # #     with open(name, 'wb') as f:
    # #         f.write(bb[:,:,idx])
    #
    # pointcloud_as_array = new_point.copy()
    #
    # dz = np.gradient(center_z).mean()
    # dx = np.gradient(center_x).mean()
    # dy = np.gradient(center_y).mean()
    # print('d:', dz, dx, dy)
    # print(Rows * SCALE, Columns * SCALE)
    # slice_nr = 0
    #
    # for Z in tqdm.tqdm(center_z):
    #     final_pointcloud_array = []
    #     for point in pointcloud_as_array:
    #
    #         if point[2] == Z:
    #             final_pointcloud_array.append(point)
    #     final_pointcloud_array = np.array(final_pointcloud_array)  # shape is (7072, 3)
    #     final_pointcloud_array = final_pointcloud_array[:, :2]
    #     if not len(final_pointcloud_array):
    #         print(final_pointcloud_array.shape)
    #         continue
    #
    #     blank = np.zeros((Rows * SCALE, Columns * SCALE))
    #     x = np.arange(minium[0], maxium[0], dx / SCALE)
    #     y = np.arange(minium[1], maxium[1], dy / SCALE)
    #
    #     for index1, i in enumerate(x[:Columns * SCALE]):
    #         for index2, j in enumerate(y[:Rows * SCALE]):
    #             pos = (i, j)
    #             distances = np.sqrt(np.sum(np.asarray(final_pointcloud_array - pos) ** 2, axis=1))
    #             if np.min(distances) <= math.sqrt(dx ** 2 + dy * 2) * 1.1:
    #                 # if np.min(distances) <= xyz_d[0]*2:
    #                 blank[index2, index1] = gray_value
    #
    #     blank = blank.astype(np.uint8)
    #
    #     name = os.path.join(save_path, str(slice_nr) + '.raw')
    #     with open(name, 'wb') as f:
    #         f.write(blank)
    #     slice_nr += 1


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
    parser.add_argument('--febfile_name', type=str, help='', default=r'error722/breast05_35_sf_200.feb')
    parser.add_argument('--dcm_name', type=str, help='', default='error722/Breast05.dcm')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='fat')

    args = parser.parse_args()
    print(args)
    # setup(args)
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
    test(save_path=save_path_raw, save_path_img=save_path_img, fat_gray_value=255, dicom_path=args.dcm_name)
    # test(save_path=save_path_Tissue, part='Tissue', gray_value=128, dicom_path=args.dcm_name)
