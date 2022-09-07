import argparse
import os
import pickle

import numpy as np
import tqdm

from read import FEmapping


def setup(args, check_exists=False):
    # step 1 : initialize all name parameters
    febfile_name = args.febfile_name
    Node_name = args.Node_name

    with open(febfile_name, "rb") as f:
        feb_data = f.read()
    febname = os.path.basename(febfile_name).split('.')[0]
    # step 2 : Check if the pkl file exists,
    # if it does then go straight to the reconstruction step

    if check_exists:
        if os.path.exists(febname + '_fat_eck.pkl'):
            print(febname + '_fat_eck.pkl is existed')
            return
    fe = FEmapping()

    # step 3 : analyse feb file
    '''
    workflow :
    according to matname in  'MeshDomains - SolidDomain' to know the element name ,i.e 'Part1'
    according to element name to get element infos
    according to args.Node_name to get node infos

    '''

    Fat_element = fe.get_Ele(feb_data, 'Part2')
    Tissue_element = fe.get_Ele(feb_data, 'Part1')

    # 得到的是纯element 比如 id：1 ---123123，2323,23131
    # "Copy" the corresponding pure element section
    Fat_element_dic, Fat_element_dic_full = fe.get_node_single_ele(Fat_element)
    Tissue_element_dic, Tissue_element_dic_full = fe.get_node_single_ele(Tissue_element)

    # 得到的是纯node 比如 id：1 ---123123，2323,23131
    # "Copy" the corresponding pure Node section
    node_dic = fe.get_node_dic(feb_data, f'{Node_name}')

    # Let element and node be combined,
    # now element contains the coordinates of the node instead of the ordinal number
    # if use variable Fat_element_dic_full and Tissue_element_dic_full then all 20 node of element will be used
    fe.get_element_detail(Fat_element_dic, node_dic)
    fe.get_element_detail(Tissue_element_dic, node_dic, cls='Tissue')
    fat_eck, tissue_eck = fe.get_ecke()

    # to numpy
    all_node_cood_fat_eck = fat_eck.values()
    all_node_cood_tissue_eck = tissue_eck.values()
    print(len(all_node_cood_fat_eck))
    all_node_cood_fat_eck = np.array(list(all_node_cood_fat_eck))
    all_node_cood_tissue_eck = np.array(list(all_node_cood_tissue_eck))
    with open(febname + "_fat_eck.pkl", 'wb') as f:
        pickle.dump(all_node_cood_fat_eck, f)
    with open(febname + f"_tissue_eck.pkl", 'wb') as f:
        pickle.dump(all_node_cood_tissue_eck, f)


def recon(save_path='recon', save_path_img='', febname=''):
    fat_gray_value = 128
    tissue_gray_value = 255
    with open(febname + '_fat_eck.pkl', 'rb') as f:
        fat = pickle.load(f)

    with open(febname + '_tissue_eck.pkl', 'rb') as f:
        tissue = pickle.load(f)
    separator = fat.shape[0]
    nr_eck = fat.shape[1]
    # temp= fat.reshape(-1,3)[:16,:]
    # print(len(fat))
    # print(len(tissue))
    # input()
    # imgetoshow3DFast(temp)

    # connect fat and tissue to obtain the maximum and minimum values
    # of the entire point cloud in xyz direction
    all_element = np.concatenate([fat, tissue], axis=0)
    all_element_reshape = all_element.reshape(-1, 3)
    # The entire point cloud is translated to
    # ensure that there are no negative values and thus corresponds to the image
    minium = np.min(all_element_reshape, axis=0)
    all_element_reshape -= minium
    maxium = np.max(all_element_reshape, axis=0)
    # back reshape to (length of node,8,3)

    all_element = all_element_reshape.reshape(-1, nr_eck, 3)
    print(all_element.shape)

    # Create a blank canvas
    scale = 1000
    grad_shape = (np.around(maxium, 4) + 1 / scale) * scale
    grid_3d = np.zeros(grad_shape.astype(int))
    all_element *= scale
    all_element_ds = np.around(all_element, 4)

    # Iterate over element ,tissue first then fat
    for element in tqdm.tqdm(all_element_ds[separator:]):
        maxium = np.max(element, axis=0).astype(np.uint8)
        minium = np.min(element, axis=0).astype(np.uint8)
        grid_3d[minium[0]:maxium[0], minium[1]:maxium[1], minium[2]:maxium[2]] = tissue_gray_value

    for element in tqdm.tqdm(all_element_ds[:separator]):
        maxium = np.max(element, axis=0).astype(np.uint8)
        minium = np.min(element, axis=0).astype(np.uint8)

        grid_3d[minium[0]:maxium[0], minium[1]:maxium[1], minium[2]:maxium[2]] = fat_gray_value

    # scale_factor = (4, 4, 4)
    # grid_3d = zoom(grid_3d, scale_factor, order=2).astype(np.uint8)
    # print(grid_3d.shape)
    print(grid_3d.shape)
    x_padding = 300 - grid_3d.shape[0]
    y_padding = 300 - grid_3d.shape[1]
    z_padding = 300 - grid_3d.shape[2]
    grid_3d = np.pad(grid_3d, (
    (x_padding // 2, x_padding - x_padding // 2), (y_padding // 2, y_padding - y_padding // 2),
    (z_padding // 2, z_padding - z_padding // 2)), 'constant', constant_values=0)
    grid_3d = np.asarray(grid_3d, dtype=np.uint8)

    dim = str(grid_3d.shape[2]) + 'x' + str(grid_3d.shape[1]) + 'x' + str(grid_3d.shape[0])
    name = os.path.join(save_path, febname + '_uint8_' + dim + '.raw')
    with open(name, 'wb') as f:
        f.write(grid_3d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'error722/breast05_35_sf_200.feb')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='fat')

    args = parser.parse_args()
    print(args)
    setup(args, check_exists=True)
    febname = os.path.basename(args.febfile_name).split('.')[0]
    save_path_raw = os.path.join('recon_raw', febname)
    save_path_img = os.path.join('recon_img', febname)
    if not os.path.exists(save_path_raw):
        os.mkdir(save_path_raw)
    if not os.path.exists(save_path_img):
        os.mkdir(save_path_img)
    recon(save_path=save_path_raw, save_path_img=save_path_img, febname=febname)
