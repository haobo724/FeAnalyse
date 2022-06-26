import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import open3d as o3d
import cv2
from read import FEmapping


def test():
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

    for k in fat:
        k = list(float(i) for i in k)
        a.append(k)


    for j in tissue:
        j = list(float(i) for i in j)
        a.append(j)

    pointcloud_as_array = np.array(a) #shape is (7072, 3)
    pointcloud_as_array=np.around(pointcloud_as_array, 3)
    # hist, bin_edges=np.histogram(pointcloud_as_array[:,2])


    # print(np.max(pointcloud_as_array,axis=0),np.min(pointcloud_as_array,axis=0))
    maxium = np.max(abs(pointcloud_as_array),axis=0)
    pointcloud_as_array = pointcloud_as_array/maxium

    distance = np.max(pointcloud_as_array,axis=0)-np.min(pointcloud_as_array,axis=0)

    print(distance)
    xyz_d  = distance/np.array([Rows,Columns,NumberofFrames])
    print(xyz_d)
    res, num = np.unique(pointcloud_as_array[:, 2], return_counts=True)
    index = np.where(num > 500)
    print(res[index])


    # _ = plt.hist(pointcloud_as_array[:,2], bins='auto')  # arguments are passed to np.histogram
    # plt.show()
    # print()
    # pointcloud_as_array = pointcloud_as_array/(pointcloud_as_array.max(axis=0))
    # pointcloud_as_array = pointcloud_as_array+abs(np.min(pointcloud_as_array,axis=0))
    # pointcloud_as_array = pointcloud_as_array*[1.08435303, 0.88868057, 0.57180586]
    # X_codinate = pointcloud_as_array[:,0]/ pointcloud_as_array[:,0].max()
    # y_codinate = pointcloud_as_array[:,1]
    # z_codinate = pointcloud_as_array[:,2]

    Z = np.min(pointcloud_as_array[:,2])
    print('Z:',Z)
    d = 0.065
    plt.ion()
    while Z<2:
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
        xy_codinate = final_pointcloud_array[:,:2]
        blank = np.zeros((Rows*2,Columns*2))
        y = np.arange(0,2,xyz_d[0]/2)-1
        x = np.arange(0,2,xyz_d[1]/2)-1

        for index1,i in enumerate(x[:Columns*2]):
            for index2,j in  enumerate(y[:Rows*2]):
                pos  = (i,j)
                distances = np.sqrt(np.sum(np.asarray(xy_codinate-pos) ** 2, axis=1))
                if np.min(distances) <= xyz_d[0]*2:
                    blank[index2, index1] = 255
        plt.pause(0.1)
        plt.imshow(blank)
        plt.show()
        cv2.imwrite('recon2/'+str(Z)+'.jpg',blank)
        Z+=xyz_d[2]

    # Create Open3D point cloud object from array
    # final_pointcloud = o3d.geometry.PointCloud()
    # final_pointcloud.points = o3d.utility.Vector3dVector(final_pointcloud_array)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=final_pointcloud,
    #                                                                voxel_size=xyz_d[0])
    # o3d.visualization.draw_geometries([voxel_grid])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    # vis.get_render_option().point_size = 5.0
    # vis.add_geometry(voxel_grid)
    # vis.capture_screen_image("file.raw", do_render=True)
    # vis.destroy_window()

    # imgetoshow3DFast(pointcloud_as_array)

if __name__ == '__main__':
    test()



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--febfile_name', type=str, help='', default=r'breast_new_sg.feb')
    # parser.add_argument('--dcm_name', type=str, help='', default='Breast06_left.dcm')
    # parser.add_argument('--Node_name', type=str, help='', default='breast')
    # parser.add_argument('--mat_name', type=str, help='', default='fat')
    #
    # args = parser.parse_args()
    # print(args)
    #
    # # step 1 : initialize all name parameters
    # febfile_name = args.febfile_name
    # dcm_name = args.dcm_name
    # Node_name = args.Node_name
    # mat_name = args.mat_name
    #
    #
    # with open(febfile_name, "rb") as f:
    #     feb_data = f.read()
    # fe = FEmapping()
    # fe.read_dicom(dcm_name)
    #
    # # step 3 : analyse feb file
    # '''
    # workflow :
    # according to matname in  'MeshDomains - SolidDomain' to know the element name ,i.e 'Part1'
    # according to element name to get element infos
    # according to args.Node_name to get node infos
    #
    # '''
    #
    # Fat_element = fe.get_Ele(feb_data, 'Part62')
    # Tissue_element = fe.get_Ele(feb_data, 'Part61')
    # Fat_element_dic, Fat_element_dic_full = fe.get_node_single_ele(Fat_element)
    # Tissue_element_dic, Tissue_element_dic_full = fe.get_node_single_ele(Tissue_element)
    # node_dic = fe.get_node_dic(feb_data, f'{Node_name}')
    # fe.get_center_from_element(Fat_element_dic, node_dic)
    # fe.get_center_from_element(Tissue_element_dic, node_dic,cls='Tissue')
    #
    #
    #
    #
    #
    # fat_center, tissue_center = fe.get_center()
    # with open("fat_new.pkl", 'wb') as f:
    #     pickle.dump(fat_center, f)
    # with open(f"tissue_new.pkl", 'wb') as f:
    #     pickle.dump(tissue_center, f)