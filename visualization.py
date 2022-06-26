import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import open3d as o3d
import cv2

def imgetoshow3DFast(imgcloudflatten):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(imgcloudflatten)
    point_cloud.scale(1 / np.max(point_cloud.get_max_bound() - point_cloud.get_min_bound()),
              center=point_cloud.get_center())
    o3d.visualization.draw_geometries([point_cloud])

def imgetoshow3DVol(imgcloudflatten):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(imgcloudflatten)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=point_cloud,
                                                                   voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])

def show_result(name='node_bad.pkl',vox =False):
    with open(name, 'rb') as f:
        tst = pickle.load(f)
    a = []
    try:
        for k in tst.values():
            k = list(float(i) for i in k)
            a.append(k)
            # a.append([0,0.1,0])
            a.append([0, 0, 0.2])
            a.append([0, 0, -0.2])

            a.append([0.2, 0, 0])
            a.append([-0.2, 0, 0])
    except:
        for k in tst:
            k = list(float(i) for i in k) #k is [x,y,z]
            a.append(k)
            # a.append([0,0,0])
            # a.append([0,0,1])
            # a.append([1,0,0])
    cloud = np.array(a) #shape is (7072, 3)
    # print(cloud.shape)
    # test=np.array([[1,2,3],[4,5,6],[12,22,0]])
    part = name[:-3]
    print(len(tst), 'elements shoule be', f"{part}")
    # print(np.max(cloud, axis=0))
    # print(np.min(cloud, axis=0))
    if vox:
        imgetoshow3DVol(cloud)
        imgetoshow3DFast(cloud)
    else:
        imgetoshow3DFast(cloud)


def fake_3d_breast():
    point_cloud = o3d.geometry.PointCloud()
    path = r'F:\Siemens\MA\Mini_study\N\mask\131_mask.tiff'
    paths = glob.glob(r'F:\Siemens\MA\Mini_study\N\mask\*.tiff')
    for path in paths:
        img = cv2.imread(path,0)
        img_copy =img.astype(np.float64)
        temp =(img_copy*255//2).astype(np.uint8)
        print(np.unique(temp))
        cv2.imshow('test',temp)
        # cv2.waitKey()
        points_A=np.array(np.where(img==1)).transpose((1,0))
        points_B=np.array(np.where(img==2)).transpose((1,0))
        # cloud_A = [i.append(np.random.randint(10,30)) for i in points_A]
        cloud_A = [np.append(i,np.random.randint(100,200)) for i in points_A]
        cloud_B = [np.append(i,np.random.randint(90,150)) for i in points_B]
        cloud_A = np.array(cloud_A+cloud_B)
        point_cloud.points = o3d.utility.Vector3dVector(cloud_A)

        o3d.visualization.draw_geometries([point_cloud])
        print('voxelization')

if __name__ == '__main__':
    pass


    