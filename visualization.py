import numpy as np
import pickle
import open3d


def imgetoshow3DFast(imgcloudflatten):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(imgcloudflatten)
    open3d.visualization.draw_geometries([point_cloud])


def show_result(name='node_bad.pkl'):
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
            k = list(float(i) for i in k)
            a.append(k)
            # a.append([0,0,0])
            # a.append([0,0,1])
            # a.append([1,0,0])
    cloud = np.array(a)
    # test=np.array([[1,2,3],[4,5,6],[12,22,0]])
    part = name[:-3]
    print(len(tst), 'elements shoule be', f"{part}")
    # print(np.max(cloud, axis=0))
    # print(np.min(cloud, axis=0))
    imgetoshow3DFast(cloud)


if __name__ == '__main__':

    show_result()


    