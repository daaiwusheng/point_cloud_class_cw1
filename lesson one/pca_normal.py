# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    pass
    # 作业1
    # 屏蔽开始

    # 屏蔽结束

    # if sort:
    #     sort = eigenvalues.argsort()[::-1]
    #     eigenvalues = eigenvalues[sort]
    #     eigenvectors = eigenvectors[:, sort]

    # return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    cat_index = 0  # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/Users/wangyu/Desktop/点云算法/第一张/modelnet40_normal_resampled/'  # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index], cat[cat_index] + '_0001.txt')  # 默认使用第一个点云

    # 加载自己的点云文件
    # 读取点云txt文件
    cloud_points = np.genfromtxt(filename, delimiter=",")
    cloud_points = pd.DataFrame(cloud_points[:, 0:3])
    cloud_points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(cloud_points)

    # 加载原始点云
    # point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 从点云中获取点，只对点进行处理
    cloud_points = point_cloud_pynt.points
    print('total cloud_points number is:', cloud_points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(cloud_points)
    point_cloud_vector = v[:, 2]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # o3d.visualization.draw_geometries([point_cloud_o3d])

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始

    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
