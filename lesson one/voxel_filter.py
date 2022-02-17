# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np
import random
from pyntcloud import PyntCloud


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, filter_mode='centroid'):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    max_value = np.max(point_cloud, axis=0)
    min_value = np.min(point_cloud, axis=0)
    # 定义 voxel grid size
    voxel_grid_size = leaf_size
    # 计算voxel grid 的 dimension
    # 这里向上取整,因为就相当于计算list的size, 向上取整了就不用再+1了
    dimension = np.ceil((max_value - min_value) / voxel_grid_size)
    indices = (point_cloud - min_value) // voxel_grid_size  # 地板除,向下取整,索引从0开始
    # 假设把voxel 放到一个数组中 计算其index
    h = indices[:, 0] + indices[:, 1] * dimension[0] + indices[:, 2] * dimension[0] * dimension[1]
    # h 这个算法,就是相当于展开三维索引到 一个维度的索引

    for i in np.unique(h):
        points = point_cloud[h == i]
        if filter_mode == 'centroid':
            filtered_points.append(np.mean(points, axis=0))
        else:
            filtered_points.append(random.choice(points))  # 随机采样这个接口很方便
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    cat_index = 0  # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/Users/wangyu/Desktop/点云算法/第一张/modelnet40_normal_resampled/'  # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index], cat[cat_index] + '_0001.txt')  # 默认使用第一个点云

    # 加载自己的点云文件
    point_cloud_raw = np.genfromtxt(filename, delimiter=',')
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_raw[:, 0:3])

    # 转成open3d能识别的格式
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    # 调整这个voxel grid size可以看到不同的采样结果
    filtered_cloud = voxel_filter(point_cloud_raw[:, 0:3], 0.1)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
