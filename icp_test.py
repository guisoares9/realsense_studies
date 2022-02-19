# First import library
from numpy.lib.polynomial import _polyint_dispatcher
# Import Open3D library
import open3d as o3d
# Import copy
import copy
import numpy as np
import time
import math

# Visualize surface matching result
def draw_registration_result(source, target, brain, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    brain_temp = copy.deepcopy(brain)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    new_transformation = np.matmul(transformation, trans_brain_Z)
    brain_temp.transform(new_transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, brain_temp])

# Pre-process the point cloud for global registration
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size) # Downsample

    radius_normal = voxel_size * 100
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) # Normal estimation

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) # FPFH = describes the local geometric property of a point
    return pcd_down, pcd_fpfh

# Global registration function
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# Load brain and head point cloud
head_target = o3d.io.read_point_cloud("head.pcd")
head_source = o3d.io.read_point_cloud("head_1.pcd")
brain = o3d.io.read_point_cloud("brain.pcd")

# Forward head_target position
trans_head= [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
head_target.transform(trans_head)

# Scale head_source, because it is a little bit smaller than the head_target
head_source.scale(1.1, center=head_source.get_center())

# Find the inicial position
ang_deg = 168 # Em graus
tx = 0
ty = -173
tz = 160
# trans_brain_X = [[1, 0, 0, tx], 
#                 [0, math.cos(ang_deg*math.pi/180), -math.sin(ang_deg*math.pi/180), ty], 
#                 [0, math.sin(ang_deg*math.pi/180), math.cos(ang_deg*math.pi/180), tz], 
#                 [0, 0, 0, 1]]
# trans_brain_Y = [[math.cos(ang_deg*math.pi/180), 0, -math.sin(ang_deg*math.pi/180), tx], 
#                 [0, 1, 0, ty], 
#                 [math.sin(ang_deg*math.pi/180), 0, math.cos(ang_deg*math.pi/180), tz], 
#                 [0, 0, 0, 1]]
trans_brain_Z = [[math.cos(ang_deg*math.pi/180), -math.sin(ang_deg*math.pi/180), 0, tx], 
                [math.sin(ang_deg*math.pi/180), math.cos(ang_deg*math.pi/180), 0, ty], 
                [0, 0, 1, tz], 
                [0, 0, 0, 1]]
#brain.transform(trans_brain_Z)
#o3d.visualization.draw_geometries([head_source, brain])



# # ICP: POINT-TO-POINT and POINT-TO-PLANE
# threshold = 20
# trans_init = np.asarray([[1, 0, 0, 0],
#                                 [0, 0, 1, 0],
#                                 [0, 1, 0, 0], 
#                                 [0, 0, 0, 1]])

# # print("Initial alignment")
# # evaluation = o3d.pipelines.registration.evaluate_registration(brain, head, threshold, trans_init)
# # print("evaluation: ", evaluation.transformation)

# t = time.time()
# print("Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(brain, head, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
# print(f"{time.time()-t} segundos")

# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(brain, head, reg_p2p.transformation)



# # ICP: GLOBAL REGISTRATION
# voxel_size = 0.001*1000 # in meters. *1000 is just for scaling
# source_down, source_fpfh = preprocess_point_cloud(brain, voxel_size)
# target_down, target_fpfh = preprocess_point_cloud(head, voxel_size)
# #o3d.visualization.draw_geometries([source_down])
# #o3d.visualization.draw_geometries([target_down])

# t = time.time()
# print("Apply Global Registration ICP")
# result_ransac = execute_global_registration(source_down, target_down,
#                                             source_fpfh, target_fpfh,
#                                             voxel_size)
# print(f"{time.time()-t} segundos")
# print(result_ransac)
# print(result_ransac.transformation)
# draw_registration_result(source_down, target_down, result_ransac.transformation)



# ICP: Global registration and point-to-plane

voxel_size = 0.006*1000 # in meters. *1000 is just for scaling
source_down, source_fpfh = preprocess_point_cloud(head_source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(head_target, voxel_size)
t = time.time()
print("Apply Global Registration ICP")
result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(f"{time.time()-t} segundos")
print(result_ransac)
print(result_ransac.transformation)


threshold = 20
trans_init = result_ransac.transformation
t = time.time()
print("Apply point-to-plane ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
print(f"{time.time()-t} segundos")
print(reg_p2p)
print(reg_p2p.transformation)
draw_registration_result(source_down, target_down, brain, reg_p2p.transformation)