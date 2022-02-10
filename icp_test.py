# First import library
from numpy.lib.polynomial import _polyint_dispatcher
# Import Open3D library
import open3d as o3d
# Import copy
import copy
import numpy as np
import time

# Drawing surface matching result
def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# Load brain and head point cloud
brain = o3d.io.read_point_cloud("brain.pcd")
head = o3d.io.read_point_cloud("head.pcd")

# Forward head position
trans_head= [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
head.transform(trans_head)
#head.scale(1.1, center=head.get_center())

# # Find the inicial position
# trans_brain= [[1, 0, 0, 0], [0, 0, 1, 1260], [0, -1, 0, -165], [0, 0, 0, 1]]
# brain.transform(trans_brain)
# o3d.visualization.draw_geometries([head, brain])

# ICP
threshold = 20
trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 0, 1, 195],
                                [0, -1, 0, 15], 
                                [0, 0, 0, 1]])

# print("Initial alignment")
# evaluation = o3d.pipelines.registration.evaluate_registration(brain, head, threshold, trans_init)
# print("evaluation: ", evaluation.transformation)

t = time.time()
print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(brain, head, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
print(f"{time.time()-t} segundos")

print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
#print("")
draw_registration_result(brain, head, reg_p2p.transformation)
