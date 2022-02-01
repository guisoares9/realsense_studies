import open3d as o3d

mesh = o3d.io.read_triangle_mesh("head_2.stl")
pointcloud = mesh.sample_points_poisson_disk(50000)

# you can plot and check
#o3d.visualization.draw_geometries([mesh])
o3d.io.write_point_cloud("head_2.pcd", pointcloud, write_ascii=False, compressed=False, print_progress=False)
o3d.visualization.draw_geometries([pointcloud])