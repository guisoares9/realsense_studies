# First import library
from numpy.lib.polynomial import _polyint_dispatcher
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
# Import Open3D library
import open3d as o3d
import time
import math
import scipy.spatial.distance

def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    preprocessing :
        use `_view_`, `_new_view_` or `_reshape_` with structured/recarrays
    Parameters
    ----------
    a, b : array like
        Inputs, list, tuple, array in 1, 2 or 3D form
    metric : string
        euclidean ('e', 'eu'...), sqeuclidean ('s', 'sq'...),
    Notes
    -----
    mini e_dist for 2d points array and a single point
    def e_2d(a, p):
            diff = a - p[np.newaxis, :]  # a and p are ndarrays
            return np.sqrt(np.einsum('ij,ij->i', diff, diff))
    See Also
    --------
    cartesian_dist : function
        Produces pairs of x,y coordinates and the distance, without duplicates.
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file", default="guilherme-rgbd.bag")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

origin = [0,0,0]
# Declare pointcloud object
pcd = o3d.geometry.PointCloud()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, args.input)

# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

vis = o3d.visualization.Visualizer()
vis.create_window()
# initpcd = o3d.io.read_point_cloud("sync.ply")
# vis.add_geometry(initpcd)

decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 1)
filters = [rs.disparity_transform(),
           rs.spatial_filter(),
           rs.temporal_filter(),
           rs.disparity_transform(False)]

# Streaming loop
try:
    while True:
        t = time.time()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Filter depth frame
        depth_frame = decimate.process(aligned_depth_frame)
        for f in filters:
            depth_frame = f.process(depth_frame)

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        depth_image = np.array(depth_frame.get_data())
        scaled_depth_image = depth_image*depth_scale # Transform depths values to a real world depth value in meters
        color_image = np.asanyarray(color_frame.get_data())


        # Get a point cloud and transform it in a xyz array
        
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vtx = points.get_vertices()
        vtx = np.asanyarray(vtx)
        vtx = vtx.tolist()
        vtx = np.array(vtx)

        # math.sqrt method for background removal
        num_rows = vtx.shape[0] # Number of rows in vtx (921600,3)
        num_columns = vtx.shape[1] # Number of columns in vtx
        for r in range(num_rows):
            for c in range(num_columns):
                depth = math.sqrt(vtx[r][0]**2 + vtx[r][1]**2 + vtx[r][2]**2) # Calculates distance to origin
                if depth > 1: # Background removal: if the distance is more than 1 meter, filter it
                    vtx[r,:] = 0

        # Einsum method for background removal
        #num_rows = vtx.shape[0] # Number of rows in vtx (921600,3)
        #for r in range(num_rows):
            #dist = e_dist(vtx[r], origin, metric='e')
            #if dist > 1: # Background removal: if the distance is more than 1 meter, filter it
                #vtx[r,:] = 0

        # np.linalg.norm method for background removal
        #num_rows = vtx.shape[0] # Number of rows in vtx (921600,3)
        #for r in range(num_rows):
            #dist = np.linalg.norm(vtx[r]-origin)
            #if dist > 1: # Background removal: if the distance is more than 1 meter, filter it
                #vtx[r,:] = 0

        # scipy.spatial.distance.cdist method for background removal
        #num_rows = vtx.shape[0] # Number of rows in vtx (921600,3)
        #num_columns = vtx.shape[1] # Number of columns in vtx
        #for r in range(num_rows):
            #d = scipy.spatial.distance.cdist(vtx[r],origin)
            #if d > 1: # Background removal: if the distance is more than 1 meter, filter it
                #vtx[r,:] = 0
                
        # np.sum and np.sqrt method for background removal
        #num_rows = vtx.shape[0] # Number of rows in vtx (921600,3)
        #or r in range(num_rows):
            #squared_dist = np.sum((vtx[r]-origin)**2, axis=0)
            #dist = np.sqrt(squared_dist)
            #if dist > 1: # Background removal: if the distance is more than 1 meter, filter it
                #vtx[r,:] = 0

        # Pass images, which is an array, to Open3D.o3d.geometry.PointCloud
        pcd.points = o3d.utility.Vector3dVector(vtx)
        # o3d.io.write_point_cloud("sync.ply", pcd)
        
        print(f"{time.time()-t} segundos")  
        
        trans = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(trans)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        #cv2.imshow("frame", color_image)
        
        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', imgResize)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    vis.destroy_window()
    pipeline.stop()