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

# Declare pointcloud object, for calculating pointclouds and texture mappings
#pc = rs.pointcloud()

# We want the points object to be persistent so we can display the last cloud when a frame drops
#points = rs.points()

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

        depth_image = np.asanyarray(depth_frame.get_data())
        scaled_depth_image = depth_image*depth_scale # Transform depths values to a real world depth value in meters
        color_image = np.asanyarray(color_frame.get_data())

        # Transform a 2D pixel and depth information into a xyz coordinates
    
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vtx = points.get_vertices()
        vtx = np.array(vtx)
        vtx = [list(x) for x in vtx]
        vtx = np.array(vtx)
        

        # # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))
        # imgResize = cv2.resize(images,(640,480))

        #print(depth_image.shape)
        #imgReshape = depth_image.reshape(307200, 3)
        #print(depth_image)

        # Pass images, which is an array, to Open3D.o3d.geometry.PointCloud
        pcd.points = o3d.utility.Vector3dVector(vtx)
        # o3d.io.write_point_cloud("sync.ply", pcd)
        
        print(f"{time.time()-t} segundos")
        
        
        trans = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(trans)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        cv2.imshow("frame", color_image)
        
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