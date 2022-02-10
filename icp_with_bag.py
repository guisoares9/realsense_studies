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
# Import time library
import time
# Import realsense helper function
from helper_functions import convert_depth_frame_to_pointcloud
# Import threading
import threading
# Import copy
import copy

# def updateViewer(vis, pcd):

#     vis.add_geometry(pcd)
#     vis.add_geometry(brain)
#     vis.poll_events()
#     vis.update_renderer()
#     return None

# Drawing surface matching result
def draw_registration_result(source, target, transformation):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# Load brain point cloud
brain = o3d.io.read_point_cloud("brain.pcd")

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

# First time creating the thread
#vis_trd = threading.Thread(target = updateViewer, args = [vis, pcd])

#vis_trd.start()

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
        cropped_depth_image = depth_image[50:500, 500:900] # Bounding box: cropp shoulders
        scaled_depth_image = cropped_depth_image*depth_scale # Transform depths values to a real world depth value in meters
        color_image = np.asanyarray(color_frame.get_data())


        # Convert depth frame to point cloud and do a background removal

        intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics # Intrinsics parameters
        num_rows = scaled_depth_image.shape[0] # Number of rows in scaled_depth_image
        num_columns = scaled_depth_image.shape[1] # Number of columns in scaled_depth_image

        for r in range(0, num_rows):
            for c in range(0, num_columns):
                if scaled_depth_image[r][c] > 0.75: # Background removal: if the distance is more than 1 meter, filter it
                    scaled_depth_image[r][c] = 0

        pointcloud = convert_depth_frame_to_pointcloud(scaled_depth_image, intrin)
        pointcloud = np.asanyarray(pointcloud)# Transform a tuple in a an array
        xyz = pointcloud.tolist()
        xyz = np.array(xyz).T*1000 # Multiply by 1000 to ajust scale


        # Convert xyz, which is an array, to Open3D.o3d.geometry.PointCloud
        pcd.points = o3d.utility.Vector3dVector(xyz)

        print(f"{time.time()-t} segundos")

        # Rotate pcd to 180 in y (the original pcd is upsidedown)
        trans_pcd = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(trans_pcd)

        # # Find the inicial position of brain
        # trans_brain = [[1, 0, 0, -300], [0, 0, 1, 90], [0, -1, 0, -750], [0, 0, 0, 1]]
        # brain.transform(trans_brain)
        # o3d.visualization.draw_geometries([pcd, brain])

        #downpcd = pcd.voxel_down_sample(voxel_size=1)
        #down_xyz = np.asarray(downpcd.points)
        #o3d.visualization.draw_geometries([downpcd])


        # ICP
        threshold = 20
        trans_init = np.asarray([[1, 0, 0, -300],
                                [0, 1, 0, 90],
                                [0, 0, 1, -750], 
                                [0, 0, 0, 1]])
        #draw_registration_result(brain, pcd, trans_init)

        # print("Initial alignment")
        # evaluation = o3d.pipelines.registration.evaluate_registration(brain, pcd, threshold, trans_init)
        # print("evaluation: ", evaluation.transformation)

        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(brain, pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(brain, pcd, reg_p2p.transformation)
        
        #vis_trd.join()
        #vis_trd = threading.Thread(target = updateViewer, args = [vis, pcd])

        #vis_trd.start()
        
        #cv2.imshow("frame", color_image)

        # pcd_center = pcd.get_center()
        # brain_center = brain.get_center()
        # print("pcd:", pcd_center)
        # print("brain:", brain_center)

        #o3d.visualization.draw_geometries([pcd, brain])
        # vis.add_geometry(pcd)
        # vis.add_geometry(brain)
        # vis.poll_events()
        # vis.update_renderer()
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    vis.destroy_window()
    pipeline.stop()
    #vis_trd.join()