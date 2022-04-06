# Newer college dataset
# website: https://ori-drs.github.io/newer-college-dataset/
# download link: https://drive.google.com/drive/u/0/folders/15lTH5osZzZlDpcW7oXfR_2t8TNssNARS

import rosbag
import rospy
import numpy as np
from numpy.linalg import inv
import csv
import os
import tf
import tf2_ros
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform, PoseStamped
from nav_msgs.msg import Odometry

bag_base_path = "/media/yuepan/SeagateNew/1_data/NewerCollege/2020-ouster-os1-64-realsense/long_experiments"
bag_file_name = "2020_long_filtered_07.bag"

bag_file_path = os.path.join(bag_base_path, bag_file_name)

## https://ori-drs.github.io/newer-college-dataset/stereo-cam/platform-stereo/
# in base frame [Note: The base frame is defined to be coincident with the left camera frame; however, it follows the robotic frame (x-forward, y-left, z-up).]
pose_csv_filename = "registered_poses.csv"
pose_csv_path = os.path.join(bag_base_path, pose_csv_filename)

pose_csv_file = open(pose_csv_path)
csvreader = csv.reader(pose_csv_file)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

pose_csv_file.close()

pose_data = np.array(rows)
#print(pose_data)

compression = rosbag.Compression.NONE
#bag = rosbag.Bag(bag_file_path)
bag = rosbag.Bag(bag_file_path, 'a')

start_time = bag.get_start_time()
end_time = bag.get_end_time()
# for topic, msg, t in bag.read_messages(topics=['/tf_static']):
#     print(msg)

# #  (/os1_imu to /camera_link): [m]
# camera_link is the same as the left camera frame
T_ci = np.array([[ 0.70992163, -0.70414167,  0.01399269,  0.01562753], 
                 [ 0.02460003,  0.00493623, -0.99968519, -0.01981648], 
                 [ 0.70385092,  0.71004236,  0.02082624, -0.07544143],
                 [ 0.        ,  0.        ,  0.        ,  1.        ]])
T_ic = inv(T_ci)

# #  (/os1_lidar to /os1_imu): [m]
T_is = np.eye(4) # /os1_sensor to /os1_imu
T_is[0:3,3] = np.array([0.006253, -0.011775, 0.007645])

T_os = tf.transformations.quaternion_matrix(np.array([0.0, 0.0, 1.0, 0.0])) # /os1_sensor to /os1_lidar
T_os[0:3,3] = np.array([0.0, 0.0, 0.03618])

T_io = np.matmul(T_is, inv(T_os))
T_oi = inv(T_io)

# left camera to base frame [x,y,z to z,-x,-y]
T_bc = np.array([[ 0., 0., 1., 0.], 
                 [ -1., 0., 0., 0.], 
                 [ 0., -1., 0., 0.],
                 [ 0., 0., 0., 1.]])

for pose_data_single in pose_data:
    tf_dy_msg = TFMessage()
    tf_dy_transform = TransformStamped()
        
    #tf_dy_transform.header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
    ts = float(pose_data_single[0]) + float(pose_data_single[1]) * 1e-9
    if ts < start_time:
        continue
    if ts > end_time:
        break
        
    # print(ts)
    tf_dy_transform.header.stamp = rospy.Time.from_sec(ts)
    # print(tf_dy_transform.header.stamp)

    tf_dy_transform.header.frame_id = 'world'
    # tf_dy_transform.child_frame_id = 'camera_link'  # in RS_C1 frame
    tf_dy_transform.child_frame_id = 'os1_lidar' # after the transformation

    dy_tf = Transform()

    T_wb = tf.transformations.quaternion_matrix(pose_data_single[5:9])
    T_wb[0:3,3] = pose_data_single[2:5]

    T_wc = np.matmul(T_wb, T_bc)

    T_wo = np.matmul(np.matmul(T_wc, T_ci), T_io) # lidar in world frame

    # T_wi = T_wc # lidar imu in world frame
    # T_wo = np.matmul(T_wi, T_oi) # lidar in world frame

    t_wo = T_wo[0:3, 3]
    q_wo = tf.transformations.quaternion_from_matrix(T_wo)

    dy_tf.translation.x = float(t_wo[0])
    dy_tf.translation.y = float(t_wo[1])
    dy_tf.translation.z = float(t_wo[2])

    dy_tf.rotation.x = float(q_wo[0])
    dy_tf.rotation.y = float(q_wo[1])
    dy_tf.rotation.z = float(q_wo[2])
    dy_tf.rotation.w = float(q_wo[3])

    tf_dy_transform.transform = dy_tf
    tf_dy_msg.transforms.append(tf_dy_transform)

    print(tf_dy_msg)

    bag.write('/tf', tf_dy_msg, t=tf_dy_msg.transforms[0].header.stamp)

print(bag)
bag.close()
