#!/usr/bin/env python3

# updated
import os
import json
import csv

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from PIL import Image as PilImage
import numpy as np
from numpy.linalg import inv
import tf

from std_srvs.srv import Empty, EmptyResponse
from panoptic_mapping_msgs.msg import DetectronLabel, DetectronLabels

# better use with conda deactivate

def correct_jump_in_tf_poses(tf_poses: np.ndarray):
    """ There are jumps in the poses from tf. Correct these
    """
    for i in range(1, tf_poses.shape[0]-1):
        if tf_poses[i+1,0,3]-tf_poses[i,0,3]>1e-2: # 1cm  
            print('Jump at %d .th pose (%f to %f), correct' % (i,  tf_poses[i+1,0,3], tf_poses[i,0,3]))
            rel_offset = tf_poses[i-1,0,3] - tf_poses[i,0,3]
            current = tf_poses[i+1,0,3]
            tf_poses[i+1:,0,3] = tf_poses[i+1:,0,3] - current + tf_poses[i,0,3] - rel_offset
        
    return tf_poses


class CKAFruitDataPlayer(object):
    def __init__(self):
        """  Initialize ros node and read params """
        # params
        self.data_path = rospy.get_param(
            '~data_path', '/media/yuepan/DATA/1_data/CKA/CKA_fruit/processed/2022_09_05/row1/before')
        self.global_frame_name = rospy.get_param('~global_frame_name', 'odom')
        self.sensor_frame_name = rospy.get_param('~sensor_frame_name',
                                                 "camera2_color_optical_frame")
        self.body_frame_name = rospy.get_param('~body_frame_name',
                                                 "base_link")
        self.use_detectron = rospy.get_param('~use_detectron', False)
        self.play_rate = rospy.get_param('~play_rate', 1.0)
        self.wait = rospy.get_param('~wait', False)
        self.max_frames = rospy.get_param('~max_frames', 1e9)
        self.min_frames = rospy.get_param('~min_frames', 0)
        self.refresh_rate = 2000  # Hz (0.5ms)
        
        # transformation from camera to body frame
        # self.T_bc = np.array([[1,0,0,0],
        #                       [0,0,1,0],
        #                       [0,-1,0,0],
        #                       [0,0,0,1]]) 
        
        T_b_cl = np.array( [[0,0,1,1.86],
                            [1,0,0,-0.237319],         
                            [0,1,0,2.0115],
                            [0,0,0,1]] )
        
        T_cl_cd = np.array( [[1,0,0,0.00012219222844578326],
                            [0,1,0,0.014925612136721611],         
                            [0,0,1,-1.1809094075942994e-6],
                            [0,0,0,1]] )
        
        T_cd_cc = np.array( [[0,0,1,0],
                            [-1,0,0,0],         
                            [0,-1,0,0],
                            [0,0,0,1]])
        
        self.T_bc = T_b_cl @ T_cl_cd @ T_cd_cc

        print(self.T_bc)
        
        self.rot_bc = tf.transformations.quaternion_from_matrix(self.T_bc)

        # ROS
        self.color_pub = rospy.Publisher("~color_image", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("~depth_image", Image, queue_size=100)
        self.id_pub = rospy.Publisher("~id_image", Image, queue_size=100)
        if self.use_detectron:
            self.label_pub = rospy.Publisher("~labels",
                                             DetectronLabels,
                                             queue_size=100)
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()
        #self.static_tf_broadcaster = tf.StaticTransformBroadcaster()

        self.cv_bridge = CvBridge()
        
        self.realsense_folder = os.path.join(self.data_path, 'realsense')
        self.color_folder = os.path.join(self.realsense_folder, 'color')
        self.depth_folder = os.path.join(self.realsense_folder, 'depth')
        self.masks_folder = os.path.join(self.realsense_folder, 'masks')

        # setup
        tf_file = os.path.join(self.data_path, 'rostf_poses.npz')
        self.ros_tfs = np.load(tf_file, allow_pickle=True)['arr_0'] # T_bw

        meta_tf_file = os.path.join(self.data_path, 'scaled_poses.npz')
        self.T_mcs = np.load(meta_tf_file, allow_pickle=True)['arr_0']

        # correct jump
        self.ros_tfs = correct_jump_in_tf_poses(self.ros_tfs)
        np.savez(tf_file.replace("poses", "poses_no_jump"), self.ros_tfs)

        self.T_wm = (inv(self.ros_tfs[0]) @ self.T_bc) @ inv(self.T_mcs[0])

        self.T_wcs = self.T_mcs

        self.frame_count = self.ros_tfs.shape[0]

        for i in range(self.frame_count):
            self.T_wcs[i] = self.T_wm @ self.T_mcs[i]

        np.savez(tf_file.replace("poses", "poses_metashape_aligned"), self.T_wcs)


        self.current_index = 0  # Used to iterate through

        frame_interval = 0.1 # s
        self.times = np.arange(0, (self.frame_count+1) * frame_interval, frame_interval) # in s
        # print(self.times)
        # self.times = [(x - self.times[0]) / self.play_rate for x in self.times]
        self.start_time = None
        self.last = None

        if self.wait:
            self.start_srv = rospy.Service('~start', Empty, self.start)
        else:
            self.start(None)

    def start(self, _):
        self.running = True
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.refresh_rate),
                                 self.callback)
        return EmptyResponse()

    def callback(self, _):
        # Check we should be publishing.
        if not self.running:
            return

        # Check we're not done.
        if self.current_index >= self.frame_count:
            rospy.loginfo("Finished playing the dataset.")
            rospy.signal_shutdown("Finished playing the dataset.")
            return
        
        if self.current_index < self.min_frames:
            to_publish = False
        else:
            to_publish = True

        # Check the time.
        now = rospy.Time.now()
        if self.start_time is None:
            self.start_time = now
            self.last = now
        
        if self.times[self.current_index] > (now - self.start_time).to_sec():
            return

        interval_ms = (now - self.last) / 1e6
        print("interval (ms):", interval_ms)
        
        now_bag = now - self.start_time
        print("[Data player] frame id:", self.current_index, "timestamp:", now_bag)

        # Get all data and publish.
        file_id = ("%05i" %self.current_index)

        # Color.
        color_file = os.path.join(self.color_folder, file_id+".png")
        depth_file = os.path.join(self.depth_folder, file_id+".npy")
        masks_file = os.path.join(self.masks_folder, file_id+".png")

        # print(color_file)

        # Load and publish Color image.
        if to_publish:
            cv_img = cv2.imread(color_file)
            img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
            img_msg.header.stamp = now_bag
            img_msg.header.frame_id = self.sensor_frame_name
            self.color_pub.publish(img_msg)

        # Load and publish ID image.
        # cv_img = cv2.imread(pred_file)
        if to_publish:
            cv_img2 = cv2.imread(masks_file, cv2.IMREAD_GRAYSCALE)
            # cv_img2 = cv2.rotate(cv_img2, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate here, comment later

            # print(np.max(cv_img2))
            img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img2, "8UC1")
            img_msg.header.stamp = now_bag
            img_msg.header.frame_id = self.sensor_frame_name
            self.id_pub.publish(img_msg)

        # Load and publish depth image. These are optional.
        if to_publish:
            # cv_img = PilImage.open(depth_file)
            if os.path.exists(depth_file): # may not exist
                cv_img3 = np.load(depth_file).astype(np.float32) # use npy file instead
                # print(np.mean(cv_img3))
                img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img3, "32FC1")
                img_msg.header.stamp = now_bag
                img_msg.header.frame_id = self.sensor_frame_name
                self.depth_pub.publish(img_msg)

        # cur_transform = inv(self.ros_tfs[self.current_index])

        cur_transform = self.T_wcs[self.current_index] # camera pose in metashape frame

        print(cur_transform)
        # save the new transform as the mapping pose input

        cur_quat = tf.transformations.quaternion_from_matrix(cur_transform)

        # self.tf_broadcaster.sendTransform(
        #     (cur_transform[0, 3], cur_transform[1, 3], cur_transform[2, 3]), cur_quat,
        #     now_bag, self.body_frame_name, self.global_frame_name) # T_wb

        # # static transformation
        # self.tf_broadcaster.sendTransform(
        #     (self.T_bc[0, 3], self.T_bc[1, 3], self.T_bc[2, 3]), self.rot_bc,
        #     now_bag, self.sensor_frame_name, self.body_frame_name) 

        # if to_publish:
        self.tf_broadcaster.sendTransform(
            (cur_transform[0, 3], cur_transform[1, 3], cur_transform[2, 3]), cur_quat,
            now_bag, self.sensor_frame_name, self.global_frame_name) # T_wc
        
        self.current_index += 1
        self.last = now

        if self.current_index > float(self.max_frames):
            rospy.signal_shutdown("Played reached max frames (%i)" %
                                  self.max_frames)


if __name__ == '__main__':
    rospy.init_node('cka_fruit_data_player')
    cka_fruit_data_player = CKAFruitDataPlayer()
    rospy.spin()