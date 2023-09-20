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
import tf

from std_srvs.srv import Empty, EmptyResponse
from panoptic_mapping_msgs.msg import DetectronLabel, DetectronLabels

# better use with conda deactivate

class CKADataPlayer(object):
    def __init__(self):
        """  Initialize ros node and read params """
        # params
        self.data_path = rospy.get_param(
            '~data_path', 'Datasets/flat_dataset/run1')
        self.global_frame_name = rospy.get_param('~global_frame_name', 'world')
        self.sensor_frame_name = rospy.get_param('~sensor_frame_name',
                                                 "depth_cam")
        self.body_frame_name = rospy.get_param('~body_frame_name',
                                                 "base_link")
        self.use_detectron = rospy.get_param('~use_detectron', False)
        self.play_rate = rospy.get_param('~play_rate', 1.0)
        self.wait = rospy.get_param('~wait', False)
        self.max_frames = rospy.get_param('~max_frames', 1e9)
        self.refresh_rate = 2000  # Hz (0.5ms)
        
        # transformation from camera to body frame
        self.T_bc = np.array([[1,0,0,0],
                              [0,0,1,0],
                              [0,-1,0,0],
                              [0,0,0,1]]) 
        
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

        # setup
        self.cv_bridge = CvBridge()
        stamps_file = os.path.join(self.data_path, 'timestamps.csv')
        self.times = []
        self.ids = []
        self.current_index = 0  # Used to iterate through
        if not os.path.isfile(stamps_file):
            rospy.logfatal("No timestamp file '%s' found." % stamps_file)
        with open(stamps_file, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if row[0] == "ImageID":
                    continue
                self.ids.append(str(row[0]))
                self.times.append(float(row[1]) / 1e6) # unit: s # different from the flat dataset, they reach ns, here we only reach micro second

        self.ids = [x for _, x in sorted(zip(self.times, self.ids))]
        self.times = sorted(self.times)
        # print(self.times)
        self.times = [(x - self.times[0]) / self.play_rate for x in self.times]
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
        if self.current_index >= len(self.times):
            rospy.loginfo("Finished playing the dataset.")
            rospy.signal_shutdown("Finished playing the dataset.")
            return

        # Check the time.
        now = rospy.Time.now()
        if self.start_time is None:
            self.start_time = now
            self.last = now
        
        if self.times[self.current_index] > (now - self.start_time).to_sec():
            return

        print("[Data player] frame id:", self.current_index, "timestamp:", now)

        interval_ms = (now - self.last) / 1e6
        print("interval (ms):", interval_ms)

        # Get all data and publish.
        file_id = os.path.join(self.data_path, self.ids[self.current_index])

        # Color.
        color_file = file_id + "_color.png"
        depth_file = file_id + "_depth.npy"
        pose_file = file_id + "_pose.txt"
        files = [color_file, depth_file, pose_file]
        if self.use_detectron:
            pred_file = file_id + "_predicted.png"
            labels_file = file_id + "_labels.json"
            files += [pred_file, labels_file]
        else: # ground truth
            pred_file = file_id + "_segmentation.npy"
            files.append(pred_file)
        for f in files:
            if not os.path.isfile(f):
                rospy.logwarn("Could not find file '%s', skipping frame." % f)
                self.current_index += 1
                return

        # Load and publish Color image.
        cv_img = cv2.imread(color_file)
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.sensor_frame_name
        self.color_pub.publish(img_msg)

        # Load and publish ID image.
        # cv_img = cv2.imread(pred_file)
        cv_img = np.load(pred_file) # use npy file instead
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "8UC1")
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.sensor_frame_name
        self.id_pub.publish(img_msg)

        # Load and publish depth image. These are optional.
        # cv_img = PilImage.open(depth_file)
        cv_img = np.load(depth_file) # use npy file instead
        img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(cv_img), "32FC1")
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.sensor_frame_name
        self.depth_pub.publish(img_msg)

        # Load and publish transform.
        if os.path.isfile(pose_file):
            pose_data = [float(x) for x in open(pose_file, 'r').read().split()]
            transform = np.eye(4) # the input is the pose of the body frame
            for row in range(4):
                for col in range(4):
                    transform[row, col] = pose_data[row * 4 + col]

            # transform_in_camera_frame = np.matmul(transform, self.T_bc) # we need the pose of the camera frame
            # transform = transform_in_camera_frame
            
            rotation = tf.transformations.quaternion_from_matrix(transform)

            self.tf_broadcaster.sendTransform(
                (transform[0, 3], transform[1, 3], transform[2, 3]), rotation,
                now, self.body_frame_name, self.global_frame_name) # tf is actually used

            # static transformation
            self.tf_broadcaster.sendTransform(
                (self.T_bc[0, 3], self.T_bc[1, 3], self.T_bc[2, 3]), self.rot_bc,
                now, self.sensor_frame_name, self.body_frame_name) # tf is actually used
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.global_frame_name
        pose_msg.pose.position.x = pose_data[3]
        pose_msg.pose.position.y = pose_data[7]
        pose_msg.pose.position.z = pose_data[11]
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]
        self.pose_pub.publish(pose_msg)

        self.current_index += 1
        self.last = now

        if self.current_index > float(self.max_frames):
            rospy.signal_shutdown("Played reached max frames (%i)" %
                                  self.max_frames)


if __name__ == '__main__':
    rospy.init_node('cka_data_player')
    cka_data_player = CKADataPlayer()
    rospy.spin()