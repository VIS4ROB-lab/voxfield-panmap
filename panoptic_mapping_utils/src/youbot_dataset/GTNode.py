#!/usr/bin/env python3

import rospy
import pandas as pd
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image 
from geometry_msgs.msg import PoseStamped, Transform
import tf
from tf import transformations as ts
import numpy as np
camera_z = 0.63
min_z = -1.55708286

class GTNode():

	def __init__(self):

		csvPath = rospy.get_param('~csvPath')
		self.df = pd.read_csv(csvPath)
		multi = rospy.get_param('~multiCam')

		self.body_frame_name = "base_link"
		self.sensor_frame_name = "camera0_color_optical_frame"
		self.global_frame_name = "world"

		# transformation from camera to body frame
		self.T_bc = np.array([[0,0,1,0],
                              [-1,0,0,0],
                              [0,-1,0,0],
                              [0,0,0,1]]) 

		# self.T_bc = np.eye(4)

		self.rot_bc = tf.transformations.quaternion_from_matrix(self.T_bc)

		# rospy.sleep(0.5)

		if multi:
			cam0_sub = Subscriber("/camera0/color/image_raw", Image)
			cam1_sub = Subscriber("/camera1/color/image_raw", Image)
			cam2_sub = Subscriber("/camera2/color/image_raw", Image)
			cam3_sub = Subscriber("/camera3/color/image_raw", Image)
			self.ats = ApproximateTimeSynchronizer([cam0_sub, cam1_sub, cam2_sub, cam3_sub], queue_size=1, slop=0.05)
			self.ats.registerCallback(self.callback4)

		else:
			self.sub = rospy.Subscriber("/camera0/color/image_raw", Image, self.callback) # color
			
		self.pose_pub = rospy.Publisher("gt", PoseStamped, queue_size=1)
		self.tf_pub = tf.TransformBroadcaster()

		rospy.loginfo("GTNode::Ready!")



	def callback4(self, cam0_msg, cam1_msg, cam2_msg, cam3_msg):

		t_img = cam0_msg.header.stamp.to_nsec()
		row = self.df.iloc[(self.df['t']-t_img).abs().argsort()[:1]]
		t_row = row['t'].to_numpy()[0]
		x = row['gt_x'].to_numpy()[0]
		y = row['gt_y'].to_numpy()[0]
		z = camera_z
		yaw = row['gt_yaw'].to_numpy()[0] #+ np.pi

		quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
		pose_msg = PoseStamped()
		pose_msg.pose.orientation.x = quaternion[0]
		pose_msg.pose.orientation.y = quaternion[1]
		pose_msg.pose.orientation.z = quaternion[2]
		pose_msg.pose.orientation.w = quaternion[3]
		pose_msg.pose.position.x = x
		pose_msg.pose.position.y = y
		pose_msg.pose.position.z = z
		pose_msg.header.stamp = cam0_msg.header.stamp
		pose_msg.header.frame_id = "map" # map

		self.pose_pub.publish(pose_msg)
		#rospy.loginfo("GTNode Publish!")


	def callback(self, cam0_msg):

		t_img = cam0_msg.header.stamp.to_nsec()
		row = self.df.iloc[(self.df['t']-t_img).abs().argsort()[:1]]
		t_row = row['t'].to_numpy()[0]
		x = row['gt_x'].to_numpy()[0]
		y = row['gt_y'].to_numpy()[0]
		z = 0
		yaw = row['gt_yaw'].to_numpy()[0] #+ np.pi

		quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
		pose_msg = PoseStamped()
		pose_msg.pose.orientation.x = quaternion[0]
		pose_msg.pose.orientation.y = quaternion[1]
		pose_msg.pose.orientation.z = quaternion[2]
		pose_msg.pose.orientation.w = quaternion[3]
		pose_msg.pose.position.x = x
		pose_msg.pose.position.y = y
		pose_msg.pose.position.z = z
		pose_msg.header.stamp = cam0_msg.header.stamp
		pose_msg.header.frame_id = "map" # map

		# print(x, y)
		# print(cam0_msg.header.stamp)
		ns_in_s = 1000000000
		msg_latency_ns = int(0.0 * ns_in_s) # no latency # The problem is not here, no need to change
		self.pose_pub.publish(pose_msg)

		# t_record = t_row
		t_record = cam0_msg.header.stamp
		# print("timestamp:", t_record.secs, t_record.nsecs)

		if t_record.nsecs > msg_latency_ns:
			t_record.nsecs -= msg_latency_ns
		else:
			t_record.secs -= 1
			t_record.nsecs += (ns_in_s - msg_latency_ns)
		
		# t_record.secs += 0

		self.tf_pub.sendTransform(
                (x,y,z), quaternion,
                t_record, self.body_frame_name, self.global_frame_name) # tf is actually used
		# static transformation
		self.tf_pub.sendTransform(
                (self.T_bc[0, 3], self.T_bc[1, 3], self.T_bc[2, 3]+camera_z), self.rot_bc, # body to cam
                t_record, self.sensor_frame_name, self.body_frame_name)


def main():
    rospy.init_node('GTNode', anonymous=True)
    #rate = rospy.Rate(10)
    gt = GTNode()
    rospy.spin()

if __name__ == "__main__":

    main()