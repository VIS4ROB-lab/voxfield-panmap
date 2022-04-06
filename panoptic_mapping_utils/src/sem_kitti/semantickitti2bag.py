import sys
sys.dont_write_bytecode = True
import math
import utils #import utils.py
from numpy.linalg import inv
import tf
import tf2_ros
import os
import cv2
from cv_bridge import CvBridge
import rospy
import rosbag
import progressbar
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import argparse
import glob


class SemanticKitti_Raw:
    """Load and parse raw data into a usable format"""

    def __init__(self, dataset_path, sequence_number, scanlabel_bool, **kwargs):
        self.data_path = os.path.join(dataset_path, 'dataset', 'sequences', sequence_number)

        self.frames = kwargs.get('frames', None)

        self.imtype = kwargs.get('imtype', 'png')

        self._get_file_lists(scanlabel_bool)
        #self._load_calib()
        
        self._load_timestamps()

    def _get_file_lists(self, scanlabel_bool):

        # not used
        # self.cam0_files = sorted(glob.glob(
        #     os.path.join(self.data_path, 'image_0', '*.{}'.format(self.imtype))))

        # self.cam1_files = sorted(glob.glob(
        #     os.path.join(self.data_path, 'image_1', '*.{}'.format(self.imtype))))
        
        # self.cam2_files = sorted(glob.glob(
        #     os.path.join(self.data_path, 'image_2', '*.{}'.format(self.imtype))))       
        
        # self.cam3_files = sorted(glob.glob(
        #     os.path.join(self.data_path, 'image_3', '*.{}'.format(self.imtype))))

        self.velo_files = sorted(glob.glob(
            os.path.join(self.data_path, 'velodyne', '*.bin')))

        if scanlabel_bool == 1:
            self.label_files = sorted(glob.glob(
                os.path.join(self.data_path, 'labels', '*.label')))
        #print(self.cam1_files)
        #print(self.velo_files)

        # if self.frames is not None:

    def _load_timestamps(self):
        timestamp_file = os.path.join(
                self.data_path, 'times.txt')

        self.timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                #number = datetime.fromtimestamp(float(line))
                number = float(line)
                if number == 0.0:
                    number = 0.0001
                #sign = 1.0
                
                #if line[9]=='+':
                #    sign = 1.0
                #else:
                #    sign = -1.0

                #num = float(line[10])*10 + float(line[11])*1

                #time_t = number*(10**(sign*num))
                #print(line)
                #print(type(line))
                #print(number)
                #print(type(number))
                self.timestamps.append(number)

def inv_t(transform):

    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1*R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv

    return transform_inv

def save_velo_data_with_label(bag, kitti, velo_frame_id, velo_topic, bridge):
    print("Exporting Velodyne and Label data")
    
    velo_data_dir = os.path.join(kitti.data_path, 'velodyne')
    velo_filenames = sorted(os.listdir(velo_data_dir))

    label_data_dir = os.path.join(kitti.data_path, 'labels')
    label_filenames = sorted(os.listdir(label_data_dir))

    datatimes = kitti.timestamps

    iterable = zip(datatimes, velo_filenames, label_filenames)
    bar = progressbar.ProgressBar()

    for dt, veloname, labelname in bar(iterable):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, veloname)
        label_filename = os.path.join(label_data_dir, labelname)

        veloscan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        labelscan = (np.fromfile(label_filename, dtype=np.int32)).reshape(-1,1)
        
        labeldata = utils.LabelDataConverter(labelscan)

        #normaldatafine = estimate_normal(veloscan)
        
        scan_label = []
        scan = []

        for t in range(len(labeldata.rgb_id)):
            # point_label = [veloscan[t][0], veloscan[t][1], veloscan[t][2], veloscan[t][3], labeldata.rgb_id[t], labeldata.semantic_id[t], labeldata.instance_id[t]]
            point_label = [veloscan[t][0], veloscan[t][1], veloscan[t][2], veloscan[t][3], labeldata.rgb_id[t], labelscan[t]]
            
            scan_label.append(point_label)

        # not needed
        # proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(veloscan, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=1800, max_range=50)

        # normal_data = (255 * gen_normal_map(proj_range, proj_vertex)).astype(np.uint8)

        # proj_range: projected range image with depth, each pixel contains the corresponding depth
        # proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        # proj_intensity: each pixel contains the corresponding intensity
        # proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
        # 900 instead of 1800
        # write the message of these range images 

        # range_img = cv2.imread(proj_range)
        # encoding = "bgr8"
        # base_topic = "/lidar_images"

        # encoding = "32FC1"
        # range_img_msg = bridge.cv2_to_imgmsg(proj_range, encoding=encoding)
        # range_img_msg.header.frame_id = velo_frame_id
        # range_img_msg.header.stamp = rospy.Time.from_sec(float(dt))
        # range_img_topic = "/range_image"
        # bag.write(base_topic + range_img_topic, range_img_msg, t=range_img_msg.header.stamp)

        # encoding = "32FC1"
        # intensity_img_msg = bridge.cv2_to_imgmsg(proj_intensity, encoding=encoding)
        # intensity_img_msg.header.frame_id = velo_frame_id
        # intensity_img_msg.header.stamp = rospy.Time.from_sec(float(dt))
        # intensity_img_topic = "/intensity_image"
        # bag.write(base_topic + intensity_img_topic, intensity_img_msg, t=intensity_img_msg.header.stamp)

        # # normal image has some problem for visualization
        # encoding = "rgb8"
        # normal_img_msg = bridge.cv2_to_imgmsg(normal_data, encoding=encoding)
        # normal_img_msg.header.frame_id = velo_frame_id
        # normal_img_msg.header.stamp = rospy.Time.from_sec(float(dt))
        # normal_img_topic = "/normal_image"
        # bag.write(base_topic + normal_img_topic, normal_img_msg, t=normal_img_msg.header.stamp)

        # # # read semantic probabilities from the raw file (prob of label 0-19)
        # # probs = np.fromfile(prob_paths, dtype=np.float32).reshape((-1, 20)) 
        # # # fill in a semantic image
        # # sem_color = proj_idx
        # sem_color = np.full((64, 1800, 3), 0, dtype=np.uint8) # 3-dim
        # # # sem_img[proj_idx >=0] = labeldata.rgb_id[proj_idx]
        # # sem_color[proj_idx >= 0] = labeldata.rgb_id[proj_idx[proj_idx >= 0]]
        # for i in range(np.shape(proj_idx)[0]):
        #     for j in range(np.shape(proj_idx)[1]):
        #         if proj_idx[i,j] >= 0:
        #             sem_color[i,j] = labeldata.rgb_arr_id[proj_idx[i,j]]
        
        # encoding = "rgb8"
        # sem_img_msg = bridge.cv2_to_imgmsg(sem_color, encoding=encoding)
        # sem_img_msg.header.frame_id = velo_frame_id
        # sem_img_msg.header.stamp = rospy.Time.from_sec(float(dt))
        # sem_img_topic = "/semantic_image"
        # bag.write(base_topic + sem_img_topic, sem_img_msg, t=sem_img_msg.header.stamp)

        header = Header()
        header.frame_id = velo_frame_id
        header.stamp = rospy.Time.from_sec(float(dt))

        fields =[PointField('x',  0, PointField.FLOAT32, 1),
                 PointField('y',  4, PointField.FLOAT32, 1),
                 PointField('z',  8, PointField.FLOAT32, 1),
                 PointField('intensity', 12, PointField.FLOAT32, 1),
                 PointField('rgb', 16, PointField.UINT32, 1),
                 PointField('label', 20, PointField.UINT32, 1)] 
                # lower16: instance, higher16: semantic
                # PointField('label', 20, PointField.UINT16, 1),
                # PointField('instance', 22, PointField.UINT16, 1)]

        pcl_msg = pcl2.create_cloud(header, fields, scan_label)
        bag.write(velo_topic, pcl_msg, t=pcl_msg.header.stamp)

        

def save_velo_data(bag, kitti, velo_frame_id, velo_topic):
    print("Exporting Velodyne data")
    
    velo_data_dir = os.path.join(kitti.data_path, 'velodyne')
    velo_filenames = sorted(os.listdir(velo_data_dir))

    datatimes = kitti.timestamps

    iterable = zip(datatimes, velo_filenames)
    bar = progressbar.ProgressBar()

    for dt, veloname in bar(iterable):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, veloname)

        veloscan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)

        header = Header()
        header.frame_id = velo_frame_id
        header.stamp = rospy.Time.from_sec(float(dt))

        fields =[PointField('x',  0, PointField.FLOAT32, 1),
                 PointField('y',  4, PointField.FLOAT32, 1),
                 PointField('z',  8, PointField.FLOAT32, 1),
                 PointField('intensity', 12, PointField.FLOAT32, 1)]

        pcl_msg = pcl2.create_cloud(header, fields, veloscan)
        bag.write(velo_topic, pcl_msg, t=pcl_msg.header.stamp)

def read_calib_file(filename):
    """ read calibration file 

        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    """calib1 = np.eye(4,4)
    calib1[0:3, 3] = [0.27, 0.0, -0.08]
    print(calib1)
    calib.append(calib1)

    calib2 = np.eye(4,4)
    calib2[0:3, 3] = [0.27, -0.51, -0.08]
    print(calib2)
    calib.append(calib2)

    calib3 = np.eye(4,4)
    calib3[0:3, 3] = [0.27, 0.06, -0.08]
    print(calib3)
    calib.append(calib3)

    calib4 = np.eye(4,4)
    calib4[0:3, 3] = [0.27, -0.45, -0.08]
    print(calib4)
    calib.append(calib4)"""
    calib_file = open(filename)

    key_num = 0

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4,4))
        
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    
    #print(calib)
    return calib

def read_poses_file(filename, calibration):
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) # lidar in world frame
        #poses.append(pose)

    pose_file.close()
    return poses

def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = tf.transformations.quaternion_from_matrix(transform) #Create quaternion from 4*4 homogenerous transformation matrix
    
    q_n = q / np.linalg.norm(q)

    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id #master
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = t[0]
    tf_msg.transform.translation.y = t[1]
    tf_msg.transform.translation.z = t[2]
    tf_msg.transform.rotation.x = q_n[0]
    tf_msg.transform.rotation.y = q_n[1]
    tf_msg.transform.rotation.z = q_n[2]
    tf_msg.transform.rotation.w = q_n[3]

    return tf_msg

def save_static_transforms(bag, transforms, kitti):
    print("Get static transform")
    tfm = TFMessage()
    datatimes = kitti.timestamps

    for transform in transforms:
        at = get_static_transform(transform[0], transform[1], transform[2])
        #print(at)
        tfm.transforms.append(at)

    for dt in datatimes:
        #time = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
        time = rospy.Time.from_sec(float(dt))
        #print(dt)
        #print(type(time))
        for i in range(len(tfm.transforms)):
            tfm.transforms[i].header.stamp = time
        bag.write('/tf_static', tfm, t=time)

def save_dynamic_transforms(bag, kitti, poses, master_frame_id, slave_frame_id, initial_time):
    print("Exporting time dependent transformations")

    datatimes = kitti.timestamps

    iterable = zip(datatimes, poses)
    bar = progressbar.ProgressBar()

    for dt, pose in bar(iterable):
        tf_dy_msg = TFMessage()
        tf_dy_transform = TransformStamped()
        
        #tf_dy_transform.header.stamp = rospy.Time.from_sec(float(dt.strftime("%s.%f")))
        tf_dy_transform.header.stamp = rospy.Time.from_sec(float(dt))
        #print(tf_dy_transform.header.stamp)

        tf_dy_transform.header.frame_id = master_frame_id
        tf_dy_transform.child_frame_id = slave_frame_id

        t = pose[0:3, 3]
        q = tf.transformations.quaternion_from_matrix(pose)

        dy_tf = Transform()

        dy_tf.translation.x = t[0]
        dy_tf.translation.y = t[1]
        dy_tf.translation.z = t[2]

        q_n = q / np.linalg.norm(q)

        dy_tf.rotation.x = q_n[0]
        dy_tf.rotation.y = q_n[1]
        dy_tf.rotation.z = q_n[2]
        dy_tf.rotation.w = q_n[3]

        tf_dy_transform.transform = dy_tf
        tf_dy_msg.transforms.append(tf_dy_transform)

        bag.write('/tf', tf_dy_msg, t=tf_dy_msg.transforms[0].header.stamp)

def save_camera_data(bag, kitti, calibration, bridge, camera, camera_frame_id, topic, initial_time):
    print("Exporting {} image data".format(topic))
    datatimes = kitti.timestamps

    image_file_dir = os.path.join(kitti.data_path, 'image_{}'.format(camera))
    image_file_names = sorted(os.listdir(image_file_dir))

    calib = CameraInfo()
    calib.header.frame_id = camera_frame_id
    #P = calibration["{}".format(camera)]
    #calib.P


    iterable = zip(datatimes, image_file_names)
    bar = progressbar.ProgressBar()

    for dt, filename in bar(iterable):
        image_filename = os.path.join(image_file_dir, filename)
        cv_image = cv2.imread(image_filename)
        #calib.height, calib.width = cv_image.shape[ :2]

        if camera in (0, 1):
            #image_0 and image_1 contain monocolor image, but these images are represented as RGB color
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        encoding = "mono8" if camera in (0, 1) else "bgr8"
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
        image_message.header.frame_id = camera_frame_id
        image_message.header.stamp = rospy.Time.from_sec(float(dt))
        topic_ext = "/image_raw"

        #calib.header.stamp = image_message.header.stamp

        bag.write(topic + topic_ext, image_message, t=image_message.header.stamp)
        #bag.write(topic + '/camera_info', calib, t=calib.header.stamp)


def save_pose_msg(bag, kitti, poses, master_frame_id, slave_frame_id, topic, initial_time=None):
    print("Exporting pose msg")

    datatimes = kitti.timestamps

    iterable = zip(datatimes, poses)
    bar = progressbar.ProgressBar()

    p_t1 = PoseStamped()
    dt_1 = 0.00
    counter = 0

    for dt, pose in bar(iterable):
        p = PoseStamped()
        p.header.frame_id = master_frame_id
        p.header.stamp = rospy.Time.from_sec(float(dt))

        t = pose[0:3, 3]
        q = tf.transformations.quaternion_from_matrix(pose)

        p.pose.position.x = t[0]
        p.pose.position.y = t[1]
        p.pose.position.z = t[2]

        q_n = q / np.linalg.norm(q)

        p.pose.orientation.x = q_n[0]
        p.pose.orientation.y = q_n[1]
        p.pose.orientation.z = q_n[2]
        p.pose.orientation.w = q_n[3]

        if(counter == 0):
            p_t1 = p

        bag.write(topic, p, t=p.header.stamp)

        delta_t = (dt - dt_1)
        if(counter == 0):
            delta_t = 0.00000001
        
        vx = (p.pose.position.x - p_t1.pose.position.x )/delta_t
        vy = (p.pose.position.y - p_t1.pose.position.y )/delta_t
        vz = (p.pose.position.z - p_t1.pose.position.z )/delta_t

        vqx = (p.pose.orientation.x - p_t1.pose.orientation.x)
        vqy = (p.pose.orientation.y - p_t1.pose.orientation.y)
        vqz = (p.pose.orientation.z - p_t1.pose.orientation.z)
        vqw = (p.pose.orientation.w - p_t1.pose.orientation.w)
  
        v_roll = math.atan2( 2*(vqw*vqx + vqy*vqz), 1-2*(vqx**2 + vqy**2)  )/delta_t
        v_pitch = math.asin( 2*(vqw*vqy - vqz*vqx) )/delta_t
        v_yaw = math.atan2( 2*(vqw*vqz + vqx*vqy) , 1-2*(vqy**2 + vqz**2)  )/delta_t

        odom = Odometry()
        odom.header.stamp = p.header.stamp
        odom.header.frame_id = master_frame_id
        odom.child_frame_id = slave_frame_id

        odom.pose.pose.position = p.pose.position
        odom.pose.pose.orientation = p.pose.orientation
        
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        
        
        odom.twist.twist.angular.x = v_roll
        odom.twist.twist.angular.y = v_pitch
        odom.twist.twist.angular.z = v_yaw

        bag.write('/odom_pose', odom, t=odom.header.stamp)
        
        counter += 1
        p_t1 = p
        dt_1 = dt

def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=1800, max_range=50):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_range, proj_vertex, proj_intensity, proj_idx


def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=1800):
  """ Generate a normal image given the range projection of a point cloud.(from the range image)
      Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
      Returns: 
        normal_data: each pixel contains the corresponding normal
  """
  normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32) # 3-dim
  
  # iterate over all pixels in the range image
  for x in range(proj_W):
    for y in range(proj_H - 1):
      p = current_vertex[y, x][:3]
      depth = current_range[y, x]
      
      if depth > 0:
        wrap_x = wrap(x + 1, proj_W) # cylinder on the x axis
        u = current_vertex[y, wrap_x][:3]
        u_depth = current_range[y, wrap_x]
        if u_depth <= 0:
          continue
        
        v = current_vertex[y + 1, x][:3]
        v_depth = current_range[y + 1, x]
        if v_depth <= 0:
          continue
        
        u_norm = (u - p) / np.linalg.norm(u - p)
        v_norm = (v - p) / np.linalg.norm(v - p)
        
        w = np.cross(v_norm, u_norm)
        norm = np.linalg.norm(w)
        if norm > 0:
          normal = w / norm
          normal_data[y, x] = normal
  
  return normal_data


def wrap(x, dim):
  """ Wrap the boarder of the range image.
  """
  value = x
  if value >= dim:
    value = (value - dim)
  if value < 0:
    value = (value + dim)
  return value

# def hex_to_rgb(hex):
#     hex = hex.lstrip('0x')
#     hex_1 = '0x' + hex[0:1]
#     # print(hex_1)
#     hex_2 = '0x' + hex[2:3]
#     hex_3 = '0x' + hex[4:5]
#     num_1 = int(hex_1,16) 
#     num_2 = int(hex_2,16) 
#     num_3 = int(hex_3,16) 
#     rgb = np.array([num_1, num_2, num_3], dtype=np.uint8)
#     return rgb
#     # num = int(hex,16) 

#     # hlen = len(hex)
#     # print(hlen)
#     # rgb = tuple(int(hex[i:i+hlen/3], 16) for i in range(0, hlen, hlen/3))
#     # print(rgb)
#     # return np.asarray(rgb).astype(np.uint8)

def run_semantickitti2bag():

    parser = argparse.ArgumentParser(description='Convert SemanticKITTI dataset to rosbag file')


    parser.add_argument("-p","--dataset_path", help='Path to Semantickitti file')
    parser.add_argument("-s","--sequence_number", help='Sequence number, must be written as 1 to 01')
    args = parser.parse_args()

    bridge = CvBridge()
    compression = rosbag.Compression.NONE

    #camera

    cameras = [
            (0, 'camera_gray_left', '/semantickitti/camera_gray_left'),
            (1, 'camera_gray_right', '/semantickitti/camera_gray_right'),
            (2, 'camera_color_left', '/semantickitti/camera_color_left'),
            (3, 'camera_color_right', '/semantickitti/camera_color_right')
        ]
    
    if args.dataset_path == None:
        print("Dataset path is not given.")
        sys.exit(1)
    elif args.sequence_number == None:
        print("Sequence number is not given.")
        sys.exit(1)

    scanlabel_bool = 1
    # if int(args.sequence_number) > 10:
    #     scanlabel_bool = 0

    #bag = rosbag.Bag("semanticusl_sequence{}.bag".format(args.sequence_number), 'w', compression=compression)    
    bag = rosbag.Bag("semantickitti_sequence{}.bag".format(args.sequence_number), 'w', compression=compression)

    kitti = SemanticKitti_Raw(args.dataset_path, args.sequence_number, scanlabel_bool)

    if not os.path.exists(kitti.data_path):
        print('Path {} does not exists. Force-quiting....'.format(kitti.data_path))
        sys.exit(1)

    if len(kitti.timestamps) == 0:
        print('Dataset is empty? Check your semantickitti dataset file')
        sys.exit(1)
    
    try:
        world_frame_id = 'world' # too keep it as the same as voxblox

        vehicle_frame_id = 'vehicle'
        vehicle_topic = '/vehicle'

        ground_truth_frame_id = 'ground_truth_pose'
        ground_truth_topic = '/ground_truth_pose'

        velo_frame_id = 'velodyne'
        velo_topic = '/velodyne_points'

        vehicle_frame_id = vehicle_frame_id

        T_base_link_to_velo = np.eye(4, 4)

        calibration = read_calib_file(os.path.join(kitti.data_path, 'calib.txt'))
        
        calib0 = np.eye(4,4)
        calib0[0:3, 3] = [0.27, 0.0, -0.08]
        #print(calib0)
        
        calib1 = np.eye(4,4)
        calib1[0:3, 3] = [0.27, -0.51, -0.08]
        #print(calib1)
        
        calib2 = np.eye(4,4)
        calib2[0:3, 3] = [0.27, 0.06, -0.08]
        #print(calib2)
        
        calib3 = np.eye(4,4)
        calib3[0:3, 3] = [0.27, -0.45, -0.08]
        #print(calib3)
        
        #tf-static

        transforms = [
            (vehicle_frame_id, velo_frame_id, T_base_link_to_velo),
            (vehicle_frame_id, cameras[0][1], calib0),
            (vehicle_frame_id, cameras[1][1], calib1),
            (vehicle_frame_id, cameras[2][1], calib2),
            (vehicle_frame_id, cameras[3][1], calib3)
        ]


        save_static_transforms(bag, transforms, kitti)

        # These poses are represented in world coordinate (provided by semantic KITTI, predicted by SUMA)
        poses = read_poses_file(os.path.join(kitti.data_path,'poses.txt'), calibration)
        
        # the ground truth file is named as the number of the sequence, provided by KITTI officially, but may have some problem
        ground_truth_file_name = "{}.txt".format(args.sequence_number) 
        ground_truth = read_poses_file(os.path.join(kitti.data_path, ground_truth_file_name), calibration)

        #save_dynamic_transforms(bag, kitti, poses, world_frame_id, vehicle_frame_id, initial_time=None)
        save_dynamic_transforms(bag, kitti, ground_truth, world_frame_id, ground_truth_frame_id, initial_time=None)

        #save_pose_msg(bag, kitti, poses, world_frame_id, vehicle_frame_id, vehicle_topic, initial_time=None)
        save_pose_msg(bag, kitti, ground_truth, world_frame_id, ground_truth_frame_id, ground_truth_topic, initial_time=None)

        
        if scanlabel_bool == 1:
            #print('a')
            save_velo_data_with_label(bag, kitti, velo_frame_id, velo_topic, bridge)
            #save_velo_data(bag, kitti, velo_frame_id, velo_topic)

        elif scanlabel_bool == 0:
            #print('b')
            save_velo_data(bag, kitti, velo_frame_id, velo_topic)

        # not needed
        # for camera in cameras:
        #     #print('c')
        #     save_camera_data(bag, kitti, calibration, bridge, camera=camera[0], camera_frame_id=camera[1], topic=camera[2], initial_time=None)


    finally:
        print('Convertion is done')
        print(bag)
        bag.close()
