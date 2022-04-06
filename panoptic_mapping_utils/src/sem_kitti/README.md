# semantickitti2bag

"semantickitti2bag" contains helpful python programs to convert SemanticKITTI dataset to rosbag file.

 * Link to original [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
 * Link to [SemanticKITTI dataset](http://semantic-kitti.org/)

You can convert these dataset easy way :D

![rviz_2020_08_02](https://user-images.githubusercontent.com/60866340/89119958-13a2de80-d4ed-11ea-8ffc-29a5c5f5f420.png)

This repository is based on [tomas789's](https://github.com/tomas789) [kitti2bag](https://github.com/tomas789/kitti2bag) and [PRBonn's](https://github.com/PRBonn) [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api). These code are MIT Licence.

# Data organization

Data must be organized in following format.

![direct_tree](https://user-images.githubusercontent.com/60866340/89120963-a9426c00-d4f5-11ea-8cb7-a4e2aa83e8cd.png)


SemanticKITTI dataset has voxel data, but this repository doesn't handle.

 * image 0 and 1 is monocolor, and image 2 and 3 is colored.
 * velodyne contains the pointcloud for each scan.Each .bin is list of float32 points in [x, y, z, intensity] format.
 * .label file cantains a uint32 label for each point in the corresponding .bin scan. upper 16 bit contains instance id and lower 16 bit contains semantic label(ex.. car, bicycle, people). This program doesn't implement instance id.
 * poses.txt contain pose in world coordinate as homogenerous matrix. But this file must be loaded with calib.txt otherwise you can't get correct pose. This pose is estimated by some SLAM algorithm, for example SUMA and MULLS.
 * times.txt contains timestamps for each data(LiDAR, image, pose) scan
 * sequencenumber.txt(ex.. 00.txt) contains ground truth poses (GNSSINS), might be not very consistent but have a good global positioning accuracy.

# How to install it?
```bash
git clone https://github.com/amslabtech/semantikitti2bag
```

# How to run it

```bash
cd PLACE/OF/semantickitti2bag
python __main__.py -p /media/yuepan/BackupPlus/Data/kitti-dataset/ -s 00
```

 * -s 00 specify sequence number you must write number as 0 to  00, 1 to 01, 11 to 11.

# Publishing Topic
* Camera data      -> sensor_msgs/Image
* /odom_pose       -> nav_msgs/Odometry (Ground Truth, Please read Note)
* /velodyne_points -> sensor_msgs/PointCloud2
* /vehicle         -> nav_msgs/Pose (Ground Truth)
![rostopic](https://user-images.githubusercontent.com/60866340/107331157-d3ed2d00-6af5-11eb-968d-faffdebc06fa.png)

# TF-Tree

* Map          -> World Coordinate
* ground_truth_pose -> **This tree is not Ground Truth**. It is slam algorithm's prediction Ignore it.
* Vehicle      -> Vehicle frame(actually ground truth)
* velodyne     -> This coordinate is same as Vehicle coordinate, This is divided for programming convenience.
![Screenshot from 2021-02-09 16-56-30](https://user-images.githubusercontent.com/60866340/107333453-d8671500-6af8-11eb-8ea7-77b5dcd0cedb.png)

# Note

* This program generates odometry data, but actually odometry data is ***GROUND TRUTH DATA*** and its velocity's coordinate is ***World(map) coordinate***, bacause Semantic kitti dataset only contain GT poses. Velocity data is generated from transition from t-1 to t. If you have good idea to solve this problem, please post on issue without hesitation.

* This program was made by python2.7 on ROS melodic. I didn't tested on ROS noetic or Python3 environment. So if you run this on these environment, please notify how was that on issue :)

# Author

 * [Hibiki1020](https://github.com/Hibiki1020)

# Update
* (2021-6-30) In addition, we added a "label" class to the Point cloud field to represent the semantic class from here as well. Thanks for your help in fixing the code.

# License
This repository is under MIT License.
