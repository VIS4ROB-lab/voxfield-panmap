#ifndef PANOPTIC_MAPPING_ROS_TRANSFORMER_TRANSFORMER_H_
#define PANOPTIC_MAPPING_ROS_TRANSFORMER_TRANSFORMER_H_

#include <string>

#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include "panoptic_mapping/common/common.h"

namespace panoptic_mapping {

/**
 * Class that binds to either the TF tree or resolves transformations from the
 * ROS parameter server, depending on settings loaded from ROS params.
 */
// Same interface as Voxblox, original param setting handle used
class Transformer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transformer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);

  bool lookupTransform(const std::string& from_frame,
                       const std::string& to_frame, const ros::Time& timestamp,
                       Transformation* transform,
                       bool use_body_frame = false);
                      

  void transformCallback(const geometry_msgs::TransformStamped& transform_msg);

  void poseCallback(const geometry_msgs::PoseStamped& pose_msg);
    

 private:
  bool lookupTransformTf(const std::string& from_frame,
                         const std::string& to_frame,
                         const ros::Time& timestamp, Transformation* transform,
                         bool use_body_frame = false);

  bool lookupTransformQueue(const ros::Time& timestamp,
                            Transformation* transform);
  
  bool lookupPoseQueue(const ros::Time& timestamp,
                                  Transformation* transform);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  std::string world_frame_;
  /// If set, overwrite sensor frame with this value. If empty, unused.
  std::string sensor_frame_;

  // std::string robot_mesh_file_;
  /**
   * Whether to use TF transform resolution (true) or fixed transforms from
   * parameters and transform topics (false).
   */
  bool use_tf_transforms_;
  bool use_transform_msg_;
  int64_t timestamp_tolerance_ns_;
  /**
   * B is the body frame of the robot, C is the camera/sensor frame creating
   * the pointclouds, and D is the 'dynamic' frame; i.e., incoming messages
   * are assumed to be T_G_D.
   */
  Transformation T_B_C_;
  Transformation T_B_D_;
  /**
   * If we use topic transforms, we have 2 parts: a dynamic transform from a
   * topic and a static transform from parameters.
   * Static transform should be T_G_D (where D is whatever sensor the
   * dynamic coordinate frame is in) and the static should be T_D_C (where
   * C is the sensor frame that produces the depth data). It is possible to
   * specify T_C_D and set invert_static_transform to true.
   */

  /**
   * To be replaced (at least optionally) with odometry + static transform
   * from IMU to visual frame.
   */
  tf::TransformListener tf_listener_;

  // l Only used if use_tf_transforms_ set to false.
  ros::Subscriber transform_sub_;
  ros::Subscriber pose_sub_;

  tf2_ros::TransformBroadcaster tf_broadcaster_;

  ros::Publisher robot_mesh_pub_;
    
  // l Transform queue, used only when use_tf_transforms is false.
  AlignedDeque<geometry_msgs::TransformStamped> transform_queue_;

  AlignedDeque<geometry_msgs::PoseStamped> pose_queue_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_TRANSFORMER_TRANSFORMER_H_
