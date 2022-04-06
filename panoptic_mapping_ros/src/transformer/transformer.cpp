#include "panoptic_mapping_ros/transformer/transformer.h"
#include <visualization_msgs/Marker.h>
#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>
#include <minkindr_conversions/kindr_xml.h>

namespace panoptic_mapping {

Transformer::Transformer(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      world_frame_("world"),
      sensor_frame_(""),
      use_tf_transforms_(true),
      use_transform_msg_(true),
      timestamp_tolerance_ns_(1000000) {
  nh_private_.param("world_frame", world_frame_, world_frame_);
  nh_private_.param("sensor_frame", sensor_frame_, sensor_frame_);
  nh_private_.param("use_transform_msg", use_transform_msg_, use_transform_msg_);
  // nh_private_.param("robot_mesh_file", robot_mesh_file_, robot_mesh_file_);

  const double kNanoSecondsInSecond = 1.0e9;
  double timestamp_tolerance_sec =
      timestamp_tolerance_ns_ / kNanoSecondsInSecond;
  nh_private_.param("timestamp_tolerance_sec", timestamp_tolerance_sec,
                    timestamp_tolerance_sec);
  timestamp_tolerance_ns_ =
      static_cast<int64_t>(timestamp_tolerance_sec * kNanoSecondsInSecond);

  // Transform settings.
  nh_private_.param("use_tf_transforms", use_tf_transforms_,
                    use_tf_transforms_);
  
  // If we use topic transforms, we have 2 parts: a dynamic transform from a
  // topic and a static transform from parameters (calibration).
  // Dynamic transform should be T_G_D (where D is whatever sensor the
  // dynamic coordinate frame is in) and the static should be T_D_C (where
  // C is the sensor frame that produces the depth data). It is possible to
  // specify T_C_D and set invert_static_tranform to true.

  // Retrieve T_D_C from params.
  XmlRpc::XmlRpcValue T_B_D_xml;
  // TODO(helenol): split out into a function to avoid duplication.
  if (nh_private_.getParam("T_B_D", T_B_D_xml)) {
    kindr::minimal::xmlRpcToKindr(T_B_D_xml, &T_B_D_);

    // See if we need to invert it.
    bool invert_static_tranform = false;
    nh_private_.param("invert_T_B_D", invert_static_tranform,
                      invert_static_tranform);
    if (invert_static_tranform) {
      T_B_D_ = T_B_D_.inverse();
    }
    // LOG(INFO) << "T_B_D:\n" << T_B_D_;
  }
  XmlRpc::XmlRpcValue T_B_C_xml;
  if (nh_private_.getParam("T_B_C", T_B_C_xml)) {
    kindr::minimal::xmlRpcToKindr(T_B_C_xml, &T_B_C_);

    // See if we need to invert it.
    bool invert_static_tranform = false;
    nh_private_.param("invert_T_B_C", invert_static_tranform,
                        invert_static_tranform);
    if (invert_static_tranform) {
      T_B_C_ = T_B_C_.inverse();
    }
      // LOG(INFO) << "T_B_C:\n" << T_B_C_;
  }

  if (!use_tf_transforms_) {
    if(use_transform_msg_) {
      transform_sub_ =
          nh_.subscribe("transform", 40, &Transformer::transformCallback, this);
    } else {  
      pose_sub_ =
          nh_.subscribe("pose", 40, &Transformer::poseCallback, this);
    }
    // robot_mesh_pub_ = 
    //   nh_.advertise<visualization_msgs::Marker>("Robot_mesh", 100);
  }
  //Or we will use tf_transform, we do not need the calibration parameters
  //lookupTransformTf
  // Retrieve T_D_C from params.
}

void Transformer::transformCallback(
    const geometry_msgs::TransformStamped& transform_msg) {
  transform_queue_.push_back(transform_msg);
}

void Transformer::poseCallback(
    const geometry_msgs::PoseStamped& pose_msg) {
  pose_queue_.push_back(pose_msg);
}

bool Transformer::lookupTransform(const std::string& from_frame,
                                  const std::string& to_frame,
                                  const ros::Time& timestamp,
                                  Transformation* transform,
                                  bool use_body_frame) {
  CHECK_NOTNULL(transform);
  if (use_tf_transforms_) {
    return lookupTransformTf(from_frame, to_frame, timestamp, transform, use_body_frame);
  } else {
    if (use_transform_msg_)
      return lookupTransformQueue(timestamp, transform);
    else // use pose msg
      return lookupPoseQueue(timestamp, transform);
  }
}

// Stolen from octomap_manager
bool Transformer::lookupTransformTf(const std::string& from_frame,
                                    const std::string& to_frame,
                                    const ros::Time& timestamp,
                                    Transformation* transform,
                                    bool use_body_frame) {
  CHECK_NOTNULL(transform);
  tf::StampedTransform tf_transform;
  ros::Time time_to_lookup = timestamp;

  // Allow overwriting the TF frame for the sensor.
  std::string from_frame_modified = from_frame;
  if (!sensor_frame_.empty()) {
    from_frame_modified = sensor_frame_;
  }

  // Previous behavior was just to use the latest transform if the time is in
  // the future. Now we will just wait.
  // if (!tf_listener_.canTransform(to_frame, from_frame_modified,
  //                                time_to_lookup)) {
  //   return false;
  // }

  try {
    tf_listener_.lookupTransform(to_frame, from_frame_modified, time_to_lookup,
                                 tf_transform);
  } catch (tf::TransformException& ex) {  // NOLINT
    ROS_ERROR_STREAM(
        "Error getting TF transform from sensor data: " << ex.what());
    return false;
  }

  tf::transformTFToKindr(tf_transform, transform);

  if (use_body_frame){
    *transform = (*transform) * T_B_C_.inverse(); //T_wb = T_wc * T_cb
  }

  //Point translation = transform->getPosition();
  //LOG(INFO) << "drone position: (" << translation(0) << "," << translation(1) << "," << translation(2) << ")";
  return true;
}

bool Transformer::lookupTransformQueue(const ros::Time& timestamp,
                                       Transformation* transform) {
  CHECK_NOTNULL(transform);
  if (transform_queue_.empty()) {
    ROS_WARN_STREAM_THROTTLE(30, "No match found for transform timestamp: "
                                     << timestamp
                                     << " as transform queue is empty.");
    return false;
  }
  // Try to match the transforms in the queue.
  bool match_found = false;
  std::deque<geometry_msgs::TransformStamped>::iterator it =
      transform_queue_.begin();
  for (; it != transform_queue_.end(); ++it) {
    // If the current transform is newer than the requested timestamp, we need
    // to break.
    if (it->header.stamp > timestamp) {
      if ((it->header.stamp - timestamp).toNSec() < timestamp_tolerance_ns_) {
        match_found = true;
      }
      break;
    }

    if ((timestamp - it->header.stamp).toNSec() < timestamp_tolerance_ns_) {
      match_found = true;
      break;
    }
  }

  // Match found basically means an exact match.
  Transformation T_G_D;
  if (match_found) {
    tf::transformMsgToKindr(it->transform, &T_G_D);
  } else {
    // If we think we have an inexact match, have to check that we're still
    // within bounds and interpolate.
    if (it == transform_queue_.begin() || it == transform_queue_.end()) {
      ROS_WARN_STREAM_THROTTLE(
          30, "No match found for transform timestamp: "
                  << timestamp
                  << " Queue front: " << transform_queue_.front().header.stamp
                  << " back: " << transform_queue_.back().header.stamp);
      return false;
    }
    // Newest should be 1 past the requested timestamp, oldest should be one
    // before the requested timestamp.
    Transformation T_G_D_newest;
    tf::transformMsgToKindr(it->transform, &T_G_D_newest);
    int64_t offset_newest_ns = (it->header.stamp - timestamp).toNSec();
    // We already checked that this is not the beginning.
    it--;
    Transformation T_G_D_oldest;
    tf::transformMsgToKindr(it->transform, &T_G_D_oldest);
    int64_t offset_oldest_ns = (timestamp - it->header.stamp).toNSec();

    // Interpolate between the two transformations using the exponential map.
    FloatingPoint t_diff_ratio =
        static_cast<FloatingPoint>(offset_oldest_ns) /
        static_cast<FloatingPoint>(offset_newest_ns + offset_oldest_ns);

    Transformation::Vector6 diff_vector =
        (T_G_D_oldest.inverse() * T_G_D_newest).log();
    T_G_D = T_G_D_oldest * Transformation::exp(t_diff_ratio * diff_vector);
  }

  // If we have a static transform, apply it too.
  // Transform should actually be T_G_C. So need to take it through the full
  // chain.
  *transform = T_G_D * T_B_D_.inverse() * T_B_C_; // gddbbc = gc // sensor to world
  Transformation T_G_C = *transform;

  // LOG(INFO) << "Current sensor pose:\n" << *transform; // sensor to world

  // And also clear the queue up to this point. This leaves the current
  // message in place.
  transform_queue_.erase(transform_queue_.begin(), it);

  // publish the sensor's transformation (tf tree)
  geometry_msgs::TransformStamped msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = world_frame_;
  msg.child_frame_id = sensor_frame_;
  tf::transformKindrToMsg(T_G_C.cast<double>(), &msg.transform);
  tf_broadcaster_.sendTransform(msg);

  return true;
}

bool Transformer::lookupPoseQueue(const ros::Time& timestamp,
                                  Transformation* transform) {
  // CHECK_NOTNULL(transform);
  // if (pose_queue_.empty()) {
  //   ROS_WARN_STREAM_THROTTLE(30, "No match found for transform timestamp: "
  //                                    << timestamp
  //                                    << " as transform queue is empty.");
  //   return false;
  // }
  // // Try to match the transforms in the queue.
  // bool match_found = false;
  // std::deque<geometry_msgs::PoseStamped>::iterator it =
  //     pose_queue_.begin();
  // for (; it != pose_queue_.end(); ++it) {
  //   // If the current transform is newer than the requested timestamp, we need
  //   // to break.
  //   if (it->header.stamp > timestamp) {
  //     if ((it->header.stamp - timestamp).toNSec() < timestamp_tolerance_ns_) {
  //       match_found = true;
  //     }
  //     break;
  //   }

  //   if ((timestamp - it->header.stamp).toNSec() < timestamp_tolerance_ns_) {
  //     match_found = true;
  //     break;
  //   }
  // }

  // // Match found basically means an exact match.
  // Transformation T_G_D;
  // if (match_found) {
  //   tf::poseMsgToKindr(it->pose, &T_G_D);
  // } else {
  //   // If we think we have an inexact match, have to check that we're still
  //   // within bounds and interpolate.
  //   if (it == pose_queue_.begin() || it == pose_queue_.end()) {
  //     ROS_WARN_STREAM_THROTTLE(
  //         30, "No match found for transform timestamp: "
  //                 << timestamp
  //                 << " Queue front: " << pose_queue_.front().header.stamp
  //                 << " back: " << pose_queue_.back().header.stamp);
  //     return false;
  //   }
  //   // Newest should be 1 past the requested timestamp, oldest should be one
  //   // before the requested timestamp.
  //   Transformation T_G_D_newest;
  //   tf::poseMsgToKindr(it->pose, &T_G_D_newest);
  //   int64_t offset_newest_ns = (it->header.stamp - timestamp).toNSec();
  //   // We already checked that this is not the beginning.
  //   it--;
  //   Transformation T_G_D_oldest;
  //   tf::poseMsgToKindr(it->pose, &T_G_D_oldest);
  //   int64_t offset_oldest_ns = (timestamp - it->header.stamp).toNSec();

  //   // Interpolate between the two transformations using the exponential map.
  //   FloatingPoint t_diff_ratio =
  //       static_cast<FloatingPoint>(offset_oldest_ns) /
  //       static_cast<FloatingPoint>(offset_newest_ns + offset_oldest_ns);

  //   Transformation::Vector6 diff_vector =
  //       (T_G_D_oldest.inverse() * T_G_D_newest).log();
  //   T_G_D = T_G_D_oldest * Transformation::exp(t_diff_ratio * diff_vector);
  // }

  // // If we have a static transform, apply it too.
  // // Transform should actually be T_G_C. So need to take it through the full
  // // chain.
  // *transform = T_G_D * T_B_D_.inverse() * T_B_C_; // gddbbc = gc // sensor to world
  // Transformation T_G_C = *transform;

  // // LOG(INFO) << "Current sensor pose:\n" << *transform; // sensor to world

  // // And also clear the queue up to this point. This leaves the current
  // // message in place.
  // pose_queue_.erase(pose_queue_.begin(), it);

  // // publish the sensor's transformation (tf tree)
  // geometry_msgs::TransformStamped msg;
  // msg.header.stamp = ros::Time::now();
  // msg.header.frame_id = world_frame_;
  // msg.child_frame_id = sensor_frame_;
  // tf::transformKindrToMsg(T_G_C.cast<double>(), &msg.transform);
  // tf_broadcaster_.sendTransform(msg);

  return true;
}


}  // namespace panoptic_mapping
