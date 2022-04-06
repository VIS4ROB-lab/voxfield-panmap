#ifndef PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_
#define PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core/mat.hpp>

#include "panoptic_mapping/common/common.h"

namespace panoptic_mapping {
class InputDataUser;

/**
 * All custom types of input data are defined here.
 */

// Labels supplied by the detectron2 network.
struct DetectronLabel {
  int id = 0;
  bool is_thing = true;
  int category_id = 0;
  int instance_id = 0;
  float score = 0.f;
};
typedef std::unordered_map<int, DetectronLabel> DetectronLabels;  // <id-label>

/**
 * A class that wraps all input data to be processed into a common structure.
 * Optional fields are also included here but the data is only set if required.
 */
class InputData {
 public:
  // Lists all types of possible inputs.
  // If you want to directly use point cloud, then you need some preprocessing
  enum class InputType {
    kDepthImage,
    kColorImage,
    kSegmentationImage, //panoptic segmentation
    kDetectronLabels,
    kVertexMap,         //point cloud in cv::Mat
    kValidityImage,
    kUncertaintyImage,
    kNormalImage,     
    kPointCloud,        //point cloud
    kPointColor         //point-wise color of the point cloud
  };

  static std::string inputTypeToString(InputType type) {
    switch (type) {
      case InputType::kDepthImage:
        return "Depth Image";
      case InputType::kColorImage:
        return "Color Image";
      case InputType::kSegmentationImage: //panoptic segmentation
        return "Segmentation Image";
      case InputType::kDetectronLabels:
        return "Detectron Labels";
      case InputType::kVertexMap:
        return "Vertex Map";
      case InputType::kValidityImage:
        return "Validity Image";
      case InputType::kPointCloud:
        return "Point Cloud";
      case InputType::kPointColor:
        return "Point Color";
      case InputType::kNormalImage:
        return "Normal Image";
      case InputType::kUncertaintyImage:
        return "Uncertainty Image";
      default:
        return "Unknown Input";
    }
  }

  using InputTypes = std::unordered_set<InputType>;

  // Construction.
  InputData() = default;
  virtual ~InputData() = default;

  /* Data input */
  void setT_M_C(const Transformation& T_M_C) { T_M_C_ = T_M_C; } //pose from camera to map frame
  void setTimeStamp(double timestamp) { timestamp_ = timestamp; }
  void setFrameName(const std::string& frame_name) {
    sensor_frame_name_ = frame_name;
  }
  void setDepthImage(const cv::Mat& depth_image) {
    depth_image_ = depth_image;
    contained_inputs_.insert(InputType::kDepthImage);
  }
  void setColorImage(const cv::Mat& color_image) {
    color_image_ = color_image;
    contained_inputs_.insert(InputType::kColorImage);
  }
  void setIdImage(const cv::Mat& id_image) { //panoptic segmentation
    id_image_ = id_image;
    contained_inputs_.insert(InputType::kSegmentationImage); 
  }
  void setNormalImage(const cv::Mat& normal_image) { //newly added, normal image
    normal_image_ = normal_image;
    contained_inputs_.insert(InputType::kNormalImage); 
  }
  //or rangenet labels
  void setDetectronLabels(const DetectronLabels& labels) {
    detectron_labels_ = labels;
    contained_inputs_.insert(InputType::kDetectronLabels);
  }
  void setVertexMap(const cv::Mat& vertex_map) {
    vertex_map_ = vertex_map;
    contained_inputs_.insert(InputType::kVertexMap);
  }
  void setValidityImage(const cv::Mat& validity_image) {
    validity_image_ = validity_image;
    contained_inputs_.insert(InputType::kValidityImage);
  }
  void setPointCloud(const Pointcloud& point_cloud) {
    point_cloud_ = point_cloud;
    contained_inputs_.insert(InputType::kPointCloud);
  }
  void setPointColor(const Colors& point_color) {
    point_color_ = point_color;
    contained_inputs_.insert(InputType::kPointColor);
  }
  void setUncertaintyImage(const cv::Mat& uncertainty_image) {
    uncertainty_image_ = uncertainty_image;
    contained_inputs_.insert(InputType::kUncertaintyImage);
  }
  void setSubmapIDList(const std::vector<int>& submap_id_list) {
    submap_ids_ = submap_id_list;
  }

  // Access.
  // Access to constant data.
  const Transformation& T_M_C() const { return T_M_C_; }
  const std::string& sensorFrameName() const { return sensor_frame_name_; }
  double timestamp() const { return timestamp_; }
  const cv::Mat& depthImage() const { return depth_image_; }
  const cv::Mat& colorImage() const { return color_image_; }
  const DetectronLabels& detectronLabels() const { return detectron_labels_; }
  const cv::Mat& vertexMap() const { return vertex_map_; }
  const cv::Mat& idImage() const { return id_image_; }
  const cv::Mat& validityImage() const { return validity_image_; }
  const cv::Mat& normalImage() const { return normal_image_; }
  const cv::Mat& uncertaintyImage() const { return uncertainty_image_; }
  const Pointcloud& pointCloud() const { return point_cloud_; }
  const Colors& pointColor() const { return point_color_; }
  const std::vector<int>& submapIDList() const { return submap_ids_; }

  // Access to modifyable data.
  cv::Mat* idImagePtr() { return &id_image_; }
  cv::Mat* validityImagePtr() { return &validity_image_; }

  // Tools.
  bool has(InputType input_type) const {
    return contained_inputs_.find(input_type) != contained_inputs_.end();
  }

 private:
  friend InputDataUser;
  friend class InputSynchronizer;

  // Permanent data.
  Transformation T_M_C_;  // Transform from Camera/Sensor (C) to Mission (M).
  std::string sensor_frame_name_;  // Sensor frame name, taken from depth data.
  double timestamp_ = 0.0;         // Timestamp of the inputs.

  // Common Input data.
  cv::Mat depth_image_;  // Float depth image (CV_32FC1).
  cv::Mat color_image_;  // BGR (CV_8U).
  cv::Mat id_image_;     // Mutable assigned ids as ints (CV_32SC1). //semantic segmentation label
  cv::Mat normal_image_; // normal (CV_32FC3)
  cv::Mat uncertainty_image_; // Float image containing uncertainty information (CV_32FC1) [0-1]

  Pointcloud point_cloud_;
  Colors point_color_;

  std::vector<int> submap_ids_;
  
  // Common derived data.
  cv::Mat vertex_map_;      // XYZ points (CV32FC3), can be compute via camera.
  cv::Mat validity_image_;  // 0-1 image for valid pixels (CV_8UC1).

  // Optional Input data.
  DetectronLabels detectron_labels_;

  // Content tracking.
  InputData::InputTypes contained_inputs_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_INPUT_DATA_H_
