#ifndef PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_
#define PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_

#include <algorithm>
#include <memory>
#include <vector>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/ColorRGBA.h>

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/mesh/mesh.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/Layer.h>

#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/common/input_data.h>
#include <panoptic_mapping_msgs/DetectronLabel.h>
#include <panoptic_mapping_msgs/DetectronLabels.h>

namespace panoptic_mapping {

DetectronLabel detectronLabelFromMsg(
    const panoptic_mapping_msgs::DetectronLabel& msg);

DetectronLabels detectronLabelsFromMsg(
    const panoptic_mapping_msgs::DetectronLabels& msg);


enum class MapDerializationAction : uint8_t {
  kUpdate = 0u,
  kMerge = 1u,
  kReset = 2u
};

inline void colorVoxbloxToMsg(const Color& color,
                              std_msgs::ColorRGBA* color_msg) {
  CHECK_NOTNULL(color_msg);
  color_msg->r = color.r / 255.0;
  color_msg->g = color.g / 255.0;
  color_msg->b = color.b / 255.0;
  color_msg->a = color.a / 255.0;
}

inline void colorMsgToVoxblox(const std_msgs::ColorRGBA& color_msg,
                              Color* color) {
  CHECK_NOTNULL(color);
  color->r = static_cast<uint8_t>(color_msg.r * 255.0);
  color->g = static_cast<uint8_t>(color_msg.g * 255.0);
  color->b = static_cast<uint8_t>(color_msg.b * 255.0);
  color->a = static_cast<uint8_t>(color_msg.a * 255.0);
}

inline void pointcloudToPclXYZRGBL(
    const Pointcloud& ptcloud, const Colors& colors, const Labels& labels,
    pcl::PointCloud<pcl::PointXYZRGBL>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZRGBL point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();

    point.r = colors[i].r;
    point.g = colors[i].g;
    point.b = colors[i].b;

    point.label = labels[i].full_label;

    ptcloud_pcl->push_back(point);
  }
}

inline void pointcloudToPclXYZRGB(
    const Pointcloud& ptcloud, const Colors& colors,
    pcl::PointCloud<pcl::PointXYZRGB>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZRGB point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();

    point.r = colors[i].r;
    point.g = colors[i].g;
    point.b = colors[i].b;

    ptcloud_pcl->push_back(point);
  }
}

inline void pointcloudToPclXYZ(const Pointcloud& ptcloud,
                               pcl::PointCloud<pcl::PointXYZ>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZ point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();

    ptcloud_pcl->push_back(point);
  }
}

inline void pointcloudToPclXYZI(const Pointcloud& ptcloud,
                                const std::vector<float>& intensities,
                                pcl::PointCloud<pcl::PointXYZI>* ptcloud_pcl) {
  CHECK_NOTNULL(ptcloud_pcl);
  CHECK_EQ(ptcloud.size(), intensities.size());
  ptcloud_pcl->clear();
  ptcloud_pcl->reserve(ptcloud.size());
  for (size_t i = 0; i < ptcloud.size(); ++i) {
    pcl::PointXYZI point;
    point.x = ptcloud[i].x();
    point.y = ptcloud[i].y();
    point.z = ptcloud[i].z();
    point.intensity = intensities[i];

    ptcloud_pcl->push_back(point);
  }
}

/// Check if all coordinates in the PCL point are finite.
template <typename PCLPoint>
inline bool isPointFinite(const PCLPoint& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

template <typename PCLPoint>
Color convertColor(const PCLPoint& point,
                   const std::shared_ptr<ColorMap>& color_map);

template <>
inline Color convertColor(const pcl::PointXYZRGBL& point,
                          const std::shared_ptr<ColorMap>& /*color_map*/) {
  return Color(point.r, point.g, point.b);
}

// NOTE(py): WHY ? Call of overloaded function is ambiguous

template <>
inline Color convertColor(const pcl::PointXYZRGB& point,
                          const std::shared_ptr<ColorMap>& /*color_map*/) {
  return Color(point.r, point.g, point.b, point.a);
}

template <>
inline Color convertColor(const pcl::PointXYZI& point,
                          const std::shared_ptr<ColorMap>& color_map) {
  return color_map->colorLookup(point.intensity);
}

template <>
inline Color convertColor(const pcl::PointXYZ& /*point*/,
                          const std::shared_ptr<ColorMap>& color_map) {
  return color_map->colorLookup(0);
}

/// Convert pointclouds of different PCL types to a voxblox pointcloud.
/// With color, without label
// template <typename PCLPoint>
// inline void convertPointcloud(
//     const typename pcl::PointCloud<PCLPoint>& pointcloud_pcl,
//     const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
//     Colors* colors) {
//   points_C->reserve(pointcloud_pcl.size());
//   colors->reserve(pointcloud_pcl.size());
//   for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
//     if (!isPointFinite(pointcloud_pcl.points[i])) {
//       continue;
//     }
//     points_C->push_back(Point(pointcloud_pcl.points[i].x,
//                               pointcloud_pcl.points[i].y,
//                               pointcloud_pcl.points[i].z));
//     colors->emplace_back(
//         convertColor<PCLPoint>(pointcloud_pcl.points[i], color_map));
//   }
// }

inline void convertPointcloud(const typename pcl::PointCloud<pcl::PointXYZRGB>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, 
    Pointcloud* points_C,
    Colors* colors) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<pcl::PointXYZRGB>(pointcloud_pcl.points[i], color_map));
  }
}

inline void convertPointcloud(const typename pcl::PointCloud<pcl::PointXYZI>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, 
    Pointcloud* points_C,
    Colors* colors) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<pcl::PointXYZI>(pointcloud_pcl.points[i], color_map));
  }
}

inline void convertPointcloud(const typename pcl::PointCloud<pcl::PointXYZ>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, 
    Pointcloud* points_C,
    Colors* colors) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    colors->emplace_back(
        convertColor<pcl::PointXYZ>(pointcloud_pcl.points[i], color_map));
  }
}

// Convert pointclouds of different PCL types to a voxblox pointcloud.
// With color, with label
// Here we use the label convention of semantic kitti
// TODO(py): add more possibility for the label
// TODO(py): different color for different instance label 
inline void convertPointcloud(
    const typename pcl::PointCloud<pcl::PointXYZRGBL>& pointcloud_pcl,
    const std::shared_ptr<ColorMap>& color_map, Pointcloud* points_C,
    Colors* colors, Labels* labels, bool filter_moving_object,
    bool panoptic_color_on, int id_colors_per_revolution) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  labels->reserve(pointcloud_pcl.size());

  voxblox::ExponentialOffsetIdColorMap id_color_map;
  id_color_map.setItemsPerRevolution(id_colors_per_revolution);

  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }

    Label cur_label(pointcloud_pcl.points[i].label);
    
    // only for semantic kitti dataset
    // Filter those outlier and dynamic objects
    // sem_label <=1 means unlabeled or outlier
    // sem_label > 250 means moving (dynamic) objects
    // TODO(py): in practice, these moving objects should also be considered
    if (cur_label.sem_label <= 1)  // outliers
      continue;
    if (filter_moving_object && cur_label.sem_label > 250)
      continue;
      
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));
    
    if (panoptic_color_on && (cur_label.sem_label < 40 || cur_label.sem_label > 100)) { // foreground objects
      Color cur_color = id_color_map.colorLookup(cur_label.ins_label);
      colors->emplace_back(cur_color);
    } else {
      colors->emplace_back(
          convertColor<pcl::PointXYZRGBL>(pointcloud_pcl.points[i], color_map));
    }

    labels->emplace_back(cur_label); // add definition of Label
  }
}

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_CONVERSIONS_CONVERSIONS_H_

