#ifndef PANOPTIC_MAPPING_ROS_VISUALIZATION_SUBMAP_VISUALIZER_H_
#define PANOPTIC_MAPPING_ROS_VISUALIZATION_SUBMAP_VISUALIZER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/common/globals.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <ros/node_handle.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/MultiMesh.h>
#include <voxblox_ros/mesh_vis.h>

namespace panoptic_mapping {

class SubmapVisualizer {
 public:
  // Config.
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 1;
    std::string visualization_mode = "all";  // Initial visualization mode.
    std::string color_mode = "color";        // Initial color mode.
    int submap_color_discretization = 20;
    bool visualize_mesh = true;
    bool visualize_tsdf_blocks = false;
    bool visualize_occ_voxels = false;
    bool visualize_slice = true; // slice or space point cloud
    bool visualize_free_space_submap = true;
    bool visualize_free_space_tsdf = false;
    bool visualize_free_space_esdf = false;
    bool visualize_free_space_gsdf = false;
    bool visualize_free_space_esdf_error = true;
    bool visualize_ground_tsdf = false;
    bool visualize_bounding_volumes = false;
    bool include_free_space = false;
    float slice_height = 0.0f;
    float occ_voxel_size_ratio = 0.875f;
    float tsdf_min_weight = 1e-6;
    // transparency alpha for visualization
    float alpha_occ = 0.15f;
    float alpha_block = 0.2f;
    float alpha_bounding = 0.05f;

    std::string ros_namespace;

    Config() { setConfigName("SubmapVisualizer"); }

   protected:
    void setupParamsAndPrinting() override;
    void fromRosParam() override;
    void printFields() const override;
    void checkParams() const override;
  };

  // Constructors.
  SubmapVisualizer(const Config& config, std::shared_ptr<Globals> globals,
                   bool print_config = true);
  virtual ~SubmapVisualizer() = default;

  // Visualization modes.
  enum class ColorMode {
    kColor = 0,
    kNormals,
    kSubmaps,
    kInstances,
    kClasses,
    kChange,
    kClassification,
    kPersistent,
    kUncertainty,
    kEntropy,
    kHeight
  };
  enum class VisualizationMode {
    kAll = 0,
    kActive,
    kActiveOnly,
    kInactive,
    kPersistent
  };

  // Visualization mode conversion.
  static ColorMode colorModeFromString(const std::string& color_mode);
  static std::string colorModeToString(ColorMode color_mode);
  static VisualizationMode visualizationModeFromString(
      const std::string& visualization_mode);
  static std::string visualizationModeToString(
      VisualizationMode visualization_mode);

  // Visualization message creation.
  virtual std::vector<voxblox_msgs::MultiMesh> generateMeshMsgs(
      SubmapCollection* submaps);
  virtual visualization_msgs::MarkerArray generateBlockMsgs(
      const SubmapCollection& submaps);

  virtual pcl::PointCloud<pcl::PointXYZI> generateSubmapTsdfMsg(
      const Submap& submap, bool vis_slice, float slice_height);
  virtual pcl::PointCloud<pcl::PointXYZI> generateSubmapEsdfMsg(
      const Submap& submap, bool vis_slice, float slice_height);
  virtual pcl::PointCloud<pcl::PointXYZI> generateSubmapGsdfMsg(
      const Submap& submap, bool vis_slice, float slice_height);
  virtual pcl::PointCloud<pcl::PointXYZRGB> generateEsdfErrorMsg(
    const EsdfLayer& esdf_layer, bool vis_slice, float slice_height); 
  virtual pcl::PointCloud<pcl::PointXYZI> generateEsdfMsg(
    const EsdfLayer& esdf_layer, bool vis_slice, float slice_height); 
  virtual pcl::PointCloud<pcl::PointXYZI> generateTsdfMsg(
    const TsdfLayer& tsdf_layer, bool vis_slice, float slice_height); 
  virtual pcl::PointCloud<pcl::PointXYZI> generateGsdfMsg(
    const TsdfLayer& tsdf_layer, bool vis_slice, float slice_height); 
  virtual pcl::PointCloud<pcl::PointXYZI> generateGroundTsdfMsg(
      const SubmapCollection& submaps);
  virtual visualization_msgs::MarkerArray generateBoundingVolumeMsgs(
      const SubmapCollection& submaps);
  virtual visualization_msgs::MarkerArray generateOccVoxelMsgs(
      const SubmapCollection& submaps);
  
  // Publish visualization requests.
  virtual void visualizeAll(SubmapCollection* submaps);
  virtual void visualizeMeshes(SubmapCollection* submaps);
  virtual void visualizeOccupiedVoxels(SubmapCollection& submaps);
  virtual void visualizeTsdfBlocks(const SubmapCollection& submaps);
  virtual void visualizeFreeSpace(const Submap& freespace_submap);
  virtual void visualizeEsdf(const EsdfLayer& esdf_layer);
  virtual void visualizeTsdf(const TsdfLayer& tsdf_layer);
  virtual void visualizeGsdf(const TsdfLayer& tsdf_layer);
  virtual void visualizeEsdfError(const EsdfLayer& esdf_layer);
  virtual void visualizeGroundTsdf(const SubmapCollection& submaps); // deprecated
  virtual void visualizeBoundingVolume(const SubmapCollection& submaps);
  virtual void publishTfTransforms(const SubmapCollection& submaps);

  // Interaction.
  virtual void reset();
  virtual void clearMesh();
  virtual void setVisualizationMode(VisualizationMode visualization_mode);
  virtual void setColorMode(ColorMode color_mode);
  virtual void setGlobalFrameName(const std::string& frame_name) {
    global_frame_name_ = frame_name;
  }

 protected:
  static const Color kUnknownColor_;

  struct SubmapVisInfo {
    // General.
    int id = 0;  // Corresponding submap id.
    std::string name_space;

    // Visualization data.
    bool republish_everything = false;
    bool was_deleted = false;
    bool change_color = true;
    Color color = kUnknownColor_;
    float alpha = 1.0;

    // Tracking.
    ChangeState previous_change_state;        // kChange
    bool was_active;                          // kActive
    voxblox::BlockIndexList previous_blocks;  // Track deleted blocks.
  };

  virtual void updateVisInfos(const SubmapCollection& submaps);
  virtual void setSubmapVisColor(const Submap& submap, SubmapVisInfo* info);
  virtual void generateClassificationMesh(Submap* submap,
                                          voxblox_msgs::Mesh* mesh);

 protected:
  // Settings.
  VisualizationMode visualization_mode_;
  ColorMode color_mode_;
  std::string global_frame_name_ = "world";

  // Members.
  std::shared_ptr<Globals> globals_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  voxblox::ExponentialOffsetIdColorMap id_color_map_;

  // Cached / tracked data.
  std::unordered_map<int, SubmapVisInfo> vis_infos_;
  bool vis_infos_are_updated_ = false;
  const SubmapCollection* previous_submaps_ =
      nullptr;  // Only for tracking, not for use!

  // ROS.
  ros::NodeHandle nh_;
  ros::Publisher freespace_tsdf_pub_;
  ros::Publisher freespace_esdf_pub_;
  ros::Publisher freespace_gsdf_pub_;
  ros::Publisher freespace_esdf_error_pub_;
  ros::Publisher ground_tsdf_pub_;
  ros::Publisher mesh_pub_;
  ros::Publisher occ_voxels_pub_;
  ros::Publisher tsdf_blocks_pub_;
  ros::Publisher bounding_volume_pub_;

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      SubmapVisualizer, SubmapVisualizer, std::shared_ptr<Globals>>
      registration_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_VISUALIZATION_SUBMAP_VISUALIZER_H_
