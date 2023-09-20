#ifndef PANOPTIC_MAPPING_ROS_PANOPTIC_MAPPER_H_
#define PANOPTIC_MAPPING_ROS_PANOPTIC_MAPPER_H_

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/common/globals.h>
#include <panoptic_mapping/integration/tsdf_integrator_base.h>
#include <panoptic_mapping/map/submap.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <panoptic_mapping/map_management/map_manager_base.h>
#include <panoptic_mapping/tools/data_writer_base.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <panoptic_mapping/tools/thread_safe_submap_collection.h>
// #include "panoptic_mapping/tools/map_evaluator.h"
#include <panoptic_mapping/tracking/id_tracker_base.h>
#include <panoptic_mapping_msgs/SaveLoadMap.h>
#include <panoptic_mapping_msgs/SetVisualizationMode.h>
#include <ros/ros.h>
#include <std_srvs/Empty.h>

#include "panoptic_mapping_ros/conversions/conversions.h"
#include "panoptic_mapping_ros/input/input_synchronizer.h"
#include "panoptic_mapping_ros/transformer/transformer.h"
#include "panoptic_mapping_ros/visualization/planning_visualizer.h"
#include "panoptic_mapping_ros/visualization/submap_visualizer.h"
#include "panoptic_mapping_ros/visualization/tracking_visualizer.h"



namespace panoptic_mapping {

class PanopticMapper {
 public:
  // Config.
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 2;

    bool output_data = false;
    std::string output_base = "~";

    // Frame name used for the global frame (often mission, world, or odom).
    std::string global_frame_name = "mission"; // "world"

    // How frequently to perform tasks. Execution period in seconds. Use -1 for
    // every frame, 0 for never.
    float visualization_interval = -1.f;
    float data_logging_interval = 0.f;
    float print_timing_interval = 0.f;
    float esdf_update_interval = 0.f;
    float robot_mesh_interval = 0.f;

    int time_gap_second = 0;
    int time_gap_nsecond = 0;

    // If true maintain and update the threadsafe submap collection for access 
    // NOTE(py): used for save the updated map? however, it's quite time-consuming
    bool use_threadsafe_submap_collection = false;

    // Number of threads used for ROS spinning.
    int ros_spinner_threads = std::thread::hardware_concurrency();

    // Frequency in seconds in which the input queue is queried.
    float check_input_interval = 0.001f;

    // If true loaded submaps change states are set to unknown, otherwise to
    // persistent.
    bool load_submaps_conservative = true;

    // If true, finish mapping and shutdown the panoptic mapper when no frames
    // are received for 3 seconds after the first frame was received.
    bool shutdown_when_finished = false;

    // Set this string to automatically save the map to the specified file when
    // shutting down when finished.
    std::string save_map_path_when_finished = "";

    std::string robot_mesh_file = "";

    std::string robot_frame_name = ""; // used only when querying transformation from tf

    // If true, display units when printing the component configs.
    bool display_config_units = true;

    // If true, indicate the default values when printing component configs.
    bool indicate_default_values = true;

    // import the input data from rosbag or not.
    bool input_point_cloud = true;

    // input sensor is lidar or depth camera
    bool use_lidar = true;

    // if the input is the point cloud, firstly convert it to the range image
    bool use_range_image = true;

    // estimate normal for each frame
    bool estimate_normal = true;

    // filter moving objects or not
    bool filter_moving_objects = false;

    // filter the depth image input or not
    bool filter_depth_image = false;

    // If apply depth erosion, the size (radius) in pixel
    int filter_depth_erosion_size = 3;

    // assign different color to different instance for foreground objects 
    bool use_panoptic_color = true;

    float msg_latency_s = 0.0f;

    float robot_mesh_scale = 1.0f;

    Config() { setConfigName("PanopticMapper"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  // Construction.
  PanopticMapper(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  virtual ~PanopticMapper() = default;

  // ROS callbacks.
  // Timers.
  void publishVisualizationCallback(const ros::TimerEvent&);
  void publishRobotMeshCallback(const ros::TimerEvent&);

  void dataLoggingCallback(const ros::TimerEvent&);
  void printTimingsCallback(const ros::TimerEvent&);
  void inputRGBDCallback(const ros::TimerEvent&);
  void updateFreeEsdfCallback(const ros::TimerEvent&);

  // Services.
  bool saveMapCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT
  bool loadMapCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT
  bool saveMeshCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT
  bool saveMergedMeshCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT
  bool saveFreeEsdfCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT
  bool setVisualizationModeCallback(
      panoptic_mapping_msgs::SetVisualizationMode::Request& request,    // NOLINT
      panoptic_mapping_msgs::SetVisualizationMode::Response& response); // NOLINT
  bool saveSceneGraphCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT

  bool printTimingsCallback(std_srvs::Empty::Request& request,      // NOLINT
                            std_srvs::Empty::Response& response);   // NOLINT
  bool finishMappingCallback(std_srvs::Empty::Request& request,     // NOLINT
                             std_srvs::Empty::Response& response);  // NOLINT

  void inputPointCloud(const sensor_msgs::PointCloud2::Ptr& pointcloud);

  // void integratePointcloud(const Transformation& T_G_C,
  //                          const Pointcloud& ptcloud_C, const Colors& colors,
  //                          const bool is_freespace_pointcloud = false);

  // Processing.
  // Integrate a set of input images. The input is usually gathered from ROS
  // topics and provided by the InputSynchronizer.
  void processInput(InputData* input);

  // Performs various post-processing actions.
  // NOTE(schmluk): This is currently a preliminary tool to play around with.
  void finishMapping();

  // ESDF mapping
  /// Call this to update the ESDF based on latest state of the occupancy map,
  /// considering only the newly updated parts of the occupancy map (checked
  /// with the ESDF updated bit in Update::Status).
  // void updateFreeEsdfFromOcc();

  /// Call this to update the Occupancy map based on latest state of the TSDF
  /// map
  void updateFreeOccFromTsdf();

  void updateFreeEsdfFromTsdf();

  void updateNonFreeOccFromTsdf();

  /**
   * Gets the next pointcloud that has an available transform to process from
   * the queue.
   */
  bool getNextPointcloudFromQueue(
      std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
      sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C);

  virtual void processPointCloudMessageAndInsert(
      const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
      const Transformation& T_G_C, const bool is_freespace_pointcloud);

  // IO.
  bool saveMap(const std::string& file_path);
  bool loadMap(const std::string& file_path);
  bool saveMesh(const std::string& file_path);
  bool saveFreeEsdf(const std::string& file_path);

  // Utilities.
  // Print all timings (from voxblox::timing) to console.
  void printTimings() const;

  // Update the meshes and publish the all visualizations of the current map.
  void publishVisualization();

  void publishRobotMesh();

  // Access.
  const SubmapCollection& getSubmapCollection() const { return *submaps_; }
  const ThreadSafeSubmapCollection& getThreadSafeSubmapCollection() const {
    return *thread_safe_submaps_;
  }
  const PlanningInterface& getPlanningInterface() const {
    return *planning_interface_;
  }
  MapManagerBase* getMapManagerPtr() { return map_manager_.get(); }
  const Config& getConfig() const { return config_; }

 private:
  // Setup.
  void setupMembers();
  void setupCollectionDependentMembers();
  // void setupRos();

 private:
  // Node handles.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Subscribers, Publishers, Services, Timers.
  ros::Subscriber pointcloud_sub_;
  ros::Subscriber transform_sub_;
  ros::Publisher transform_pub_;
  ros::Publisher robot_mesh_pub_;
  ros::ServiceServer load_map_srv_;
  ros::ServiceServer save_map_srv_;
  ros::ServiceServer set_visualization_mode_srv_;
  ros::ServiceServer set_color_mode_srv_;
  ros::ServiceServer print_timings_srv_;
  ros::ServiceServer finish_mapping_srv_;
  ros::ServiceServer save_mesh_srv_;
  ros::ServiceServer save_merged_mesh_srv_;
  ros::ServiceServer save_scene_graph_srv_;
  ros::ServiceServer save_free_esdf_srv_;
  ros::ServiceServer evaluate_map_srv_;
  ros::Timer visualization_timer_;
  ros::Timer data_logging_timer_;
  ros::Timer print_timing_timer_;
  ros::Timer update_esdf_timer_;
  ros::Timer input_timer_;
  ros::Timer robot_mesh_timer_;

  // Members.
  const Config config_;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  //std::string world_frame_; // we use global_frame_name here

  // Map.
  std::shared_ptr<SubmapCollection> submaps_;
  std::shared_ptr<ThreadSafeSubmapCollection> thread_safe_submaps_;

  // Mapping.
  std::unique_ptr<IDTrackerBase> id_tracker_;
  std::unique_ptr<TsdfIntegratorBase> tsdf_integrator_;
  std::unique_ptr<MapManagerBase> map_manager_;

  // Tools.
  std::shared_ptr<Globals> globals_;
  std::unique_ptr<InputSynchronizer> input_synchronizer_; // RGBD input synchronizer
  std::unique_ptr<DataWriterBase> data_logger_;
  std::shared_ptr<PlanningInterface> planning_interface_;
  std::shared_ptr<Transformer> transformer_;
  // std::shared_ptr<MapEvaluator> map_evaluator_;

  // Visualization.
  std::unique_ptr<SubmapVisualizer> submap_visualizer_;
  std::unique_ptr<PlanningVisualizer> planning_visualizer_;
  std::unique_ptr<TrackingVisualizer> tracking_visualizer_; //NOTE(py): seems to be not used

  // Colormap to use for intensity pointclouds.
  std::shared_ptr<ColorMap> color_map_;

  // Which processing to perform.
  bool compute_vertex_map_ = false;
  bool compute_validity_image_ = false;

  // Tracking variables.
  ros::WallTime previous_frame_time_ = ros::WallTime::now();
  std::unique_ptr<Timer> frame_timer_;
  ros::Time last_input_;
  bool got_a_frame_ = false;

  // Default namespaces and types for modules are defined here.
  static const std::map<std::string, std::pair<std::string, std::string>>
      default_names_and_types_;
  ros::NodeHandle defaultNh(const std::string& key) const;

  // Last message times for throttling input.
  ros::Time last_msg_time_ptcloud_;

  /**
   * Queue of incoming pointclouds, in case the transforms can't be immediately
   * resolved.
   */
  std::queue<sensor_msgs::PointCloud2::Ptr> pointcloud_queue_;

  /// Will throttle to this message rate.
  ros::Duration min_time_between_msgs_;

  int frame_count_ = 0;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_ROS_PANOPTIC_MAPPER_H_
