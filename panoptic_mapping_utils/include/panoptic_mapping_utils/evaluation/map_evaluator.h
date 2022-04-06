#ifndef PANOPTIC_MAPPING_UTILS_EVALUATION_MAP_EVALUATOR_H_
#define PANOPTIC_MAPPING_UTILS_EVALUATION_MAP_EVALUATOR_H_

#include <memory>
#include <string>
#include <vector>

#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <panoptic_mapping/3rd_party/nanoflann.hpp>
#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <panoptic_mapping_msgs/SaveLoadMap.h>
#include <panoptic_mapping_ros/visualization/submap_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <voxblox_ros/tsdf_server.h>

namespace panoptic_mapping {

/**
 * @brief Evaluation tools in a ROS node. Initially based on the
 * voxblox_ros/voxblox_eval.cc code.
 * NOTE(py) Two modes basically:
 * 1. Iterate through gt points, query each point's sdf in the pano map
 * the queried sdf is regarded as the error
 * 2. Iterate through the mesh vertices, query each vertice's nearest 
 * point in the gt point cloud (by Kd-tree), the queried distance is 
 * regarded as the error
 */
class MapEvaluator {
 public:
  struct EvaluationRequest // actually the config
      : public config_utilities::Config<EvaluationRequest> {
    int verbosity = 4;

    // Data handling.
    std::string map_file;
    std::string esdf_file_path;
    std::string ground_truth_pointcloud_file;
    std::string output_suffix = "evaluation_data";

    // Evaluation
    float trunc_dist = -1.0; // if < 0, it act as a positive ratio of the voxel size, if > 0, it's a fixed value
    float occ_voxel_size_ratio = 0.866;
    float tsdf_min_weight = 1e-6;
    float inlier_thre = 0.1;   // m, used for both mesh and tsdf accuracy evaluation, actually useless
    bool visualize = true;
    bool evaluate = true;
    bool compute_coloring = true;  // Use map_file to load and display.
    bool ignore_truncated_points = false;
    bool color_by_max_error = false;  // false: color by average error
    bool color_by_mesh_distance = true;  // true: iterate through mesh, false: iterate over gt points.
    bool is_single_tsdf = false;
    bool include_all_submaps = false;
    bool use_chamfer_dist = false;
    bool visualize_esdf_error = false; 
    bool vis_occ_esdf_error = true; // visualize_esdf_error should be enabled. if false, then visualize the gt point cloud referenced esdf error

    EvaluationRequest() { setConfigName("MapEvaluator::EvaluationRequest"); }

   protected:
    void setupParamsAndPrinting() override;
  };

  // nanoflann pointcloud adapter.
  struct TreeData {
    std::vector<Point> points;

    inline std::size_t kdtree_get_point_count() const { return points.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
      if (dim == 0)
        return points[idx].x();
      else if (dim == 1)
        return points[idx].y();
      else
        return points[idx].z();
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
      return false;
    }
  };
  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, TreeData>, TreeData, 3>
      KDTree;

  // Constructor.
  MapEvaluator(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  virtual ~MapEvaluator() = default;

  // Access.
  bool evaluate(const EvaluationRequest& request);
  void publishVisualization();
  bool setupMultiMapEvaluation();

  // Services.
  bool evaluateMapCallback(
      panoptic_mapping_msgs::SaveLoadMap::Request& request,     // NOLINT
      panoptic_mapping_msgs::SaveLoadMap::Response& response);  // NOLINT

 private:
  std::string computeReconstructionError(const EvaluationRequest& request);
  std::string computeMeshError(const EvaluationRequest& request);
  std::string computeEsdfError(const EvaluationRequest& request);
  void visualizeReconstructionError(const EvaluationRequest& request);
  void buildKdTreeGt();
  void buildKdTreeOcc();
  void buildKdTreeMesh();

 private:
  // ROS.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Files.
  std::ofstream output_file_;
  
  // Unified truncation distance for the evaluation
  float trunc_dist_;

  // Stored data.
  std::unique_ptr<pcl::PointCloud<pcl::PointXYZ>> gt_ptcloud_;
  std::unique_ptr<pcl::PointCloud<pcl::PointXYZ>> occ_ptcloud_;
  std::unique_ptr<pcl::PointCloud<pcl::PointXYZ>> mesh_ptcloud_;
  std::shared_ptr<SubmapCollection> submaps_;
  std::shared_ptr<TsdfLayer> voxblox_;
  bool use_voxblox_;
  std::string target_directory_;
  std::string target_map_name_;
  std::unique_ptr<PlanningInterface> planning_;
  std::unique_ptr<SubmapVisualizer> visualizer_;
  TreeData kdtree_data_gt_;
  TreeData kdtree_data_occ_;
  TreeData kdtree_data_mesh_;
  std::unique_ptr<KDTree> kdtree_gt_;
  std::unique_ptr<KDTree> kdtree_occ_;
  std::unique_ptr<KDTree> kdtree_mesh_;

  // Multi Map Evaluations.
  ros::ServiceServer process_map_srv_;
  EvaluationRequest request_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_UTILS_EVALUATION_MAP_EVALUATOR_H_
