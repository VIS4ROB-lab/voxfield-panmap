#include "panoptic_mapping_utils/evaluation/map_evaluator.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <pcl/io/ply_io.h>
#include <ros/ros.h>
#include <voxblox/interpolator/interpolator.h>

#include "panoptic_mapping_utils/evaluation/progress_bar.h"

namespace panoptic_mapping {

void MapEvaluator::EvaluationRequest::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("map_file", &map_file);
  setupParam("esdf_file_path", &esdf_file_path);
  setupParam("ground_truth_pointcloud_file", &ground_truth_pointcloud_file);
  setupParam("output_suffix", &output_suffix);
  setupParam("trunc_dist", &trunc_dist);
  setupParam("inlier_thre", &inlier_thre); 
  setupParam("evaluate", &evaluate);
  setupParam("visualize", &visualize);
  setupParam("compute_coloring", &compute_coloring);
  setupParam("color_by_max_error", &color_by_max_error);
  setupParam("color_by_mesh_distance", &color_by_mesh_distance);
  setupParam("ignore_truncated_points", &ignore_truncated_points);
  setupParam("is_single_tsdf", &is_single_tsdf);
  setupParam("include_all_submaps", &include_all_submaps);
  setupParam("use_chamfer_dist", &use_chamfer_dist);
  setupParam("occ_voxel_size_ratio", &occ_voxel_size_ratio);
  setupParam("tsdf_min_weight", &tsdf_min_weight);
  setupParam("visualize_esdf_error", &visualize_esdf_error);
  setupParam("vis_occ_esdf_error", &vis_occ_esdf_error);
}

MapEvaluator::MapEvaluator(const ros::NodeHandle& nh,
                           const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
  auto config =
      config_utilities::getConfigFromRos<SubmapVisualizer::Config>(nh_private_);
  visualizer_ = std::make_unique<SubmapVisualizer>(config, nullptr);
}

bool MapEvaluator::setupMultiMapEvaluation() {
  // Get evaluation configuration (wait till set).
  while (!nh_private_.hasParam("ground_truth_pointcloud_file")) {
    ros::Duration(0.05).sleep();
  }
  request_ = config_utilities::getConfigFromRos<
      panoptic_mapping::MapEvaluator::EvaluationRequest>(nh_private_);
  LOG_IF(INFO, request_.verbosity >= 1) << "\n" << request_.toString();
  if (!request_.isValid(true)) {
    LOG(ERROR) << "Invalid evaluation request.";
    return false;
  }
  use_voxblox_ = false;

  // Load GT cloud.
  gt_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(request_.ground_truth_pointcloud_file,
                                          *gt_ptcloud_) != 0) {
    LOG(ERROR) << "Could not load ground truth point cloud from '"
               << request_.ground_truth_pointcloud_file << "'.";
    return false;
  }
  buildKdTreeGt(); //build kd-tree of the gt point cloud here
  LOG_IF(INFO, request_.verbosity >= 2) << "Loaded ground truth pointcloud";

  // Setup Output File.
  // NOTE(schmluk): The map_file is used to specify the target path here.
  std::string out_file_name =
      request_.map_file + "/" + request_.output_suffix + ".csv";
  output_file_.open(out_file_name, std::ios::out);
  if (!output_file_.is_open()) {
    LOG(ERROR) << "Failed to open output file '" << out_file_name << "'.";
    return false;
  }
  output_file_
      << "MeanGTError [m],StdGTError [m],GTRMSE [m],TotalPoints [1],"
      << "UnknownPoints [1],TruncatedPoints [1],GTInliers [1],MeanMapError [m],"
      << "StdMapError [m],MapRMSE [m],MapInliers [1],MapOutliers [1]\n ";

  // Advertise evaluation service.
  process_map_srv_ = nh_private_.advertiseService(
      "process_map", &MapEvaluator::evaluateMapCallback, this);
  return true;
}

// this function is actually used by single_map_evaluation node
bool MapEvaluator::evaluate(const EvaluationRequest& request) {
  if (!request.isValid(true)) {
    return false;
  }
  LOG_IF(INFO, request.verbosity >= 2) << "Processing: \n"
                                       << request.toString();

  // Load the groundtruth pointcloud.
  if (request.evaluate || request.compute_coloring) {
    if (!request.ground_truth_pointcloud_file.empty()) {
      gt_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
      if (pcl::io::loadPLYFile<pcl::PointXYZ>(
              request.ground_truth_pointcloud_file, *gt_ptcloud_) != 0) {
        LOG(ERROR) << "Could not load ground truth point cloud from '"
                   << request.ground_truth_pointcloud_file << "'.";
        gt_ptcloud_.reset();
        return false;
      }
      buildKdTreeGt(); //build kd-tree of the gt point cloud here
      LOG_IF(INFO, request.verbosity >= 2) << "Loaded ground truth pointcloud";
    }
    if (!gt_ptcloud_) {
      LOG(ERROR) << "No ground truth pointcloud loaded.";
      return false;
    }
  }

  // Load the map to evaluate.
  if (!request.map_file.empty()) {
    // Load the map.
    const std::string extension =
        request.map_file.substr(request.map_file.find_last_of('.'));
    size_t separator = request.map_file.find_last_of('/');
    target_directory_ = request.map_file.substr(0, separator);
    target_map_name_ = request.map_file.substr(
        separator + 1,
        request.map_file.length() - separator - extension.length() - 1);

    if (extension == ".panmap") {
      // Load panoptic map.
      use_voxblox_ = false;
      submaps_ = std::make_shared<SubmapCollection>();
      if (!submaps_->loadFromFile(request.map_file)) {
        LOG(ERROR) << "Could not load panoptic map from '" << request.map_file
                   << "'.";
        submaps_.reset();
        return false;
      }
      planning_ = std::make_unique<PlanningInterface>(submaps_);
      LOG_IF(INFO, request.verbosity >= 2) << "Loaded the target panoptic map.";
    } else if (extension == ".vxblx") { // For simple comparison with the map constructed by voxblox
      use_voxblox_ = true;
      voxblox::io::LoadLayer<voxblox::TsdfVoxel>(request.map_file, &voxblox_);
      LOG_IF(INFO, request.verbosity >= 2) << "Loaded the target voxblox map.";
    } else {
      LOG(ERROR) << "cannot load file of unknown extension '"
                 << request.map_file << "'.";
      return false;
    }
  }
  if (!submaps_ && !use_voxblox_) {
    LOG(ERROR) << "No panoptic map loaded.";
    return false;
  }

  // Setup the unified truncation distance for evaluation
  if (request.trunc_dist > 0)
    trunc_dist_ = request.trunc_dist;

  if (use_voxblox_) {
    if (request.trunc_dist < 0) {
      trunc_dist_ = -1.0 * request.trunc_dist * voxblox_->voxel_size();
    }
  } else {
    if (request.trunc_dist < 0) {
      int free_space_submap_id = 0;
      Submap* free_space_submap = submaps_->getSubmapPtr(free_space_submap_id); 
      trunc_dist_ = -1.0 * request.trunc_dist * free_space_submap->getTsdfLayerPtr()->voxel_size();
    }
  }

  // Setup output file
  if (request.evaluate) {
    std::string out_file_name = target_directory_ + "/" + target_map_name_ +
                                "_" + request.output_suffix + ".csv";
    output_file_.open(out_file_name, std::ios::out);
    if (!output_file_.is_open()) {
      LOG(ERROR) << "Failed to open output file '" << out_file_name << "'.";
      return false;
    }

    // Evaluate.
    LOG_IF(INFO, request.verbosity >= 2) << "Computing reconstruction error:";
    output_file_ << "MeanTsdfError [m],StdTsdfError [m],TsdfRMSE [m],Coverage [%],InlierRatio [%],"
                 << "TotalPoints [#],UnknownPoints [#],ObservedPoints [#],InlierPoints [#],OutlierPoints [#],"
                 << "TruncatedPoints [#],InlierDistTsdf [m],TruncDistTsdf [m],"
                 << "MeanMeshError [m],StdMeshError [m],MeshRMSE [m],MeshInlierRatio [%],"
                 << "MeshInliers [#],MeshOutliers [#],MeanEsdfErrorRefOcc [m],StdEsdfErrorRefOcc [m],"
                 << "EsdfRMSERefOcc [m],TruncDistEsdfRefOcc [m],MeanEsdfErrorRefGT [m],StdEsdfErrorRefGT [m],"
                 << "EsdfRMSERefGT [m],TruncDistEsdfRefGT [m]\n";

    output_file_ << computeReconstructionError(request) << ","
                 << computeMeshError(request) << ","
                 << computeEsdfError(request);
    output_file_.close();
  }

  // Compute visualization if required.
  if (request.compute_coloring) {
    LOG_IF(INFO, request.verbosity >= 2) << "Computing visualization coloring:";
    visualizeReconstructionError(request);
  }

  // Display the mesh.
  if (request.visualize) {
    LOG_IF(INFO, request.verbosity >= 2) << "Publishing mesh.";
    publishVisualization();
  }

  LOG_IF(INFO, request.verbosity >= 2) << "Done.";
  return true;
}

// Actually the tsdf error
std::string MapEvaluator::computeReconstructionError(
    const EvaluationRequest& request) {
  // Go through each point in the ground truth point cloud, 
  // use trilateral interpolation to figure out the
  // distance at that point.

  // Setup.
  // total_points = unknown_points + observed_points
  // observed_points = inliers + outliers
  uint64_t total_points = 0;
  uint64_t unknown_points = 0;
  uint64_t observed_points = 0;
  uint64_t inliers = 0;
  uint64_t outliers = 0;
  uint64_t truncated_points = 0;
  std::vector<float> abserror;
  abserror.reserve(gt_ptcloud_->size());  // Just reserve the worst case.

  // Setup progress bar.
  const uint64_t interval = gt_ptcloud_->size() / 100;
  uint64_t count = 0;
  ProgressBar bar;

  // Evaluate gt pcl based(# gt points within < trunc_dist_)
  std::unique_ptr<voxblox::Interpolator<voxblox::TsdfVoxel>> interp;

  if (use_voxblox_) {
    interp.reset(new voxblox::Interpolator<voxblox::TsdfVoxel>(voxblox_.get()));
  } 

  // for each point in the gt point cloud
  for (const auto& pcl_point : *gt_ptcloud_) {
    const Point point(pcl_point.x, pcl_point.y, pcl_point.z);
    total_points++;

    // Lookup the distance (tsdf)
    float distance;
    bool observed;
    if (use_voxblox_) {
      observed = interp->getDistance(point, &distance, true);
    } else {
      if (request.is_single_tsdf) { // mono map, only with the freespace map
        observed = planning_->getDistance(point, &distance, false, true);
      } else {
        observed = planning_->getDistance(point, &distance, !request.include_all_submaps, false); // not include free space
      }
    }

    // Compute the error.
    if (observed) {
      if (std::abs(distance) > trunc_dist_) {
        truncated_points++;
        if (!request.ignore_truncated_points) { // don't ignore
          abserror.push_back(trunc_dist_); //use the truncation dist
        }
      } 
      else { // not truncated
        abserror.push_back(std::abs(distance));
      }
      if (std::abs(distance) <= request.inlier_thre) {
        inliers++;
      }
      else{
        outliers++;
      }
      observed_points++;
    } else { //unobserved
      unknown_points++;
    }

    // Progress bar.
    if (count % interval == 0) {
      bar.display(static_cast<float>(count) / gt_ptcloud_->size());
    }
    count++;
  }
  bar.display(1.f);

  // Report summary.
  float mean = 0.0;
  float rmse = 0.0;
  for (auto value : abserror) {
    mean += value;
    rmse += std::pow(value, 2);
  }
  if (!abserror.empty()) {
    mean /= static_cast<float>(abserror.size());
    rmse = std::sqrt(rmse / static_cast<float>(abserror.size()));
  }
  float stddev = 0.0;
  for (auto value : abserror) {
    stddev += std::pow(value - mean, 2.0);
  }
  if (abserror.size() > 2) {
    stddev = sqrt(stddev / static_cast<float>(abserror.size() - 1));
  }

  float coverage = 1.0 * observed_points / total_points;
  float inlier_ratio = 1.0 * inliers / observed_points;

  std::stringstream ss;
  ss << mean << "," << stddev << "," << rmse << "," 
     << coverage * 100.0 << "," << inlier_ratio * 100.0 << "," 
     << total_points << "," << unknown_points << "," << observed_points << 
     "," << inliers << "," << outliers << "," << truncated_points
     << "," << request.inlier_thre << "," << trunc_dist_;
  
  //TODO(py): add the evaluation metric of coverage

  if (request.verbosity > 1){
    ROS_INFO("TSDF mapping evaluation:");
    ROS_INFO("Check %d GT points", total_points);
    ROS_INFO("MeanError    [m]: %f", mean);
    ROS_INFO("StdError     [m]: %f", stddev);
    ROS_INFO("RMSE         [m]: %f", rmse);
    ROS_INFO("Inlier ratio [%]: %f", inlier_ratio*100.0);
    ROS_INFO("Coverage     [%]: %f", coverage*100.0);
  }

  return ss.str();
}

// TODO(py): only work for panmap format, add the interface for voxblox format
// The mesh reconstruction error
std::string MapEvaluator::computeMeshError(const EvaluationRequest& request) {

  if (use_voxblox_) { // use voxblox format
    // MeshIntegratorConfig mesh_config;
    // mesh_layer_.reset(new MeshLayer(tsdf_layer_->block_size()));
    // mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
    // mesh_config, tsdf_layer_.get(), mesh_layer_.get()));

    // constexpr bool only_mesh_updated_blocks = false;
    // constexpr bool clear_updated_flag = true;
    // mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);

    // // Publish mesh.
    // visualization_msgs::MarkerArray marker_array;
    // marker_array.markers.resize(1);
    // marker_array.markers[0].header.frame_id = frame_id_;
    // fillMarkerWithMesh(mesh_layer_, color_mode_, &marker_array.markers[0]);
    // mesh_pub_.publish(marker_array);
    LOG(WARNING)<<"currently do not support voxblox format tsdf map";
    return "Nan";
  }
  
  // Setup progress bar.
  float counter = 0.f;
  float max_counter = 0.f;
  ProgressBar bar;
  for (auto& submap : *submaps_) {
    voxblox::BlockIndexList block_list;
    submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
    max_counter += block_list.size();
  }

  // Setup error computation.
  uint64_t inliers = 0;
  uint64_t outliers = 0;
  uint64_t truncated_points = 0;
  std::vector<float> errors_mesh2gt;

  mesh_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();

  // Parse all submaps
  for (auto& submap : *submaps_) {
    if (!request.is_single_tsdf || !request.include_all_submaps) {
      if (submap.getLabel() == PanopticLabel::kFreeSpace ||
          submap.getChangeState() == ChangeState::kAbsent ||
          submap.getChangeState() == ChangeState::kUnobserved) {
        voxblox::BlockIndexList block_list;
        submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
        counter += block_list.size();
        bar.display(counter / max_counter);
        continue; // skip the submap
      }
    }

    // Parse all mesh vertices of the current submap
    voxblox::BlockIndexList block_list;
    submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
    for (auto& block_index : block_list) {
      if (!ros::ok()) {
        return "";
      }
      // For each mesh vertex, do the KNN in the tree built by ground truth point cloud
      // the mesh error for the vertex is just defined as this nearest distance
      // TODO(py): Maybe it's better to use chamfer distance
      for (const Point& point :
           submap.getMeshLayer().getMeshByIndex(block_index).vertices) {
        // Find closest GT point.
        pcl::PointXYZ cur_mesh_vertex(point.x(), point.y(), point.z());
        mesh_ptcloud_->points.push_back(cur_mesh_vertex);

        float query_pt[3] = {point.x(), point.y(), point.z()};
        std::vector<size_t> ret_index(1);
        std::vector<float> out_dist_sqr(1);
        int num_results = kdtree_gt_->knnSearch(&query_pt[0], 1, &ret_index[0],
                                             &out_dist_sqr[0]);

        if (num_results != 0) { //found
          const float error =
              (kdtree_data_gt_.points[ret_index[0]] - point).norm();

          if (error > trunc_dist_) {
            truncated_points++;
            if (!request.ignore_truncated_points) { // don't ignore
              errors_mesh2gt.push_back(trunc_dist_); //use the truncation dist
            }
          } 
          else { //not truncated
            errors_mesh2gt.emplace_back(error);
          }
          if (error <= request.inlier_thre) {
            inliers++;
          } else {
            outliers++;
          }
        }
      }

      // Show progress.
      counter += 1.f;
      bar.display(counter / max_counter);
    }
  }

  // Compute result.
  float mean_mesh2gt = 0.0;
  float rmse_mesh2gt = 0.0;
  for (auto value : errors_mesh2gt) {
    mean_mesh2gt += value;
    rmse_mesh2gt += std::pow(value, 2);
  }
  if (!errors_mesh2gt.empty()) {
    mean_mesh2gt /= static_cast<float>(errors_mesh2gt.size());
    rmse_mesh2gt = std::sqrt(rmse_mesh2gt / static_cast<float>(errors_mesh2gt.size()));
  }
  float stddev = 0.0;
  for (auto value : errors_mesh2gt) {
    stddev += std::pow(value - mean_mesh2gt, 2.0);
  }
  if (errors_mesh2gt.size() > 2) {
    stddev = sqrt(stddev / static_cast<float>(errors_mesh2gt.size() - 1));
  }

  int total_vertices = inliers + outliers;
  float inlier_ratio = 1.0 * inliers / total_vertices;

  std::stringstream ss;
  ss << mean_mesh2gt << "," << stddev << "," << rmse_mesh2gt << "," << inlier_ratio 
     << "," << inliers << "," << outliers;


  if (request.verbosity > 1){
    ROS_INFO("Mesh reconstruction evaluation:");
    ROS_INFO("Check %d mesh vertices", total_vertices);
    ROS_INFO("MeanError    [m]: %f", mean_mesh2gt);
    ROS_INFO("StdError     [m]: %f", stddev);
    ROS_INFO("RMSE         [m]: %f", rmse_mesh2gt);
    ROS_INFO("Inlier ratio [%]: %f", inlier_ratio * 100.0);
  }

  if (request.use_chamfer_dist) {
    // ROS_INFO("Build kd tree for the mesh vertices point cloud [ %d points ]", mesh_ptcloud_->points.size());
    buildKdTreeMesh();

    int total_points = 0;
    int inlier_points = 0;
    std::vector<float> errors_gt2mesh;
    const uint64_t interval = gt_ptcloud_->size() / 100;
    ProgressBar bar_2;

    // for each point in the gt point cloud
    for (const auto& pcl_point : *gt_ptcloud_) {
      
      float query_pt[3] = {pcl_point.x, pcl_point.y, pcl_point.z};
      std::vector<size_t> ret_index(1);
      std::vector<float> out_dist_sqr(1);
      int num_results = kdtree_mesh_->knnSearch(&query_pt[0], 1, &ret_index[0],
                                                &out_dist_sqr[0]);
      if (num_results != 0) { // found
        const float error = std::sqrt(out_dist_sqr[0]);
          //(kdtree_data_gt_.points[ret_index[0]] - point).norm();

        if (error > trunc_dist_) {
          if (!request.ignore_truncated_points) { // don't ignore
              errors_gt2mesh.push_back(trunc_dist_); //use the truncation dist
          }
        } else { //not truncated
          errors_gt2mesh.emplace_back(error);
          inlier_points++;
        }
      }
      total_points++;

      // Progress bar.
      if (total_points % interval == 0) {
        bar_2.display(static_cast<float>(total_points) / gt_ptcloud_->size());
      }
    }
    bar_2.display(1.f);

    // Compute result.
    float mean_gt2mesh = 0.0;
    float rmse_gt2mesh = 0.0;
    for (auto value : errors_gt2mesh) {
      mean_gt2mesh += value;
      rmse_gt2mesh += std::pow(value, 2);
    }
    if (!errors_gt2mesh.empty()) {
      mean_gt2mesh /= static_cast<float>(errors_gt2mesh.size());
      rmse_gt2mesh = std::sqrt(rmse_gt2mesh / static_cast<float>(errors_gt2mesh.size()));
    }
    float stddev = 0.0;
    for (auto value : errors_gt2mesh) {
      stddev += std::pow(value - mean_gt2mesh, 2.0);
    }
    if (errors_gt2mesh.size() > 2) {
      stddev = sqrt(stddev / static_cast<float>(errors_gt2mesh.size() - 1));
    }

    float chamfer_dist_l2_root = std::sqrt(0.5 * (rmse_gt2mesh * rmse_gt2mesh + rmse_mesh2gt * rmse_mesh2gt));
    float chamfer_dist_l1 = 0.5 * (mean_gt2mesh + mean_mesh2gt);
    float mesh_coverage_recall = 1.0 * inlier_points / total_points;  // inlier ratio in the gt point cloud


    if (request.verbosity > 1){
      ROS_INFO("Chamfer-L2   [m]: %f", chamfer_dist_l2_root);
      ROS_INFO("Chamfer-L1   [m]: %f", chamfer_dist_l1);
      ROS_INFO("Mesh coverage[%]: %f", mesh_coverage_recall * 100.0);
    }
  }
  
  return ss.str();
}

// Evaluate the accuracy of ESDF mapping, referenced to current occupancy map
// add it later to a seperate class
std::string MapEvaluator::computeEsdfError(const EvaluationRequest& request) {

  occ_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  
  int free_space_submap_id = 0;
  Submap* free_space_submap = submaps_->getSubmapPtr(free_space_submap_id);
  
  voxblox::BlockIndexList tsdf_blocks;
  std::shared_ptr<TsdfLayer> tsdf_layer = free_space_submap->getTsdfLayerPtr();

  tsdf_layer->getAllAllocatedBlocks(&tsdf_blocks);
 
  int voxels_per_side = tsdf_layer->voxels_per_side();
  float voxel_size = tsdf_layer->voxel_size();

  const float weight_thre = request.tsdf_min_weight;
  const float dist_ratio_thre = request.occ_voxel_size_ratio;
  float dist_thre = voxel_size * dist_ratio_thre;

  for (const BlockIndex& block_index : tsdf_blocks) {
    TsdfBlock::ConstPtr tsdf_block =
        tsdf_layer->getBlockPtrByIndex(block_index);
    if (!tsdf_block) {
      continue;
    }
    const size_t num_voxels_per_block = tsdf_block->num_voxels();
    for (size_t lin_index = 0u; lin_index < num_voxels_per_block; ++lin_index) {
      const TsdfVoxel& tsdf_voxel =
          tsdf_block->getVoxelByLinearIndex(lin_index);
      if (tsdf_voxel.weight < weight_thre || std::abs(tsdf_voxel.distance) > dist_thre) continue;
      // those left are the occupied voxels
      VoxelIndex voxel_index =
          tsdf_block->computeVoxelIndexFromLinearIndex(lin_index);
      GlobalIndex global_index = voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, voxels_per_side);

      Point center_point = voxblox::getCenterPointFromGridIndex(global_index, voxel_size);

      pcl::PointXYZ pt(center_point(0), center_point(1), center_point(2));
      occ_ptcloud_->points.push_back(pt);
    }
  }
  // ROS_INFO("Build kd tree for the occupied voxel center point cloud [ %d points ]", occ_ptcloud_->points.size());
  buildKdTreeOcc();

  std::shared_ptr<EsdfLayer> esdf_layer;
  if (!voxblox::io::LoadLayer<EsdfVoxel>(request.esdf_file_path, &esdf_layer))
    LOG(ERROR) << "Could not load voxblox esdf map.";

  // std::shared_ptr<EsdfLayer> esdf_layer = free_space_submap->getEsdfLayerPtr();
  voxblox::BlockIndexList esdf_blocks;
  esdf_layer->getAllAllocatedBlocks(&esdf_blocks);

  // visualizer_->generateSubmapEsdfMsg();

  // LOG(INFO) << "Found "<< esdf_blocks.size() << " Esdf blocks";

  // Setup progress bar.
  float counter = 0.f;
  float max_counter = esdf_blocks.size();
  ProgressBar bar;

  std::vector<float> error_gt_ref;
  std::vector<float> error_occ_ref;

  for (const BlockIndex& block_index : esdf_blocks) {
    EsdfBlock::Ptr esdf_block =
        esdf_layer->getBlockPtrByIndex(block_index);
    if (!esdf_block) {
      continue;
    }
    const size_t num_voxels_per_block = esdf_block->num_voxels();
    for (size_t lin_index = 0u; lin_index < num_voxels_per_block; ++lin_index) {
      // EsdfVoxel esdf_voxel =
      //     esdf_block->getVoxelByLinearIndex(lin_index);

      
      VoxelIndex voxel_index =
          esdf_block->computeVoxelIndexFromLinearIndex(lin_index);
      GlobalIndex global_index = voxblox::getGlobalVoxelIndexFromBlockAndVoxelIndex(
          block_index, voxel_index, voxels_per_side);
      
      EsdfVoxel* esdf_voxel = esdf_layer->getVoxelPtrByGlobalIndex(global_index);
      if (!esdf_voxel->observed) continue; 

      Point point = voxblox::getCenterPointFromGridIndex(global_index, voxel_size);
      float query_pt[3] = {point(0), point(1), point(2)};
      std::vector<size_t> ret_index_gt(1);
      std::vector<float> out_dist_sqr_gt(1);
      int num_results_gt_ref = kdtree_gt_->knnSearch(&query_pt[0], 1, &ret_index_gt[0],
                                             &out_dist_sqr_gt[0]);
      
      std::vector<size_t> ret_index_occ(1);
      std::vector<float> out_dist_sqr_occ(1);
      int num_results_occ_ref = kdtree_occ_->knnSearch(&query_pt[0], 1, &ret_index_occ[0],
                                             &out_dist_sqr_occ[0]);

      float dist_esdf = std::abs(esdf_voxel->distance);
      // if (dist_esdf > 0)
      //   LOG(INFO) <<"esdf serializaed successfully";

      // TODO, two reference: occ_ptcloud and gt_ptcloud
      // gt point cloud reference
      if (num_results_gt_ref != 0) { // found
        float dist_gt = (kdtree_data_gt_.points[ret_index_gt[0]] - point).norm();
        float dist_error = std::abs(dist_gt - dist_esdf); // TODO (py): consider the case when the sign is opposite
        if (dist_error > trunc_dist_) {
          if (!request.ignore_truncated_points) {  // don't ignore
            error_gt_ref.push_back(trunc_dist_); //use the truncation dist
          }
          if (!request.vis_occ_esdf_error) {
            esdf_voxel->error = trunc_dist_;
          }
        } else {
          error_gt_ref.push_back(dist_error);
          if (!request.vis_occ_esdf_error) {
            esdf_voxel->error = dist_error;
          }
        }
      }

      // occ center reference
      if (num_results_occ_ref != 0) { // found
        float dist_occ = (kdtree_data_occ_.points[ret_index_occ[0]] - point).norm();
        float dist_error = std::abs(dist_occ - dist_esdf);
        if (dist_error > trunc_dist_) {
          if (!request.ignore_truncated_points) {  // don't ignore
            error_occ_ref.push_back(trunc_dist_); //use the truncation dist
          }
          if (request.vis_occ_esdf_error) {
            esdf_voxel->error = trunc_dist_;
          }
        } else {
          error_occ_ref.push_back(dist_error);
          if (request.vis_occ_esdf_error) {
            esdf_voxel->error = dist_error;
          }
        }
      }
    }

    // Show progress.
    counter += 1.f;
    bar.display(counter / max_counter);
  }

  // Compute result.
  float mean_gt_ref = 0.0;
  float rmse_gt_ref = 0.0;
  for (auto value : error_gt_ref) {
    mean_gt_ref += value;
    rmse_gt_ref += std::pow(value, 2);
  }
  if (!error_gt_ref.empty()) {
    mean_gt_ref /= static_cast<float>(error_gt_ref.size());
    rmse_gt_ref = std::sqrt(rmse_gt_ref / static_cast<float>(error_gt_ref.size()));
  }
  float stddev_gt_ref = 0.0;
  for (auto value : error_gt_ref) {
    stddev_gt_ref += std::pow(value - mean_gt_ref, 2.0);
  }
  if (error_gt_ref.size() > 2) {
    stddev_gt_ref = sqrt(stddev_gt_ref / static_cast<float>(error_gt_ref.size() - 1));
  }

  float mean_occ_ref = 0.0;
  float rmse_occ_ref = 0.0;
  for (auto value : error_occ_ref) {
    mean_occ_ref += value;
    rmse_occ_ref += std::pow(value, 2);
  }
  if (!error_gt_ref.empty()) {
    mean_occ_ref /= static_cast<float>(error_occ_ref.size());
    rmse_occ_ref = std::sqrt(rmse_occ_ref / static_cast<float>(error_occ_ref.size()));
  }
  float stddev_occ_ref = 0.0;
  for (auto value : error_occ_ref) {
    stddev_occ_ref += std::pow(value - mean_occ_ref, 2.0);
  }
  if (error_occ_ref.size() > 2) {
    stddev_occ_ref = sqrt(stddev_occ_ref / static_cast<float>(error_occ_ref.size() - 1));
  }

  if (request.verbosity > 1){
    ROS_INFO("ESDF mapping evaluation (ref: occupancy voxel centers):");
    ROS_INFO("MeanError    [m]: %f", mean_occ_ref);
    ROS_INFO("StdError     [m]: %f", stddev_occ_ref);
    ROS_INFO("RMSE         [m]: %f", rmse_occ_ref);

    ROS_INFO("ESDF mapping evaluation (ref: GT point cloud):");
    ROS_INFO("MeanError    [m]: %f", mean_gt_ref);
    ROS_INFO("StdError     [m]: %f", stddev_gt_ref);
    ROS_INFO("RMSE         [m]: %f", rmse_gt_ref);
  }

  std::stringstream ss;
  ss << mean_occ_ref << "," << stddev_occ_ref << "," << rmse_occ_ref << "," << trunc_dist_ << ","
     << mean_gt_ref << "," << stddev_gt_ref << "," << rmse_gt_ref << "," << trunc_dist_;

  if (request.visualize_esdf_error) {
      LOG_IF(INFO, request.verbosity >= 2) << "Publishing esdf error slice.";
      visualizer_->visualizeEsdfError(*esdf_layer);
  }

  visualizer_->visualizeTsdf(*tsdf_layer);
  visualizer_->visualizeGsdf(*tsdf_layer);
  visualizer_->visualizeEsdf(*esdf_layer);
  
  return ss.str();
}


//For multi map evaluation
bool MapEvaluator::evaluateMapCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  // Load map.
  submaps_ = std::make_shared<SubmapCollection>();
  if (!submaps_->loadFromFile(request.file_path)) {
    LOG(ERROR) << "Could not load panoptic map from '" << request.file_path
               << "'.";
    submaps_.reset();
    return false;
  }
  planning_ = std::make_unique<PlanningInterface>(submaps_);

  // Evaluate.
  output_file_ << computeReconstructionError(request_) << ","
               << computeMeshError(request_) << "\n";
  output_file_.flush();
  return true;
}

void MapEvaluator::visualizeReconstructionError(
    const EvaluationRequest& request) {
  // Coloring: grey -> unknown, green -> 0 error, red -> maximum error,
  // purple -> truncated to max error.

  constexpr int max_number_of_neighbors_factor = 25000;  // points per cubic
  // meter depending on voxel size for faster nn search.
  // buildKdTree(); //already built before

  // Remove inactive maps if include_all_submaps is false
  if (!request.is_single_tsdf) {
    std::vector<int> submaps_to_remove;
    for (const Submap& submap : *submaps_) {
      if (submap.getLabel() == PanopticLabel::kFreeSpace ||
         (submap.getChangeState() != ChangeState::kPersistent &&
         !request.include_all_submaps)) {
        submaps_to_remove.emplace_back(submap.getID());
      }
    }
    for (int id : submaps_to_remove) {
      submaps_->removeSubmap(id);
    }
  }

  // Setup progress bar.
  float counter = 0.f;
  float max_counter = 0.f;
  ProgressBar bar;
  for (auto& submap : *submaps_) {
    voxblox::BlockIndexList block_list;
    if (request.color_by_mesh_distance) { // mesh error
      if (use_voxblox_) {
        LOG(WARNING)<<"Voxblox meshing not supported now";
        return;
      }
      submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
    } else { // tsdf error
      submap.getTsdfLayer().getAllAllocatedBlocks(&block_list);
    }
    max_counter += block_list.size();
  }

  if (request.color_by_mesh_distance) { // rendered with mesh error
    for (auto& submap : *submaps_) {
      submap.updateMesh(false);
      voxblox::BlockIndexList block_list;
      submap.getMeshLayer().getAllAllocatedMeshes(&block_list);

      // for each mesh vertex
      for (auto& block_id : block_list) {
        auto& mesh = submap.getMeshLayerPtr()->getMeshByIndex(block_id);
        const size_t size = mesh.vertices.size();
        mesh.colors.resize(size);
        for (size_t i = 0; i < size; ++i) {
          const float query_pt[3] = {mesh.vertices[i].x(), mesh.vertices[i].y(),
                                     mesh.vertices[i].z()};
          size_t ret_index;
          float out_dist_sqr;
          int num_results =
              kdtree_gt_->knnSearch(&query_pt[0], 1, &ret_index, &out_dist_sqr);

          const float distance = std::sqrt(out_dist_sqr);
          
          if (distance > trunc_dist_) {
            mesh.colors[i] = Color(128, 128, 128);
            continue;
          }

          const float frac = std::min(distance, trunc_dist_) / trunc_dist_;
                             
          const float r = std::min((frac - 0.5f) * 2.f + 1.f, 1.f) * 255.f;
          float g = (1.f - frac) * 2.f * 255.f;
          if (frac <= 0.5f) {
            g = 190.f + 130.f * frac;
          }
          mesh.colors[i] = Color(r, g, 0);
        }

        mesh.updated = false;
        bar.display(++counter / max_counter);
      }
    }

    // Store colored submaps.
    submaps_->saveToFile(target_directory_ + "/" + target_map_name_ +
                         "_evaluated.panmap");
  } else { // rendered with tsdf error
    // Parse all submaps
    for (auto& submap : *submaps_) {
      const size_t num_voxels_per_block =
          std::pow(submap.getTsdfLayer().voxels_per_side(), 3);
      const float voxel_size = submap.getTsdfLayer().voxel_size();
      const float voxel_size_sqr = voxel_size * voxel_size;
      const float truncation_distance = submap.getConfig().truncation_distance;
      const int max_number_of_neighbors =
          max_number_of_neighbors_factor / std::pow(1.f / voxel_size, 2.f);
      voxblox::Interpolator<TsdfVoxel> interpolator(
          submap.getTsdfLayerPtr().get());

      // Parse all voxels.
      voxblox::BlockIndexList block_list;
      submap.getTsdfLayer().getAllAllocatedBlocks(&block_list);
      int block_count = 0;
      for (auto& block_index : block_list) {
        if (!ros::ok()) {
          return;
        }

        voxblox::Block<TsdfVoxel>& block =
            submap.getTsdfLayerPtr()->getBlockByIndex(block_index);
        for (size_t linear_index = 0; linear_index < num_voxels_per_block;
             ++linear_index) {
          TsdfVoxel& voxel = block.getVoxelByLinearIndex(linear_index);
          if (voxel.distance > truncation_distance ||
              voxel.distance < -truncation_distance) {
            continue;  // these voxels can never be surface.
          }
          Point center = block.computeCoordinatesFromLinearIndex(linear_index);

          // Find surface points within 1 voxel size.
          float query_pt[3] = {center.x(), center.y(), center.z()}; // voxel center
          std::vector<size_t> ret_index(max_number_of_neighbors);
          std::vector<float> out_dist_sqr(max_number_of_neighbors);
          // For each voxel, query the nearest point within the gt point cloud to the voxel center
          int num_results =
              kdtree_gt_->knnSearch(&query_pt[0], max_number_of_neighbors,
                                 &ret_index[0], &out_dist_sqr[0]);

          if (num_results == 0) {
            // No nearby surface.
            voxel.color = Color(128, 128, 128); //Gray
            continue;
          }

          // Get average error.
          float total_error = 0.f;
          float max_error = 0.f;
          int counted_voxels = 0;
          float min_dist_sqr = 1000.f;
          for (int i = 0; i < num_results; ++i) {
            min_dist_sqr = std::min(min_dist_sqr, out_dist_sqr[i]);
            if (out_dist_sqr[i] > voxel_size_sqr) {
              continue;
            }
            voxblox::FloatingPoint distance;
            if (interpolator.getDistance(kdtree_data_gt_.points[ret_index[i]],
                                         &distance, true)) {
              const float error = std::abs(distance);
              total_error += error;
              max_error = std::max(max_error, error);
              counted_voxels++;
            }
          }
          // Coloring.
          if (counted_voxels == 0) {
            counted_voxels = 1;
            total_error += std::sqrt(min_dist_sqr);
            max_error = min_dist_sqr;
          }
          float frac;
          if (request.color_by_max_error) {
            frac = std::min(max_error, trunc_dist_) /
                   trunc_dist_;
          } else {
            frac = std::min(total_error / counted_voxels,
                            trunc_dist_) /
                   trunc_dist_;
          }

          float r = std::min((frac - 0.5f) * 2.f + 1.f, 1.f) * 255.f;
          float g = (1.f - frac) * 2.f * 255.f;
          if (frac <= 0.5f) {
            g = 190.f + 130.f * frac;
          }
          voxel.color = voxblox::Color(r, g, 0);
        }

        // Show progress.
        counter += 1.f;
        bar.display(counter / max_counter);
      }
      submap.updateMesh(false);
    }

    // Store colored submaps.
    std::string output_name =
        target_directory_ + "/" + target_map_name_ + "_evaluated_" +
        (request.color_by_max_error ? "max" : "mean") + ".panmap";
    submaps_->saveToFile(output_name);
  }
}

// GT point cloud
void MapEvaluator::buildKdTreeGt() {
  kdtree_data_gt_.points.clear();
  kdtree_data_gt_.points.reserve(gt_ptcloud_->size());
  for (const auto& point : *gt_ptcloud_) {
    kdtree_data_gt_.points.emplace_back(point.x, point.y, point.z);
  }
  kdtree_gt_.reset(new KDTree(3, kdtree_data_gt_,
                           nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  kdtree_gt_->buildIndex();
}

// Occ voxel center point
void MapEvaluator::buildKdTreeOcc() {
  kdtree_data_occ_.points.clear();
  kdtree_data_occ_.points.reserve(occ_ptcloud_->size());
  for (const auto& point : *occ_ptcloud_) {
    kdtree_data_occ_.points.emplace_back(point.x, point.y, point.z);
  }
  kdtree_occ_.reset(new KDTree(3, kdtree_data_occ_,
                           nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  kdtree_occ_->buildIndex();
}

// Mesh vertices
void MapEvaluator::buildKdTreeMesh() {
  kdtree_data_mesh_.points.clear();
  kdtree_data_mesh_.points.reserve(mesh_ptcloud_->size());
  for (const auto& point : *mesh_ptcloud_) {
    kdtree_data_mesh_.points.emplace_back(point.x, point.y, point.z);
  }
  kdtree_mesh_.reset(new KDTree(3, kdtree_data_mesh_,
                           nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  kdtree_mesh_->buildIndex();
}

void MapEvaluator::publishVisualization() {
  // Make sure the tfs arrive otherwise the mesh will be discarded.
  visualizer_->visualizeAll(submaps_.get());
}

}  // namespace panoptic_mapping
