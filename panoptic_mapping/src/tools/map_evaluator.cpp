#include "panoptic_mapping/tools/map_evaluator.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <pcl/io/ply_io.h>
#include <ros/ros.h>
#include <voxblox/interpolator/interpolator.h>

#include "panoptic_mapping/tools/progress_bar.h"

namespace panoptic_mapping {

//TODO(py): add evaluation for ESDF !!!

void MapEvaluator::EvaluationRequest::checkParams() const {
  checkParamGT(trunc_dist, 0.f, "trunc_dist");
  checkParamGT(inlier_thre, 0.f, "inlier_thre");
}

void MapEvaluator::EvaluationRequest::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("map_file", &map_file);
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
  buildKdTree(); //build kd-tree of the gt point cloud here
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
      buildKdTree(); //build kd-tree of the gt point cloud here
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
    output_file_ << "MeanError [m],StdError [m],RMSE [m],Coverage [%],InlierRatio [%],"
                 << "TotalPoints [#],UnknownPoints [#],ObservedPoints [#],InlierPoints [#],OutlierPoints [#],"
                 << "TruncatedPoints [#],InlierDistance [m],TruncationDistance [m],"
                 << "MeanMeshError [m],StdMeshError [m],MeshRMSE [m],MeshInlierRatio [%],"
                 << "MeshInliers [#],MeshOutliers [#]\n";

    // output_file_ << computeReconstructionError(request);

    output_file_ << computeReconstructionError(request) << ","
                 << computeMeshError(request);
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

  // Evaluate gt pcl based(# gt points within < trunc_dist)
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
      if (std::abs(distance) > request.trunc_dist) {
        truncated_points++;
        if (!request.ignore_truncated_points) { // don't ignore
          abserror.push_back(request.trunc_dist); //use the truncation dist
        }
      } 
      else { //not truncated
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
     << "," << request.inlier_thre << "," << request.trunc_dist;
  
  //TODO(py): add the evaluation metric of coverage

  if (request.verbosity > 1){
    ROS_INFO("TSDF mapping evaluation:");
    ROS_INFO("MeanError    [m]: %f", mean);
    ROS_INFO("StdError     [m]: %f", stddev);
    ROS_INFO("RMSE         [m]: %f", rmse);
    ROS_INFO("Inlier ratio [%]: %f", inlier_ratio*100.0);
    ROS_INFO("Coverage     [%]: %f", coverage*100.0);
  }

  return ss.str();
}

// TODO(py): add the ESDF evaluation function (should be almost the same as TSDF one)


// Now only work for panmap, add the interface for vblox
// The mesh reconstruction error
std::string MapEvaluator::computeMeshError(const EvaluationRequest& request) {
  
  if (use_voxblox_) {
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
    LOG(WARNING)<<"currently do not support voxblox tsdf map";
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
  std::vector<float> errors;

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
        float query_pt[3] = {point.x(), point.y(), point.z()};
        std::vector<size_t> ret_index(1);
        std::vector<float> out_dist_sqr(1);
        int num_results = kdtree_->knnSearch(&query_pt[0], 1, &ret_index[0],
                                             &out_dist_sqr[0]);

        if (num_results != 0) { //found
          const float error =
              (kdtree_data_.points[ret_index[0]] - point).norm();
          errors.emplace_back(error);
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
  float mean = 0.0;
  float rmse = 0.0;
  for (auto value : errors) {
    mean += value;
    rmse += std::pow(value, 2);
  }
  if (!errors.empty()) {
    mean /= static_cast<float>(errors.size());
    rmse = std::sqrt(rmse / static_cast<float>(errors.size()));
  }
  float stddev = 0.0;
  for (auto value : errors) {
    stddev += std::pow(value - mean, 2.0);
  }
  if (errors.size() > 2) {
    stddev = sqrt(stddev / static_cast<float>(errors.size() - 1));
  }

  float inlier_ratio = 1.0 * inliers / (inliers + outliers);

  std::stringstream ss;
  ss << mean << "," << stddev << "," << rmse << "," << inlier_ratio 
     << "," << inliers << "," << outliers;
    
  if (request.verbosity > 1){
    ROS_INFO("Mesh reconstruction evaluation:");
    ROS_INFO("MeanError    [m]: %f", mean);
    ROS_INFO("StdError     [m]: %f", stddev);
    ROS_INFO("RMSE         [m]: %f", rmse);
    ROS_INFO("Inlier ratio [%]: %f", inlier_ratio*100.0);
  }
  
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

  if (request.color_by_mesh_distance) { // mesh error
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
              kdtree_->knnSearch(&query_pt[0], 1, &ret_index, &out_dist_sqr);

          const float distance = std::sqrt(out_dist_sqr);
          
          if (distance > request.trunc_dist) {
            mesh.colors[i] = Color(128, 128, 128);
            continue;
          }

          const float frac = std::min(distance, request.trunc_dist) /
                             request.trunc_dist;
          
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
  } else { // tsdf error
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
              kdtree_->knnSearch(&query_pt[0], max_number_of_neighbors,
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
            if (interpolator.getDistance(kdtree_data_.points[ret_index[i]],
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
            frac = std::min(max_error, request.trunc_dist) /
                   request.trunc_dist;
          } else {
            frac = std::min(total_error / counted_voxels,
                            request.trunc_dist) /
                   request.trunc_dist;
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

void MapEvaluator::buildKdTree() {
  kdtree_data_.points.clear();
  kdtree_data_.points.reserve(gt_ptcloud_->size());
  for (const auto& point : *gt_ptcloud_) {
    kdtree_data_.points.emplace_back(point.x, point.y, point.z);
  }
  kdtree_.reset(new KDTree(3, kdtree_data_,
                           nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  kdtree_->buildIndex();
}

void MapEvaluator::publishVisualization() {
  // Make sure the tfs arrive otherwise the mesh will be discarded.
  visualizer_->visualizeAll(submaps_.get());
}

}  // namespace panoptic_mapping
