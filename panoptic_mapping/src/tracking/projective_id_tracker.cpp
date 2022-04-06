#include "panoptic_mapping/tracking/projective_id_tracker.h"

#include <future>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<IDTrackerBase, ProjectiveIDTracker,
                                           std::shared_ptr<Globals>>
    ProjectiveIDTracker::registration_("projective");

void ProjectiveIDTracker::Config::checkParams() const {
  checkParamGT(rendering_threads, 0, "rendering_threads");
  checkParamNE(depth_tolerance, 0.f, "depth_tolerance");
  checkParamGT(rendering_subsampling, 0, "rendering_subsampling");
  checkParamConfig(renderer);
}

void ProjectiveIDTracker::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("depth_tolerance", &depth_tolerance);
  setupParam("tracking_metric", &tracking_metric);
  setupParam("match_acceptance_threshold", &match_acceptance_threshold);
  setupParam("use_class_data_for_matching", &use_class_data_for_matching);
  setupParam("use_approximate_rendering", &use_approximate_rendering);
  setupParam("rendering_subsampling", &rendering_subsampling);
  setupParam("min_allocation_size", &min_allocation_size);
  setupParam("rendering_threads", &rendering_threads);
  setupParam("renderer", &renderer);
  setupParam("vis_render_image", &vis_render_image);
  setupParam("use_lidar", &use_lidar);
}

ProjectiveIDTracker::ProjectiveIDTracker(const Config& config,
                                         std::shared_ptr<Globals> globals,
                                         bool print_config)
    : IDTrackerBase(std::move(globals)),
      config_(config.checkValid()),
      renderer_(config.renderer, globals_, config.use_lidar, false) {
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();
  addRequiredInputs({InputData::InputType::kColorImage,
                     InputData::InputType::kDepthImage,
                     InputData::InputType::kSegmentationImage});
                     
}

void ProjectiveIDTracker::processInput(SubmapCollection* submaps,
                                       InputData* input) {
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  CHECK(inputIsValid(*input));
  
  Timer vis_timer("visualization/images");
  // visualize the input panoptic range image
  if (visualizationIsOn()) {
    // cv::Mat input_vis = renderer_.colorIdImage(input->idImage(), 29); //should be before the processing 
    cv::Mat input_vis = renderer_.colorPanoImage(input->idImage(), input->colorImage(), 19); //should be before the processing 
    visualize(input_vis, "input"); //input instance id
    input_vis.release();
  }
  vis_timer.Pause();

 // Render all submaps.
  Timer timer("tracking");
  auto t0 = std::chrono::high_resolution_clock::now();
  Timer com_track_timer("tracking/compute_tracking_data");
  TrackingInfoAggregator tracking_data = computeTrackingData(submaps, input);
  com_track_timer.Stop();

  // Assign the input ids to tracks or allocate new maps.
  std::unordered_map<int, int> input_to_output;
  input_to_output[-1] = -1; // for nan pixels
  
  std::stringstream info;
  int n_matched = 0;
  int n_new = 0;
  Timer alloc_timer("tracking/allocate_submaps");
  
  // allocate freespace submap as submap 0
  freespace_allocator_->allocateSubmap(submaps, input);
  alloc_timer.Pause();

  std::set<int> cur_submap_id_set;


  Timer match_timer("tracking/match_ids");
  // for each input label
  for (const int input_id : tracking_data.getInputIDs()) {
    
    match_timer.Unpause();
    
    int submap_id;
    bool matched = false;
    float value;
    bool any_overlap;
    std::stringstream logging_details;

    // ROS_INFO("Debuging/Track input_id %d", input_id);

    // Find matches.
    if (config_.use_class_data_for_matching || config_.verbosity >= 4) {
      std::vector<std::pair<int, float>> ids_values; // <id, iou>
      //calculate the metric (IOU or overlap ratio) of the submaps
      any_overlap = tracking_data.getAllMetrics(input_id, &ids_values,
                                                config_.tracking_metric);
                            
      // Check for classes if requested.
      if (config_.use_class_data_for_matching) { //default: true
        for (const auto& id_value : ids_values) {
          // These are ordered in decreasing overlap metric.
          if (id_value.second < config_.match_acceptance_threshold) { //iou of such label smaller than threshold
            // No more matches possible.
            break;
          } 
          // first judge overlapping ratio, and then judge the semantic class consistency
          if (classesMatch(
              input_id,
              submaps->getSubmap(id_value.first).getClassID())) {
            // Found the best match.
            matched = true;
            submap_id = id_value.first;
            value = id_value.second; 
            break;
          }
        }
      }
      // Not check class consistency 
      else if (any_overlap && ids_values.front().second >
                                    config_.match_acceptance_threshold) {
        // Check only for the highest match.
        matched = true;
        submap_id = ids_values.front().first;
        value = ids_values.front().second;
      }

      // Print the matching statistics for all submaps.
      if (config_.verbosity >= 4) {
        logging_details << ". Overlap: ";
        for (const auto& id_value : ids_values) { //from large to small
          logging_details << " " << id_value.first << "(" << std::fixed
                          << std::setprecision(2) << id_value.second << ")";
        }
      }
      // else {
      //   logging_details << ". No overlap found.";
      // }
    } 
    else if (tracking_data.getHighestMetric(input_id, &submap_id, &value,
                                            config_.tracking_metric)) {
      // Only consider the highest metric candidate., do not consider the class consistency (not used)
      if (value > config_.match_acceptance_threshold) {
        matched = true;
      }
    }
    match_timer.Pause();

    // Allocate new submap if necessary and store tracking info.
    alloc_timer.Unpause();
    bool allocate_new_submap = tracking_data.getNumberOfInputPixels(input_id) >=
                               config_.min_allocation_size; //number of pixel > threshold, or ignored 
                               //but if matched, we do not care about this
    
    if (matched) {
      n_matched++;
      input_to_output[input_id] = submap_id;
      cur_submap_id_set.insert(submap_id);
      submaps->getSubmapPtr(submap_id)->setWasTracked(true); //if there's a match --> tracked
    } else if (allocate_new_submap) {
      n_new++;
      Submap* new_submap = allocateSubmap(input_id, submaps, input);
      if (new_submap) {
        input_to_output[input_id] = new_submap->getID();
        cur_submap_id_set.insert(new_submap->getID());

        // NOTE(py): Only for debugging, used for semantic kitti dataset
        if (submaps->getActiveGroundSubmapID() < 0 && 
            new_submap->getClassID() == 40) // road: semantic kitti class label 40 
        {
          submaps->setActiveGroundSubmapID(new_submap->getID()); 
        }
      } else {
        input_to_output[input_id] = -1;
      }
      //ROS_INFO("Debuging/Allocate new submap");
    } else {
      // Ignore these.
      input_to_output[input_id] = -1;
    }

    // Logging.(who match who)
    if (config_.verbosity >= 3) {
      if (matched) {
        info << "\n  " << input_id << "->" << submap_id << " (" << std::fixed
             << std::setprecision(2) << value << ")";
      } else {
        if (allocate_new_submap) {
          info << "\n  " << input_id << "->" << input_to_output[input_id]
               << " [new]";
        } else {
          info << "\n  " << input_id << " [ignored]";
        }
        // if (any_overlap) {
        //   info << " (" << std::fixed << std::setprecision(2) << value << ")";
        // }
      }
      info << logging_details.str();
    }

    alloc_timer.Pause();
  }
  alloc_timer.Stop();
  match_timer.Unpause();
  // Translate the id image. (input_id -> tracked submap id)
  for (auto it = input->idImagePtr()->begin<int>();
       it != input->idImagePtr()->end<int>(); ++it) { // for each pixel
    *it = input_to_output[*it];
  }
  match_timer.Stop();
  //ROS_INFO("Debuging/Image translating from input_id to submap_id");

  std::vector<int> cur_submap_id_vec(cur_submap_id_set.begin(), cur_submap_id_set.end()); // set to vector
  cur_submap_id_vec.push_back(0); // add the free space submap
  input->setSubmapIDList(cur_submap_id_vec);

  // Finish.
  auto t1 = std::chrono::high_resolution_clock::now();
  timer.Stop();
  if (config_.verbosity >= 2) {
    LOG(INFO) << "Tracked IDs in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                     .count()
              << "ms, " << n_matched << " matched, " << n_new
              << " newly allocated." << info.str();
  }

  vis_timer.Unpause();
  // Publish Visualization if requested.
  if (visualizationIsOn()) {
    // TODO(py): make it more efficient 
    Timer vis_render_timer("visualization/images/rendered");
    if (config_.use_approximate_rendering && config_.vis_render_image) {
       rendered_vis_ = renderer_.colorIdImage(
          renderer_.renderActiveSubmapIDs(*submaps, input->T_M_C()), 30); //rendered image
    }
    vis_render_timer.Stop();

    Timer vis_track_timer("visualization/images/tracked");
    cv::Mat tracked_vis = renderer_.colorIdImage(input->idImage(), 30);

    float max_depth;
    if (config_.use_lidar) {
      max_depth = globals_->lidar()->getConfig().max_range; 
    } else {
      max_depth = globals_->camera()->getConfig().max_range; 
    }
    cv::Mat depth_vis = renderer_.colorGrayImage(input->depthImage(), max_depth);
    cv::Mat normal_vis;
    
    // visualize image 
    visualize(input->colorImage(), "color"); //original color
    visualize(tracked_vis, "tracked"); //tracked submap id
    visualize(depth_vis, "depth"); // you need to change it also to CV_U8C3, now it's CV_F32C1
    
    if (input->has(InputData::InputType::kNormalImage)){
      normal_vis = renderer_.colorFloatImage(input->normalImage());
      visualize(normal_vis, "normal"); // you need to change it also to CV_U8C3, now it's CV_F32C3
    }

    if(config_.vis_render_image)
      visualize(rendered_vis_, "rendered");  

    tracked_vis.release();
    depth_vis.release();
    normal_vis.release();

    vis_track_timer.Stop();
  }
  vis_timer.Stop();
}

Submap* ProjectiveIDTracker::allocateSubmap(int input_id,
                                            SubmapCollection* submaps,
                                            InputData* input) {
  LabelEntry label;
  if (globals_->labelHandler()->segmentationIdExists(input_id)) {
    label = globals_->labelHandler()->getLabelEntry(input_id);
  }
  return submap_allocator_->allocateSubmap(submaps, input, input_id, label);
}

bool ProjectiveIDTracker::classesMatch(int input_id, int submap_class_id) {
  if (!globals_->labelHandler()->segmentationIdExists(input_id)) {
    // Unknown ID.
    return false;
  }
  return globals_->labelHandler()->getClassID(input_id) == submap_class_id;
}

// important
TrackingInfoAggregator ProjectiveIDTracker::computeTrackingData(
    SubmapCollection* submaps, InputData* input) {
  // Render each active submap in parallel to collect overlap statistics.

  // Find current submaps that are in the FOV
  std::vector<int> submap_ids;
  if (config_.use_lidar) {
    submap_ids = globals_->lidar()->findVisibleSubmapIDs(*submaps, input->T_M_C()); //also include inactive submap
  } else {
    submap_ids = globals_->camera()->findVisibleSubmapIDs(*submaps, input->T_M_C());
  }

  if (config_.verbosity > 4)
    ROS_INFO("found %d visible submaps", submap_ids.size());

  SubmapIndexGetter index_getter(submap_ids);
      
  std::vector<std::future<std::vector<TrackingInfo>>> threads;
  TrackingInfoAggregator tracking_data;
  // In parallel processing
  for (int i = 0; i < config_.rendering_threads; ++i) {
    threads.emplace_back(std::async(
        std::launch::async,
        [this, i, &tracking_data, &index_getter, submaps,
         input]() -> std::vector<TrackingInfo> {
          // Also process the input image. 
          if (i == 0) {
            if (config_.use_lidar){
              tracking_data.insertInputImage(
                input->idImage(), input->depthImage(), //what actually needed is the idImage and depthImage
                globals_->lidar()->getConfig(), config_.rendering_subsampling);
            }
            else{
              tracking_data.insertInputImage(
                input->idImage(), input->depthImage(), 
                globals_->camera()->getConfig(), config_.rendering_subsampling);
            }
          }
          std::vector<TrackingInfo> result;
          int index;
          // render the exsited submaps
          while (index_getter.getNextIndex(&index)) {
            if (config_.use_approximate_rendering) { //default: true
              if(config_.use_lidar){
                result.emplace_back(this->renderTrackingInfoApproximateLidar(
                    submaps->getSubmap(index), *input));
              }
              else{
                result.emplace_back(this->renderTrackingInfoApproximateCamera(
                    submaps->getSubmap(index), *input));
              }
            } else {
              result.emplace_back(this->renderTrackingInfoVertices(
                  submaps->getSubmap(index), *input));
            }
          }
          return result;
        }));
  }

  // Join all threads.
  std::vector<TrackingInfo> infos;
  for (auto& thread : threads) {
    for (const TrackingInfo& info : thread.get()) {
      infos.emplace_back(std::move(info));
    }
  }
  tracking_data.insertTrackingInfos(infos);
  
  if (config_.verbosity > 4)
    ROS_INFO("tracking info count: %d", infos.size());

  // Render the data if required.
  if (visualizationIsOn() && !config_.use_approximate_rendering) {
    Timer timer("visualization/tracking/rendered");
    cv::Mat vis;
    if (config_.use_lidar){
      vis = cv::Mat::ones(globals_->lidar()->getConfig().height,
                          globals_->lidar()->getConfig().width, CV_32SC1) * -1;   
    }
    else{
      vis = cv::Mat::ones(globals_->camera()->getConfig().height,
                          globals_->camera()->getConfig().width, CV_32SC1) * -1; 
    }   
    for (const TrackingInfo& info : infos) {
      for (const Eigen::Vector2i& point : info.getPoints()) {
        vis.at<int>(point.y(), point.x()) = info.getSubmapID();
      }
    }
    rendered_vis_ = renderer_.colorIdImage(vis);
  }
  return tracking_data;
}

//render the already built submap in current view
TrackingInfo ProjectiveIDTracker::renderTrackingInfoApproximateCamera(
    const Submap& submap, const InputData& input) const {
  // Approximate rendering by projecting the surface points of the submap into
  // the camera and fill in a patch of the size a voxel has (since there is 1
  // vertex per voxel).

  // Setup.
  const Camera& camera = *globals_->camera();
  TrackingInfo result(submap.getID(), camera.getConfig());
  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  const float size_factor_x =
      camera.getConfig().fx * submap.getTsdfLayer().voxel_size() / 2.f;
  const float size_factor_y =
      camera.getConfig().fy * submap.getTsdfLayer().voxel_size() / 2.f;
  const float block_size = submap.getTsdfLayer().block_size();
  const FloatingPoint block_diag_half = std::sqrt(3.0f) * block_size / 2.0f;
  const float depth_tolerance =
      config_.depth_tolerance > 0
          ? config_.depth_tolerance
          : -config_.depth_tolerance * submap.getTsdfLayer().voxel_size();
  const cv::Mat& depth_image = input.depthImage();

  // Parse all blocks.
  voxblox::BlockIndexList index_list;
  submap.getMeshLayer().getAllAllocatedMeshes(&index_list);
  for (const voxblox::BlockIndex& index : index_list) {
    if (!camera.blockIsInViewFrustum(submap, index, T_C_S, block_size,
                                     block_diag_half)) {
      continue;
    }
    for (const Point& vertex :
         submap.getMeshLayer().getMeshByIndex(index).vertices) {
      // Project vertex and check depth value.
      const Point p_C = T_C_S * vertex;
      int u, v;
      if (!camera.projectPointToImagePlane(p_C, &u, &v)) {
        continue;
      }
      if (std::abs(depth_image.at<float>(v, u) - p_C.z()) >= depth_tolerance) {
        continue;
      }

      // Compensate for vertex sparsity.
      const int size_x = std::ceil(size_factor_x / p_C.z());
      const int size_y = std::ceil(size_factor_y / p_C.z());
      result.insertRenderedPoint(u, v, size_x, size_y);
    }
  }
  result.evaluate(input.idImage(), depth_image);
  return result;
}

TrackingInfo ProjectiveIDTracker::renderTrackingInfoApproximateLidar(
    const Submap& submap, const InputData& input) const {
  // Approximate rendering by projecting the surface points of the submap into
  // the camera and fill in a patch of the size that a voxel has (since there is 1
  // vertex per voxel). 
  // Render only those mesh vertex points

  // Setup.
  const Lidar& lidar = *globals_->lidar();

  TrackingInfo result(submap.getID(), lidar.getConfig());

  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  
  const float size_factor =
      submap.getTsdfLayer().voxel_size() / 2.f;

  const float block_size = submap.getTsdfLayer().block_size();
  
  const FloatingPoint block_diag_half = std::sqrt(3.0f) * block_size / 2.0f;
  const float depth_tolerance = //depth difference threshold for the input image and the rendered image
      config_.depth_tolerance > 0
          ? config_.depth_tolerance
          : -config_.depth_tolerance * submap.getTsdfLayer().voxel_size(); //here, voxel_size
  const cv::Mat& depth_image = input.depthImage();

  // Parse all blocks.
  voxblox::BlockIndexList index_list;
  submap.getMeshLayer().getAllAllocatedMeshes(&index_list);
  //ROS_INFO("submap_id: %d, #render_block: %d", submap.getID(), index_list.size());

  //each block have lots of vertices
  for (const voxblox::BlockIndex& index : index_list) {
    if (!lidar.blockIsInViewFrustum(submap, index, T_C_S, block_size,
                                    block_diag_half)) { 
      continue;
    }
    for (const Point& vertex :
         submap.getMeshLayer().getMeshByIndex(index).vertices) {
      // Project vertex and check depth value.
      const Point p_C = T_C_S * vertex;
      float p_C_depth = p_C.norm();
      //ROS_INFO("render point: (%f, %f, %f)", p_C(0), p_C(1), p_C(2));
      
      int u, v;
      if (lidar.projectPointToImagePlane(p_C, &u, &v) < 0.0) { //failed
        continue;
      }
      float depth = depth_image.at<float>(v, u);
      if (depth < 0.0 || 
          std::abs(depth - p_C_depth) >= depth_tolerance) {
        continue; //inconsistent depth 
      }

      float ang_unit_rad = size_factor / depth;

      // Compensate for vertex sparsity. (how large should these be)
      const int size_x = std::ceil(ang_unit_rad / (2.0 * M_PI) * lidar.getConfig().width);
      const int size_y = std::ceil(ang_unit_rad / (lidar.getConfig().fov_rad) * lidar.getConfig().height);
      //ROS_INFO("size_x: %d, size_y: %d", size_x, size_y);

      result.insertRenderedPoint(u, v, size_x, size_y);
    }
  }
  // calculate
  result.evaluate(input.idImage(), depth_image); 
  return result;
}

//TODO(py): also add two options for camera and lidar separately for this function
TrackingInfo ProjectiveIDTracker::renderTrackingInfoVertices(
    const Submap& submap, const InputData& input) const {
  TrackingInfo result(submap.getID());

  // Compute the maximum extent to lookup vertices.
  const Transformation T_C_S = input.T_M_C().inverse() * submap.getT_M_S();
  const Point origin_C = T_C_S * submap.getBoundingVolume().getCenter();
  std::vector<size_t> limits(4);  // x_min, x_max, y_min, y_max
  const Camera::Config& cam_config = globals_->camera()->getConfig();

  // NOTE(schmluk): Currently just iterate over the whole frame since the sphere
  // tangent computation was not robustly implemented.
  size_t subsampling_factor = 1;
  limits = {0u, static_cast<size_t>(cam_config.width), 0u,
            static_cast<size_t>(cam_config.height)};
  const Transformation T_S_C = T_C_S.inverse();
  const TsdfLayer& tsdf_layer = submap.getTsdfLayer();
  const float depth_tolerance =
      config_.depth_tolerance > 0
          ? config_.depth_tolerance
          : -config_.depth_tolerance * submap.getTsdfLayer().voxel_size();
  for (size_t u = limits[0]; u < limits[1];
       u += config_.rendering_subsampling) {
    for (size_t v = limits[2]; v < limits[3];
         v += config_.rendering_subsampling) {
      const float depth = input.depthImage().at<float>(v, u);
      if (depth < cam_config.min_range || depth > cam_config.max_range) {
        continue;
      }
      const cv::Vec3f& vertex = input.vertexMap().at<cv::Vec3f>(v, u);
      const Point P_S = T_S_C * Point(vertex[0], vertex[1], vertex[2]);
      const voxblox::BlockIndex block_index =
          tsdf_layer.computeBlockIndexFromCoordinates(P_S);
      const auto block = tsdf_layer.getBlockPtrByIndex(block_index);
      if (block) {
        const size_t voxel_index =
            block->computeLinearIndexFromCoordinates(P_S);
        const TsdfVoxel& voxel = block->getVoxelByLinearIndex(voxel_index);
        bool classes_match = true;
        if (submap.hasClassLayer()) {
          classes_match = submap.getClassLayer()
                              .getBlockConstPtrByIndex(block_index)
                              ->getVoxelByLinearIndex(voxel_index)
                              .belongsToSubmap();
        }
        if (voxel.weight > 1e-6 && std::abs(voxel.distance) < depth_tolerance) {
          result.insertVertexPoint(input.idImage().at<int>(v, u));
          if (visualizationIsOn()) {
            result.insertVertexVisualizationPoint(u, v);
          }
        }
      }
    }
  }
  return result;
}

}  // namespace panoptic_mapping
