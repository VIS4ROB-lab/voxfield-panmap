#include "panoptic_mapping/tracking/single_tsdf_tracker.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<IDTrackerBase, SingleTSDFTracker,
                                           std::shared_ptr<Globals>>
    SingleTSDFTracker::registration_("single_tsdf");

void SingleTSDFTracker::Config::checkParams() const {
  checkParamConfig(submap);
  checkParamConfig(renderer);
}

void SingleTSDFTracker::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("submap", &submap);
  setupParam("use_lidar", &use_lidar);
  setupParam("use_range_image", &use_range_image);
  setupParam("use_detectron", &use_detectron);
  setupParam("use_instance_classification", &use_instance_classification);
  setupParam("renderer", &renderer);
}

SingleTSDFTracker::SingleTSDFTracker(const Config& config,
                                     std::shared_ptr<Globals> globals)
    : config_(config.checkValid()), IDTrackerBase(std::move(globals)),
      renderer_(config.renderer, globals_, config.use_lidar, false) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  
  if (config_.use_range_image) {
    addRequiredInput(InputData::InputType::kColorImage);
    addRequiredInput(InputData::InputType::kDepthImage);
    if (config_.submap.useClassLayer()) {
      addRequiredInput(InputData::InputType::kSegmentationImage);
    }
    if (config_.use_detectron) {
      addRequiredInput(InputData::InputType::kDetectronLabels);
    }
  }
}

void SingleTSDFTracker::processInput(SubmapCollection* submaps,
                                     InputData* input) {
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  CHECK(inputIsValid(*input));

  // Check whether the map is already allocated.
  if (!is_setup_) {
    setup(submaps);
  }

  if (config_.use_detectron) {
    parseDetectronClasses(input);
  }

  if (visualizationIsOn()) {
    // TODO(py): make it more efficient 
    Timer vis_img_timer("visualization/images");

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
    visualize(depth_vis, "depth"); // you need to change it also to CV_U8C3, now it's CV_F32C1
    
    if (input->has(InputData::InputType::kNormalImage)){
      normal_vis = renderer_.colorFloatImage(input->normalImage());
      visualize(normal_vis, "normal"); // you need to change it also to CV_U8C3, now it's CV_F32C3
    }

    depth_vis.release();
    normal_vis.release();
    vis_img_timer.Stop();
  }

}

void SingleTSDFTracker::parseDetectronClasses(InputData* input) {
  std::unordered_map<int, int> detectron_to_class_id;
  for (auto it = input->idImagePtr()->begin<int>();
       it != input->idImagePtr()->end<int>(); ++it) {
    if (*it == 0) {
      // Zero indicates unknown class / no prediction.
      continue;
    }
    auto class_it = detectron_to_class_id.find(*it);
    if (class_it == detectron_to_class_id.end()) {
      // First time we encounter this ID, write to the map.
      const int class_id = input->detectronLabels().at(*it).category_id;
      detectron_to_class_id[*it] = class_id;
      *it = class_id;
    } else {
      *it = class_it->second;
    }
  }
}

void SingleTSDFTracker::setup(SubmapCollection* submaps) {
  // Check if there is a loaded map.
  if (submaps->size() > 0) {
    Submap& map = *(submaps->begin());
    if (map.getConfig().voxel_size != config_.submap.voxel_size ||
        map.getConfig().voxels_per_side != config_.submap.voxels_per_side ||
        map.getConfig().truncation_distance !=
            config_.submap.truncation_distance ||
        map.getConfig().useClassLayer() != config_.submap.useClassLayer()) {
      LOG(WARNING)
          << "Loaded submap config does not match the specified config.";
    }
    map.setIsActive(true);
    map_id_ = map.getID();
  } else {
    // Allocate the single map.
    Submap* new_submap = submaps->createSubmap(config_.submap);
    new_submap->setLabel(PanopticLabel::kBackground);
    map_id_ = new_submap->getID();
  }
  submaps->setActiveFreeSpaceSubmapID(map_id_);
  LOG(INFO) << "freespace submap id: " << map_id_;
  is_setup_ = true;
}

}  // namespace panoptic_mapping