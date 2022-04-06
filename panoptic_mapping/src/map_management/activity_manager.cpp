#include "panoptic_mapping/map_management/activity_manager.h"

#include <unordered_set>
#include <ros/ros.h>

namespace panoptic_mapping {

void ActivityManager::Config::checkParams() const {
  //  checkParamNE(error_threshold, 0.f, "error_threshold");
}

void ActivityManager::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("required_reobservations", &required_reobservations);
  setupParam("deactivate_after_missed_detections",
             &deactivate_after_missed_detections);
  setupParam("update_after_deactivation", &update_after_deactivation);
}

ActivityManager::ActivityManager(const Config& config)
    : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
}

void ActivityManager::processSubmaps(SubmapCollection* submaps) {
  CHECK_NOTNULL(submaps);

  Timer timer("umap_management/manage_submap_activity/manage");

  std::unordered_set<int> submaps_to_delete;
  for (Submap& submap : *submaps) {
    // Parse only active object maps.
    // NOTE(schmluk): Could be extended to free space for global consistency.
    if (!submap.isActive() || submap.getLabel() == PanopticLabel::kFreeSpace) {
      continue;
    }

    // Check for re-detections of new submaps. 
    // If the submap is not tracked continously for k frames, the submap would be removed
    if (!checkRequiredRedetection(&submap)) {
      submaps_to_delete.insert(submap.getID());
      if (config_.verbosity >= 3)
        ROS_INFO("Removed submap %d (%s) [redetection not enough]", submap.getID(), submap.getName().c_str());
      continue;
    }

    // Check tracking for active submaps.
    // if the active submap is not not detected for Y consecutive frames, deactivate the active submap
    checkMissedDetections(&submap);
  }

  timer.Stop();

  // Remove requested submaps.
  for (const int id : submaps_to_delete) {
    submaps->removeSubmap(id);
  }

  // Reset tracked status:
  // would be true again if the submap is tracked through the projective tracker at the next frame
  for (Submap& submap : *submaps) {
    submap.setWasTracked(false);
  }
}

bool ActivityManager::checkRequiredRedetection(Submap* submap) {
  // Check the submap was re-detected in X consecutive frames after the first allocation.
  if (config_.required_reobservations <= 0) {
    return true;
  }
  const int submap_id = submap->getID();
  auto it = submap_redetection_counts_.find(submap_id);
  if (it == submap_redetection_counts_.end()) { // not found, start record the submap
    // This is a new submap.
    submap_redetection_counts_[submap_id] = config_.required_reobservations;
    return true;
  }
  if (it->second <= 0) {
    // This submap already passed the re-detection test, the submap would be kept
    return true;
  }
  if (submap->wasTracked()) {
    // Was re-observed, decrease remaining required re-observations.
    it->second--;
    return true;
  }
  // Not detected (tracked) for X consecutive frames, this submap would be a candidate for removing.
  return false;
}

void ActivityManager::checkMissedDetections(Submap* submap) {
  // Check whether a submap was not detected for X consecutive frames.
  // If the submap is not detected for X consecutive frames, the submap 
  // would become an inactive submap
  if (config_.deactivate_after_missed_detections <= 0) {
    return;
  }
  if (submap->wasTracked()) {
    // Was tracked so reset the counter.
    submap_missed_detection_counts_.erase(submap->getID());
  } else {
    auto it = submap_missed_detection_counts_.find(submap->getID());
    if (it == submap_missed_detection_counts_.end()) {
      // First missed detection, add to counter.
      // submap_missed_detection_counts_[submap->getID()] = config_.deactivate_after_missed_detections;
      it = submap_missed_detection_counts_.insert(
          submap_missed_detection_counts_.end(),
          std::pair<int, int>(submap->getID(),
                    config_.deactivate_after_missed_detections));
    }
    it->second--;
    if (it->second <= 0) {
      submap->finishActivePeriod(config_.update_after_deactivation); // deactivate
      if (config_.verbosity > 3) {
        ROS_INFO("Deactivate submap %d (%s)", submap->getID(), submap->getName().c_str());
      }
    }
  }
}

}  // namespace panoptic_mapping
