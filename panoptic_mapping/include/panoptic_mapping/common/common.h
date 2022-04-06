#ifndef PANOPTIC_MAPPING_COMMON_COMMON_H_
#define PANOPTIC_MAPPING_COMMON_COMMON_H_

#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

#include <glog/logging.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox/utils/timing.h>
#include <voxblox/utils/color_maps.h>

namespace panoptic_mapping {

// Aligned Eigen containers
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;
template <typename Type>
using AlignedDeque = std::deque<Type, Eigen::aligned_allocator<Type>>;
template <typename Type>
using AlignedQueue = std::queue<Type, AlignedDeque<Type>>;
template <typename Type>
using AlignedStack = std::stack<Type, AlignedDeque<Type>>;
template <typename Type>
using AlignedList = std::list<Type, Eigen::aligned_allocator<Type>>;

// Types.
// Type used for counting voxels. This stores up to ~65k measurements so should
// never run out. If this parameter is changed double check that all
// serialization still works!
using ClassificationCount = uint16_t;

// Type definitions to work with a voxblox map.
using FloatingPoint = voxblox::FloatingPoint;

// Geometry.
using Point = voxblox::Point;
using Ray = voxblox::Ray;
using Color = voxblox::Color;
using Transformation = voxblox::Transformation;
using Pointcloud = voxblox::Pointcloud;
using GlobalIndex = voxblox::GlobalIndex;
using BlockIndex = voxblox::BlockIndex;
using VoxelIndex = voxblox::VoxelIndex;
using IndexList = voxblox::GlobalIndexList;
using LongIndexHash = voxblox::LongIndexHash;
using Colors = voxblox::Colors;
using ColorMap = voxblox::ColorMap;
using RainbowColorMap = voxblox::RainbowColorMap;


// Tsdf and class Maps.
using TsdfVoxel = voxblox::TsdfVoxel;
using TsdfBlock = voxblox::Block<voxblox::TsdfVoxel>;
using TsdfLayer = voxblox::Layer<voxblox::TsdfVoxel>;
using EsdfVoxel = voxblox::EsdfVoxel;
using EsdfBlock = voxblox::Block<voxblox::EsdfVoxel>;
using EsdfLayer = voxblox::Layer<voxblox::EsdfVoxel>;
using OccVoxel = voxblox::OccupancyVoxel;
using OccBlock = voxblox::Block<voxblox::OccupancyVoxel>;
using OccLayer = voxblox::Layer<voxblox::OccupancyVoxel>;
using MeshLayer = voxblox::MeshLayer;

struct Label;
typedef AlignedVector<Label> Labels;

// Constants used across the library.
constexpr FloatingPoint kEpsilon = 1e-6; /**< Used for coordinates. */
constexpr float kFloatEpsilon = 1e-6;    /**< Used for weights. */
constexpr int kKITTIMaxIntstance = 1000;    /**< Used for assign an unqiue panoptic label. */

// Panoptic type labels.
enum class PanopticLabel { kUnknown = 0, kInstance, kBackground, kFreeSpace };
inline std::string panopticLabelToString(const PanopticLabel& label) {
  switch (label) {
    case PanopticLabel::kUnknown:
      return "Unknown";
    case PanopticLabel::kInstance:
      return "Instance";
    case PanopticLabel::kBackground:
      return "Background";
    case PanopticLabel::kFreeSpace:
      return "FreeSpace";
  }
}

struct Label {
  Label() : sem_label(0), ins_label(0) {}
  Label(short int _sem_label, short int _ins_label) 
      : sem_label(_sem_label), ins_label(_ins_label) {}
  Label(uint32_t label) {
    full_label = label;
    sem_label = label & 0xFFFF; 
    ins_label = label >> 16; 
    // TODO(py): to do a better hashing or increase the number of 1000 here
    id_label = sem_label * kKITTIMaxIntstance + ins_label; 
    // name = semanticKittiLabelNameLUT(sem_label);
  }

  int id_label; 
  uint32_t full_label;
  short int sem_label; //int16_t
  short int ins_label; //int16_t
  //std::string name;
};

// Iso-surface-points are used to check alignment and represent the surface
// of finished submaps.
struct IsoSurfacePoint {
  IsoSurfacePoint(Point _position, FloatingPoint _weight)
      : position(std::move(_position)), weight(_weight) {}
  Point position;
  FloatingPoint weight;
};

// Change detection data stores relevant information for associating submaps.
enum class ChangeState {
  kNew = 0,
  kMatched,
  kUnobserved,
  kAbsent,
  kPersistent
};

inline std::string changeStateToString(const ChangeState& state) {
  switch (state) {
    case ChangeState::kNew:
      return "New";
    case ChangeState::kMatched:
      return "Mathced";
    case ChangeState::kPersistent:
      return "Persistent";
    case ChangeState::kAbsent:
      return "Absent";
    case ChangeState::kUnobserved:
      return "Unobserved";
    default:
      return "UnknownChangeState";
  }
}

/**
 * Frame names are abbreviated consistently (in paranthesesalternative
 * explanations):
 * S - Submap
 * M - Mission (Map / World)
 * C - Camera (Sensor)
 */

// Timing.
#define PANOPTIC_MAPPING_TIMING_ENABLED  // Unset to disable all timers.
#ifdef PANOPTIC_MAPPING_TIMING_ENABLED
using Timer = voxblox::timing::Timer;
#else
using Timer = voxblox::timing::DummyTimer;
#endif  // PANOPTIC_MAPPING_TIMING_ENABLED
using Timing = voxblox::timing::Timing;

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_COMMON_H_
