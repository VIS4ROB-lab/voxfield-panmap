#ifndef PANOPTIC_MAPPING_TOOLS_MAP_RENDERER_H_
#define PANOPTIC_MAPPING_TOOLS_MAP_RENDERER_H_

#include <opencv2/core/mat.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <voxblox/utils/color_maps.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/globals.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"

namespace panoptic_mapping {

/**
 * Preliminary tool to visualize the map for debugging. Not very efficient or
 * sophisticated.
 */
class MapRenderer {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    bool impaint_voxel_size = true; // time consuming but better visualization

    Config() { setConfigName("MapRenderer"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  // MapRenderer(const Config& config, const Camera::Config& camera,
  //             bool print_config = true);

  MapRenderer(const Config& config, std::shared_ptr<Globals> globals,
              bool use_lidar = true, bool print_config = true);
  

  virtual ~MapRenderer() = default;

  // Tools.
  // Approximate rendering.
  cv::Mat renderActiveSubmapIDs(const SubmapCollection& submaps,
                                const Transformation& T_M_C);
  cv::Mat renderActiveSubmapClasses(const SubmapCollection& submaps,
                                    const Transformation& T_M_C);
  // colorize images
  cv::Mat colorIdImage(const cv::Mat& id_image, int colors_per_revolution = 20);
  cv::Mat colorPanoImage(const cv::Mat& pano_id_image, const cv::Mat& sem_image, int colors_per_revolution = 20);
  cv::Mat colorPanoImageOverlaid(const cv::Mat& pano_id_image, const cv::Mat& color_image, int colors_per_revolution = 20);
  cv::Mat colorGrayImage(const cv::Mat &image, float trunc_max_value = -1.0, int color_scale = 1); //Jet colormap
  cv::Mat colorFloatImage(const cv::Mat &input_img);
  cv::Mat depthFilter(const cv::Mat &image, const cv::Mat &depth_image, float min_depth, float max_depth);

 private:
  const Config config_;

  std::shared_ptr<Globals> globals_;
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<Lidar> lidar_;

  bool use_lidar_;

  //
  Eigen::MatrixXf range_image_;
  voxblox::ExponentialOffsetIdColorMap id_color_map_;

  // Methods.
  cv::Mat render(const SubmapCollection& submaps, const Transformation& T_M_C,
                 bool only_active_submaps, int (*paint)(const Submap&));
  static int paintSubmapID(const Submap& submap);
  static int paintClass(const Submap& submap);
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_TOOLS_MAP_RENDERER_H_
