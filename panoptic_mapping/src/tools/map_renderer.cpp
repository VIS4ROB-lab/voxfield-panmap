#include "panoptic_mapping/tools/map_renderer.h"

#include <fstream>
#include <string>

namespace panoptic_mapping {

void MapRenderer::Config::checkParams() const {}

void MapRenderer::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("impaint_voxel_size", &impaint_voxel_size);
}

MapRenderer::MapRenderer(const Config& config, std::shared_ptr<Globals> globals,
                         bool use_lidar, bool print_config) 
    : config_(config.checkValid()), globals_(globals), use_lidar_(use_lidar) {
  
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();

  camera_ = globals->camera();
  lidar_ = globals->lidar();
  
  // Allocate range image.
  if (use_lidar)
    range_image_ = Eigen::MatrixXf(lidar_->getConfig().height, lidar_->getConfig().width);
  else 
    range_image_ = Eigen::MatrixXf(camera_->getConfig().height, camera_->getConfig().width);
}

cv::Mat MapRenderer::render(const SubmapCollection& submaps,
                            const Transformation& T_M_C,
                            bool only_active_submaps,
                            int (*paint)(const Submap&)) {
  // Use the mesh vertices as an approximation to render active submaps.
  // Assumes that all active submap meshes are up to date and does not perform
  // a meshing step of its own. Very inefficient due to pixel duplicates.
  range_image_.setOnes();
  
  cv::Mat result;
  if (use_lidar_)
  {
    range_image_ *= lidar_->getConfig().max_range;
    result = cv::Mat::ones(lidar_->getConfig().height,
                           lidar_->getConfig().width, CV_32SC1) * -1;
  }
  else {
    range_image_ *= camera_->getConfig().max_range;
    result = cv::Mat::ones(camera_->getConfig().height,
                           camera_->getConfig().width, CV_32SC1) * -1;
  }           

  // Parse all submaps.
  for (const Submap& submap : submaps) {
    // Filter out submaps.
    if (!submap.isActive() && only_active_submaps) {
      continue;
    }
    if (submap.getLabel() == PanopticLabel::kFreeSpace) {
      continue;
    }
    if (use_lidar_) {
      if (!lidar_->submapIsInViewFrustum(submap, T_M_C)) {
       continue;
      }
    } else {
      if (!camera_->submapIsInViewFrustum(submap, T_M_C)) {
       continue;
      }
    }

    // Project all surface points.
    const Transformation T_C_S = T_M_C.inverse() * submap.getT_M_S();

    float size_factor_x, size_factor_y;
    float size_factor = submap.getTsdfLayer().voxel_size() / 2.f;

    if (!use_lidar_) {
      size_factor_x = camera_->getConfig().fx * size_factor;
      size_factor_y = camera_->getConfig().fy * size_factor;
    }

    voxblox::BlockIndexList index_list;
    submap.getMeshLayer().getAllAllocatedMeshes(&index_list);
    for (const voxblox::BlockIndex& index : index_list) {
      for (const Point& vertex :
           submap.getMeshLayer().getMeshByIndex(index).vertices) {
        const Point p_C = T_C_S * vertex;
        int u, v;
        
        if (use_lidar_) { 
          // Lidar input
          if (lidar_->projectPointToImagePlane(p_C, &u, &v) > 0.0) { //success
            const float range = p_C.norm();
           
            if (config_.impaint_voxel_size) { 
              
              const float ang_unit_rad = size_factor / range;
              
              // Compensate for vertex sparsity.
              const int size_x = ang_unit_rad / (2.0 * M_PI) * lidar_->getConfig().width;
              const int size_y = ang_unit_rad / (lidar_->getConfig().fov_rad) * lidar_->getConfig().height;
    
              for (int dx = -size_x; dx <= size_x; ++dx) {
                // 360 degree horizontal angle
                int u_new = u + dx;
                if (u_new >= lidar_->getConfig().width)
                  u_new -= lidar_->getConfig().width;
                else if (u_new < 0)
                  u_new += lidar_->getConfig().width;
      
                for (int dy = -size_y; dy <= size_y; ++dy) {
                  // limited vertical angle
                  const int v_new = v + dy;
                  if (v_new < 0 || v_new >= lidar_->getConfig().height) {
                    continue;
                  }
                  if (range < range_image_(v_new, u_new)) { //find the nearest one
                    range_image_(v_new, u_new) = range;
                    result.at<int>(v_new, u_new) = (*paint)(submap);
                  }
                }
              }
            } 
            else {
              if (range < range_image_(v, u)) {
                range_image_(v, u) = range;
                result.at<int>(v, u) = (*paint)(submap);
              }
            }
          }
        }
        else {
          // Camera input
          if (camera_->projectPointToImagePlane(p_C, &u, &v)) {
            const float range = p_C.norm();

            if (config_.impaint_voxel_size) {
              // Compensate for vertex sparsity.
              const int size_x = std::ceil(size_factor_x / p_C.z());
              const int size_y = std::ceil(size_factor_y / p_C.z());
              for (int dx = -size_x; dx <= size_x; ++dx) {
                const int u_new = u + dx;
                if (u_new < 0 || u_new >= camera_->getConfig().width) {
                  continue;
                }
                for (int dy = -size_y; dy <= size_y; ++dy) {
                  const int v_new = v + dy;
                  if (v_new < 0 || v_new >= camera_->getConfig().height) {
                    continue;
                  }
                  if (range < range_image_(v_new, u_new)) {
                    range_image_(v_new, u_new) = range;
                    result.at<int>(v_new, u_new) = (*paint)(submap);
                  }
                }
              }
            } 
            else {
              if (range < range_image_(v, u)) {
                range_image_(v, u) = range;
                result.at<int>(v, u) = (*paint)(submap);
              }
            }
          }
        }
      }
    }
  }
  return result;
}

int MapRenderer::paintSubmapID(const Submap& submap) { return submap.getID(); }

int MapRenderer::paintClass(const Submap& submap) {
  return submap.getClassID();
}

cv::Mat MapRenderer::renderActiveSubmapIDs(const SubmapCollection& submaps,
                                           const Transformation& T_M_C) {
  return render(submaps, T_M_C, true, paintSubmapID);
}

cv::Mat MapRenderer::renderActiveSubmapClasses(const SubmapCollection& submaps,
                                               const Transformation& T_M_C) {
  return render(submaps, T_M_C, true, paintClass);
}

// TODO(py): a kind of dirty and ugly fix for visualization the standard panoptic image
cv::Mat MapRenderer::colorPanoImage(const cv::Mat& pano_id_image, 
                                    const cv::Mat& sem_image,
                                    int colors_per_revolution) {
  // Take an id_image (int) and render each ID to color using the exponential
  // color wheel for better visualization.
  cv::Mat result(pano_id_image.rows, pano_id_image.cols, CV_8UC3);
  if (pano_id_image.type() != CV_32SC1) {
    LOG(WARNING) << "Input 'id_image' is not of type 'CV_32SC1', skipping.";
    return result;
  }

  id_color_map_.setItemsPerRevolution(colors_per_revolution);
  for (int u = 0; u < result.cols; ++u) {
    for (int v = 0; v < result.rows; ++v) {
      int id = pano_id_image.at<int>(v, u);
      if (id < 0) {
        result.at<cv::Vec3b>(v, u) = cv::Vec3b{0, 0, 0};
      } else if (id > kKITTIMaxIntstance * 9 && 
                 id % kKITTIMaxIntstance == 0) { // Dirty fix: KITTI instance object label begin from 10 
        result.at<cv::Vec3b>(v, u) = sem_image.at<cv::Vec3b>(v, u);
      } else {
        const voxblox::Color color = id_color_map_.colorLookup(id);
        result.at<cv::Vec3b>(v, u) = cv::Vec3b{color.b, color.g, color.r};
      }
    }
  }
  return result;
}

cv::Mat MapRenderer::colorIdImage(const cv::Mat& id_image,
                                  int colors_per_revolution) {
  // Take an id_image (int) and render each ID to color using the exponential
  // color wheel for better visualization.
  cv::Mat result(id_image.rows, id_image.cols, CV_8UC3);
  if (id_image.type() != CV_32SC1) {
    LOG(WARNING) << "Input 'id_image' is not of type 'CV_32SC1', skipping.";
    return result;
  }

  id_color_map_.setItemsPerRevolution(colors_per_revolution);
  for (int u = 0; u < result.cols; ++u) {
    for (int v = 0; v < result.rows; ++v) {
      int id = id_image.at<int>(v, u);
      if (id < 0) {
        result.at<cv::Vec3b>(v, u) = cv::Vec3b{0, 0, 0};
      } else {
        const voxblox::Color color = id_color_map_.colorLookup(id);
        result.at<cv::Vec3b>(v, u) = cv::Vec3b{color.b, color.g, color.r};
      }
    }
  }
  return result;
}

cv::Mat MapRenderer::colorFloatImage(const cv::Mat &input_img)
{
  cv::Mat result;
	// Normalize the image to 0-255 
	cv::normalize(cv::abs(input_img), result, 255, 0, cv::NORM_MINMAX);
  result.convertTo(result, CV_8UC3);
  return result;
}

cv::Mat MapRenderer::colorGrayImage(const cv::Mat &image, 
                                    float max_range, int color_scale){
    
    CV_Assert(image.channels() == 1); // single channel gray input image
    
    if (max_range > 0)
      cv::threshold(image, image, max_range, -1, cv::THRESH_TRUNC);

    cv::Mat result;
    
    // Normalize the grayscale image to 0-255
	  // cv::normalize(image, result, 255, 0, cv::NORM_MINMAX);
    result = 255.0 / max_range * image;
	  result.convertTo(result, CV_8UC1);
    
    if (color_scale == 1) // COLORMAP_JET
        cv::applyColorMap(result, result, cv::COLORMAP_JET);
    else if (color_scale == 2) //COLORMAP_AUTUMN
        cv::applyColorMap(result, result, cv::COLORMAP_AUTUMN);
    else if (color_scale == 3) //COLORMAP_HOT
        cv::applyColorMap(result, result, cv::COLORMAP_HOT);
    else //default: gray
        return result;
    
    return result;
}

// /**
//  * @brief colorGrayImage       Use the color bar grayscale image to color (red to blue) 1: red, param1: yellow, param2: green, 0: blue)
//  * @param phase                Gray image input, channel 1
//  * @param param1               Color bar parameter 1
//  * @param param2               Color bar parameter 2
//  * @return                     Colored image
//  */
// cv::Mat MapRenderer::colorGrayImage(const cv::Mat &phase, 
//                                          float param1, float param2)
// {
// 	CV_Assert(phase.channels() == 1);

//     // Color bar parameter 1 must be greater than color bar parameter 2
// 	if (param2 >= param1)
// 	{
// 		return cv::Mat::zeros(10, 10, CV_8UC1);
// 	}

// 	cv::Mat temp, result, mask;
// 	// Normalize the grayscale image to 0-255
// 	cv::normalize(phase, temp, 255, 0, cv::NORM_MINMAX);
// 	temp.convertTo(temp, CV_8UC1);
// 	// The mask is created to isolate the interference of nan value
// 	mask = cv::Mat::zeros(phase.size(), CV_8UC1);
// 	mask.setTo(255, phase == phase);

// 	// Initialize three channel color map
// 	cv::Mat color1, color2, color3;
// 	color1 = cv::Mat::zeros(temp.size(), temp.type());
// 	color2 = cv::Mat::zeros(temp.size(), temp.type());
// 	color3 = cv::Mat::zeros(temp.size(), temp.type());
// 	int row = phase.rows;
// 	int col = phase.cols;

// 	// Based on the gray level of the gray image, color it. The lowest gray value 0 is blue (255, 0,0), the highest gray value 255 is red (0,0,255), and the middle gray value 127 is green (0255,0)
// 	// Don't be surprised why blue is (255,0,0), because OpenCV use BGR instead of RGB by default
// 	for (int i = 0; i < row; ++i)
// 	{
// 		uchar *c1 = color1.ptr<uchar>(i);
// 		uchar *c2 = color2.ptr<uchar>(i);
// 		uchar *c3 = color3.ptr<uchar>(i);
// 		uchar *r = temp.ptr<uchar>(i);
// 		uchar *m = mask.ptr<uchar>(i);
// 		for (int j = 0; j < col; ++j)
// 		{
// 			if (m[j] == 255)
// 			{
// 				if (r[j] > (param1 * 255) && r[j] <= 255) //long dist -> red
// 				{
// 					c1[j] = 255; //r
// 					c2[j] = uchar((1 / (1 - param1)) * (255 - r[j])); //g
// 					c3[j] = 0;
// 				}
// 				else if (r[j] <= (param1 * 255) && r[j] > (param2 * 255)) //from red to blue
// 				{
// 					c1[j] = uchar((1 / (param1 - param2)) * r[j] - (param2 / (param1 - param2)) * 255);
// 					c2[j] = 255;
// 					c3[j] = 0;
// 				}
// 				else if (r[j] <= (param2 * 255) && r[j] >= 0) //short dist -> blue
// 				{
// 					c1[j] = 0;
// 					c2[j] = uchar((1 / param2) * r[j]);  //g
// 					c3[j] = uchar(255 - (1 / param2) * r[j]); //b
// 				}
// 				else {
// 					c1[j] = 0;
// 					c2[j] = 0;
// 					c3[j] = 0;
// 				}
// 			}
// 		}
// 	}

// 	// The three channels are combined to obtain the color map
// 	std::vector<cv::Mat> images;
// 	images.push_back(color3); //b
// 	images.push_back(color2); //g
// 	images.push_back(color1); //r
// 	cv::merge(images, result);

// 	return result;
// }

}  // namespace panoptic_mapping
