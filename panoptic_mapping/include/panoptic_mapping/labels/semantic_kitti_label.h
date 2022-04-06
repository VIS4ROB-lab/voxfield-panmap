#ifndef PANOPTIC_MAPPING_LABELS_SEMANTIC_KITTI_LABEL_H_
#define PANOPTIC_MAPPING_LABELS_SEMANTIC_KITTI_LABEL_H_

#include <string>

namespace panoptic_mapping {

    // NOTE(py): this function now only works for ground truth semantic KITTI like labels
    inline bool semanticKittiLabelLUT(const int &sem_label, LabelEntry &label)
    {   
        std::string label_name;
        std::string instance_size;
        bool is_background;
        bool normal_reliable;
        bool is_moving = false;

        switch(sem_label){
            case 10:
                label_name = "Car";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 11: 
                label_name = "Bicycle";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 13:
                label_name = "Bus";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 15:
                label_name = "Motorcycle";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 16:
                label_name = "Tram";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 18:
                label_name = "Truck";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 20:
                label_name = "OtherVehicle";
                instance_size = "M";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 30:
                label_name = "Person";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 31:
                label_name = "Bicyclist";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 32:
                label_name = "Motorcyclist";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 40:
                label_name = "Road";
                instance_size = "L";
                is_background = true;
                normal_reliable = true;
                is_moving = false;
                break;    
            case 44:
                label_name = "Parking";
                instance_size = "L";
                is_background = true;
                normal_reliable = true;
                is_moving = false;
                break;
            case 48:
                label_name = "Sidewalk";
                instance_size = "L";
                is_background = true;
                normal_reliable = true;
                is_moving = false;
                break;
            case 49:
                label_name = "Ground";
                instance_size = "L";
                is_background = true;
                normal_reliable = true;
                is_moving = false;
                break; 
            case 50:
                label_name = "Building";
                instance_size = "L";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 51:
                label_name = "Fence";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;   
            case 52:
                label_name = "Structure";
                instance_size = "L";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break; 
            case 60: 
                label_name = "Roadmarking";
                instance_size = "S";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 70: 
                label_name = "Vegetation";
                instance_size = "M";
                is_background = true;
                normal_reliable = false;
                is_moving = false;
                break;
            case 71: 
                label_name = "Trunck";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 72: 
                label_name = "Terrain";
                instance_size = "L";
                is_background = true;
                normal_reliable = true;
                is_moving = false;
                break;
            case 80: 
                label_name = "Pole";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 81: 
                label_name = "Traffic_sign";
                instance_size = "S";
                is_background = false;
                normal_reliable = true;
                is_moving = false;
                break;
            case 99: 
                label_name = "Other_Object";
                instance_size = "M";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
                break;
            case 252:
                label_name = "Car";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = true;
                break;
            case 253:
                label_name = "Bicycle";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = true;
                break;
            case 254:
                label_name = "Person";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = true;
                break;
            case 255:
                label_name = "Motorcycle";
                instance_size = "S";
                is_background = false;
                normal_reliable = false;
                is_moving = true;
                break;
            case 256:
                label_name = "Tram";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = true;
                break;
            case 257:
                label_name = "Bus";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = true;
                break;
            case 258:
                label_name = "Truck";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = true;
                break;
            case 259:
                label_name = "OtherVehicle";
                instance_size = "M";
                is_background = false;
                normal_reliable = true;
                is_moving = true;
                break;
            default:
                label_name = "UninitializedName";
                instance_size = "M";
                is_background = false;
                normal_reliable = false;
                is_moving = false;
        }

        label.name = label_name;
        label.size = instance_size;
        label.label = is_background ? PanopticLabel::kBackground : PanopticLabel::kInstance;
        label.normal_reliable = normal_reliable;
        label.is_moving = is_moving;
        
        return true;
    }

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_LABELS_SEMANTIC_KITTI_LABEL_H_