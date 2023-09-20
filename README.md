# Panmap

Panoptic mapping system for RGB-D or LiDAR data

Reference: 

- [Panoptic Multi-TSDFs](https://github.com/ethz-asl/panoptic_mapping)
- [Voxfield-Panmap](https://github.com/VIS4ROB-lab/voxfield-panmap)


## Instructions

Please follow the guide from [here](https://github.com/VIS4ROB-lab/voxfield-panmap).

By using the `xxx_mono.yaml` config files, the panoptic labels are not needed. A simple TSDF-fusion based volumetric mapping pipeline would be used.

For the mapping in Horticultural Domain, you can use:

```
source ../../devel/setup.bash

roslaunch panoptic_mapping_ros run_homa.launch
```

after correctly setting the data path, config file and the input ros topics.

## TODO