# mrcnn_integrate

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides the data generator and training script for the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repo don't need the `ros` runtime.

To use this repo. please first follow the instruction [here](https://github.com/weigao95/mankey-ros) to setup the dataset. After setup, you can run `mrcnn_integrate/dataproc/scripts/build_dataset.py` to generate a coco dataset for maskrcnn training. You need to change the config in the `build_singleobj_database` function, which is the same as the `consruct_datset` function described [here](https://github.com/weigao95/mankey-ros).