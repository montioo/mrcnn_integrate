Fork of [weigao95/mrcnn_integrate](https://github.com/weigao95/mrcnn_integrate)

# mrcnn_integrate

Contents of upstream `README.md` below.

## Notes:

### Installation and Dependencies

This project uses the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) project developed by facebook research which is not maintained anymore. [Here](https://github.com/montioo/maskrcnn-benchmark) a fork of maskrcnn-benchmark which updates some things to keep this repository usable.

To run this project, **use the Dockerfile** in the linked maskrcnn-benchmark fork. Use it to train the network and make predictions. An embarrassingly simple server is scripted in `inference/prediction_server.py`. Send images to this server running in the docker container and receive predictions.

### Using a Pretrained Model

A script was added to the `train_tools` folder which helps with using existing models. Once you downloaded a pretrained model, this script helps with changing the model's architecture to make it suitable for the number of classes you want it to predict.

### Create Dataset

Will convert the dataset structure from the one used with pytorch-dense-correspondence to coco data set format.

```bash
# add this module to the pythonpath
export PYTHONPATH=`pwd`":${PYTHONPATH}"

cd dataproc/scripts

# builds the dataset. See config file to adjust dataset generation.
python3 build_dataset.py
```

---

## Upstream `README.md`

This repo is part of [kPAM](https://github.com/weigao95/kPAM) that provides the data generator and training script for the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repo don't need the `ros` runtime.

To use this repo. please first follow the instruction [here](https://github.com/weigao95/mankey-ros) to setup the dataset. After setup, you can run `mrcnn_integrate/dataproc/scripts/build_dataset.py` to generate a coco dataset for maskrcnn training. You need to change the config in the `build_singleobj_database` function, which is the same as the `consruct_datset` function described [here](https://github.com/weigao95/mankey-ros).
