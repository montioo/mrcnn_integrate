
# Modifies a pretrained model to change the amount of classes it can predict.
# From: https://gist.github.com/bernhardschaefer/01905b0fe83615f79e2928a2a10b6f28

import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


keys_to_remove = [
    'module.roi_heads.box.predictor.cls_score.weight',
    'module.roi_heads.box.predictor.cls_score.bias',
    'module.roi_heads.box.predictor.bbox_pred.weight',
    'module.roi_heads.box.predictor.bbox_pred.bias',
    'module.roi_heads.mask.predictor.mask_fcn_logits.weight', # mask
    'module.roi_heads.mask.predictor.mask_fcn_logits.bias'    # mask
]

def trim_maskrcnn_benchmark_model(model_path: str, trimmed_model_path: str):
    state_dict = torch.load(model_path, map_location="cpu")

    model = state_dict['model']

    for key in keys_to_remove:
        if key in model:
            del model[key]
            print('key: {} is removed'.format(key))
        else:
            print('key: {} is not present'.format(key))

    print("Also deleting optimizer, scheduler, and iteration entries")
    del state_dict['optimizer']
    del state_dict['scheduler']
    del state_dict['iteration']

    torch.save(state_dict, trimmed_model_path)
    print(f'saved to: {trimmed_model_path}')


# usage example:
model_path = "/mrcnn_integrate/train_tools/pretrained_models/e2e_mask_rcnn_R-50-FPN_1x.pth"
trimmed_model_path = "/mrcnn_integrate/train_tools/pretrained_models/e2e_mask_rcnn_R_50_FPN_1x_trimmed.pth"
trim_maskrcnn_benchmark_model(model_path, trimmed_model_path)
