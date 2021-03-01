from maskrcnn_benchmark.config import cfg
from utils.imgproc import get_visible_mask
from inference.coco_predict import COCODPredictor
from inference.coco_visualizer import COCOVisualizer
import cv2
import os
import numpy as np


def main():
    # load image and then run prediction
    image = cv2.imread("/pdc/logs_proto/coco/stick_db_val/images/10000.png", cv2.IMREAD_COLOR)

    # The config file
    # used from within docker container, thus the abs filepath
    config_file = "/mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_stick.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(["MODEL.WEIGHT", "/mrcnn_integrate/train_tools/tmp/model_0010000.pth"])

    # Construct the predict and visualizer
    coco_predict = COCODPredictor(cfg, min_image_size=800, confidence_threshold=0.7)
    coco_vis = COCOVisualizer(cfg, min_image_size=800)

    # Do it
    predictions_raw = coco_predict.run_on_opencv_image(image)
    print(predictions_raw.bbox)
    print(predictions_raw.mode)

    # Not interested invsiualizations
    return

    # predictions = coco_vis.visualize_prediction(image, predictions_raw)

    # Save the predicted image
    tmp_dir = 'tmp_inference/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # The name and path
    image_path = os.path.join(tmp_dir, 'prediction.png')
    cv2.imwrite(image_path, predictions)

    # The mask for the first predcition
    mask = predictions_raw.get_field('mask').numpy()
    labels = predictions_raw.get_field("labels").numpy()
    boxes = predictions_raw.bbox.numpy()
    num_obj = mask.shape[0]
    for i in range(num_obj):
        assert labels[i] == 1
        topleft_x, topleft_y = boxes[i, 0], boxes[i, 1]
        bottomright_x, bottomright_y = boxes[i, 2], boxes[i, 3]
        mask_i = mask[i, 0, :, :].astype(np.float)
        mask_path = os.path.join(tmp_dir, 'mask-%d.png' % i)
        cv2.imwrite(mask_path, get_visible_mask(mask_i))


if __name__ == '__main__':
    main()
