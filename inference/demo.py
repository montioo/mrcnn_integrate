from maskrcnn_benchmark.config import cfg
from inference.predictor import COCODemo
import cv2
import os


def main():
    config_file = '/home/wei/Coding/mrcnn/mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_mug.yaml'

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(["MODEL.WEIGHT", "/home/wei/data/pdc/coco/output/model_0003000.pth"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    # load image and then run prediction
    image = cv2.imread('/home/wei/data/mankey_pdc_data/mug/0_rgb.png', cv2.IMREAD_COLOR)
    predictions = coco_demo.run_on_opencv_image(image)

    # Save the predicted image
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # The name and path
    image_path = os.path.join(tmp_dir, 'prediction.png')
    cv2.imwrite(image_path, predictions)


if __name__ == '__main__':
    main()
