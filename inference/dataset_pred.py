from dataproc.spartan_singleobj_database import SpartanSingleObjMaskDatabaseConfig, SpartanSingleObjMaskDatabase
from maskrcnn_benchmark.config import cfg
from inference.predictor import COCODemo
import random
import cv2
import os


def build_singleobj_database() -> (SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig):
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/wei/data/pdc'
    config.scene_list_filepath = '/home/wei/Coding/mankey/config/shoe_boot_logs.txt'
    config.category_name_key = 'mug'
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def run_prediction():
    # Load the model
    config_file = '/home/wei/Coding/mrcnn/mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_shoe.yaml'

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(["MODEL.WEIGHT", "/home/wei/data/pdc/coco/output_shoe/model_0142500.pth"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    # Save the predicted image
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Iterate over the dataset
    dataset, _ = build_singleobj_database()
    test_size = 100
    for i in range(test_size):
        idx = random.randint(0, len(dataset))

        # Load the dataset
        elem = dataset[idx]
        image = elem.rgb_image
        prediction = coco_demo.run_on_opencv_image(image)

        # Save the result
        save_name = 'prediction-%d.png' % i
        save_path = os.path.join(tmp_dir, save_name)
        cv2.imwrite(save_path, prediction)


if __name__ == '__main__':
    run_prediction()
