import attr
import os
from typing import List
import yaml
import cv2
from utils.imgproc import PixelCoord, mask2bbox
from dataproc.abstract_db import AbstractMaskDatabase, ImageWithAnnotation, AnnotationEntry


@attr.s
class SpartanSingleObjMaskDatabaseConfig:
    # ${pdc_data_root}/logs_proto/2018-10....
    pdc_data_root: str = ''

    # A list of file indicates which logs will be used for dataset.
    scene_list_filepath: str = ''

    # The name of object category
    category_name_key: str = ''

    # The name of the yaml file with pose annotation
    # Relative to the "${pdc_data_root}/logs_proto/2018-10..../processed" folder
    # Should be in ${pdc_data_root}/logs_proto/2018-10..../processed/${keypoint_yaml_name}
    pose_yaml_name: str = 'images/pose_data.yaml'

    # Simple flag
    verbose: bool = True


@attr.s
class SpartanSingleObjMaskDatabaseEntry:
    # The path to rgb
    rgb_image_path = ''

    # The path to depth image
    depth_image_path = ''

    # The path to mask image
    binary_mask_path = ''

    # The bounding box is tight
    bbox_top_left = PixelCoord()
    bbox_bottom_right = PixelCoord()


class SpartanSingleObjMaskDatabase(AbstractMaskDatabase):

    def __init__(self, config: SpartanSingleObjMaskDatabaseConfig):
        super(SpartanSingleObjMaskDatabase, self).__init__()
        self._config = config

        # Build a list of scene that will be used as the dataset
        self._scene_path_list = self._get_scene_from_config(config)

        # For each scene
        self._image_entry_list = []
        for scene_root in self._scene_path_list:
            # The info code
            if config.verbose:
                print('Processing: ', scene_root)

            # The processing code
            scene_entry = SpartanSingleObjMaskDatabase._build_scene_entry(scene_root, config.pose_yaml_name)
            for item in scene_entry:
                self._image_entry_list.append(item)

        # Simple info
        print('The number of images is %d' % len(self._image_entry_list))

    def __len__(self):
        return len(self._image_entry_list)

    @property
    def path_entry_list(self) -> List[SpartanSingleObjMaskDatabaseEntry]:
        return self._image_entry_list

    def __getitem__(self, idx: int) -> ImageWithAnnotation:
        image_path_entry = self._image_entry_list[idx]
        # The returned type
        result = ImageWithAnnotation()

        # The raw RGB image
        result.rgb_image = cv2.imread(image_path_entry.rgb_image_path, cv2.IMREAD_ANYCOLOR)

        # The annotation, there is only one object
        annotation = AnnotationEntry()
        annotation.category_name = self._config.category_name_key
        annotation.binary_mask = cv2.imread(image_path_entry.binary_mask_path, cv2.IMREAD_GRAYSCALE)
        annotation.bbox_topleft = image_path_entry.bbox_top_left
        annotation.bbox_bottomright = image_path_entry.bbox_bottom_right

        # Append to result and return
        result.annotation_list = [annotation]
        return result

    @staticmethod
    def _get_scene_from_config(config: SpartanSingleObjMaskDatabaseConfig) -> List[str]:
        assert os.path.exists(config.pdc_data_root)
        assert os.path.exists(config.scene_list_filepath)

        # Read the config file
        scene_root_list = []
        with open(config.scene_list_filepath, 'r') as config_file:
            lines = config_file.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                scene_root = os.path.join(config.pdc_data_root, line)
                if SpartanSingleObjMaskDatabase._is_scene_valid(scene_root, config.pose_yaml_name):
                    scene_root_list.append(scene_root)
                else:
                    print("invalid")

        # OK
        return scene_root_list

    @staticmethod
    def _is_scene_valid(scene_root: str, pose_yaml_name: str) -> bool:
        # The path must be valid
        if not os.path.exists(scene_root):
            return False

        # Must contains keypoint annotation
        scene_processed_root = os.path.join(scene_root, 'processed')
        keypoint_yaml_path = os.path.join(scene_processed_root, pose_yaml_name)
        print(keypoint_yaml_path)
        if not os.path.exists(keypoint_yaml_path):
            return False

        # OK
        return True

    @staticmethod
    def _build_scene_entry(scene_root: str, pose_yaml_name: str) -> List[SpartanSingleObjMaskDatabaseEntry]:
        # Get the yaml file
        scene_processed_root = os.path.join(scene_root, 'processed')
        pose_yaml_path = os.path.join(scene_processed_root, pose_yaml_name)
        assert os.path.exists(pose_yaml_path)

        # Read the yaml map
        pose_yaml_file = open(pose_yaml_path, 'r')
        pose_yaml_map = yaml.load(pose_yaml_file, Loader=yaml.FullLoader)
        pose_yaml_file.close()

        # Iterate over image
        entry_list = []
        for image_key in pose_yaml_map.keys():
            image_map = pose_yaml_map[image_key]
            image_entry = SpartanSingleObjMaskDatabase._get_image_entry(image_map, scene_root)
            if image_entry is not None:
                entry_list.append(image_entry)

        # Ok
        return entry_list

    @staticmethod
    def _get_image_entry(image_map, scene_root: str) -> SpartanSingleObjMaskDatabaseEntry:
        entry = SpartanSingleObjMaskDatabaseEntry()
        # The path for rgb image
        rgb_name = image_map['rgb_image_filename']
        rgb_path = os.path.join(scene_root, 'processed/images/' + rgb_name)
        assert os.path.exists(rgb_path)
        entry.rgb_image_path = rgb_path

        # The path for depth image
        depth_name = image_map['depth_image_filename']
        depth_path = os.path.join(scene_root, 'processed/images/' + depth_name)
        assert os.path.exists(depth_path) # Spartan must have depth image
        entry.depth_image_path = depth_path

        # The path for mask image
        mask_name = depth_name[0:6] + '_mask.png'
        mask_path = os.path.join(scene_root, 'processed/image_masks/' + mask_name)
        assert os.path.exists(mask_path)
        entry.binary_mask_path = mask_path

        # The image bounding box
        top_left = PixelCoord()
        bottom_right = PixelCoord()
        if ('bbox_top_left_xy' in image_map) and ('bbox_bottom_right_xy' in image_map):
            # Already in the image map
            top_left.x, top_left.y = image_map['bbox_top_left_xy'][0], image_map['bbox_top_left_xy'][1]
            bottom_right.x, bottom_right.y = image_map['bbox_bottom_right_xy'][0], image_map['bbox_bottom_right_xy'][1]
        else:
            # Compute from mask
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            top_left, bottom_right = mask2bbox(mask_img)
        entry.bbox_top_left = top_left
        entry.bbox_bottom_right = bottom_right

        # OK
        return entry


# The debugging method
def path_entry_sanity_check(entry):
    if len(entry.rgb_image_path) < 1 or (not os.path.exists(entry.rgb_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.depth_image_path)):
        return False

    if len(entry.rgb_image_path) >= 1 and (not os.path.exists(entry.binary_mask_path)):
        return False

    if (not entry.bbox_top_left.is_valid()) or (not entry.bbox_bottom_right.is_valid()):
        return False

    # OK
    return True


def test_spartan_singleobj_database():
    import utils.imgproc as imgproc
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/pdc'
    config.scene_list_filepath = "/pdc/logs_proto/stick_samples_2021-01-30_all.txt"
    config.category_name_key = "stick"
    config.pose_yaml_name = "stick_2_keypoint_image.yaml"
    database = SpartanSingleObjMaskDatabase(config)
    path_entry_list = database.path_entry_list
    for entry in path_entry_list:
        assert path_entry_sanity_check(entry)

    # Write the image and annotation
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Iterate over the dataset
    for i in range(len(database)):
        entry = database[i]
        # Write the rgb image
        rgb_img = entry.rgb_image
        rgb_img_path = os.path.join(tmp_dir, 'img_%d_rgb.png' % i)
        cv2.imwrite(rgb_img_path, rgb_img)

        # The mask image
        assert len(entry.annotation_list) == 1
        mask_img = entry.annotation_list[0].binary_mask
        mask_img_path = os.path.join(tmp_dir, 'img_%d_mask.png' % i)
        cv2.imwrite(mask_img_path, imgproc.get_visible_mask(mask_img))

        # the bounding box
        bbox_img = imgproc.draw_bounding_box(480, 640,
                                  entry.annotation_list[0].bbox_topleft, entry.annotation_list[0].bbox_bottomright)
        bbox_img_path = os.path.join(tmp_dir, 'img_%d_bbox.png' % i)
        cv2.imwrite(bbox_img_path, bbox_img)


if __name__ == '__main__':
    test_spartan_singleobj_database()
