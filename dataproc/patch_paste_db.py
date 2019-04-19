import attr
import random
import numpy as np
from dataproc.abstract_db import AbstractMaskDatabase, ImageWithAnnotation, AnnotationEntry
from utils.imgproc import mask2bbox


@attr.s
class PatchPasteDatabaseConfig:
    # The nominal size of the paste database
    # This database can generate images randomly
    nominal_size: int = 1000

    # The input database
    # Must be specified
    db_input: AbstractMaskDatabase = None

    # The generation parameter
    min_instance: int = 1
    max_instance: int = 5


class PatchPasteDatabase(AbstractMaskDatabase):

    def __init__(self, config: PatchPasteDatabaseConfig):
        # Maintain the config and check validity
        self._config = config
        assert self._config.nominal_size > 0
        assert self._config.db_input is not None

    def __len__(self):
        return self._config.nominal_size

    def __getitem__(self, item) -> ImageWithAnnotation:
        # Get an initial sample
        current_image_rgb, current_annotation = self._get_random_image_and_annotation()
        current_image_mask: np.ndarray = current_annotation.binary_mask.copy()
        object_category_list = [current_annotation.category_name]
        object_category_foreground_val = [1]

        # The Processing loop
        num_instance = random.randint(self._config.min_instance, self._config.max_instance)
        for i in range(2, 1 + num_instance):
            # Sample another image
            sampled_rgb, sampled_annotation = self._get_random_image_and_annotation()
            object_category_list.append(sampled_annotation.category_name)

            # Merge them
            current_image_rgb, current_image_mask = PatchPasteDatabase._merge_image(
                current_image_rgb, current_image_mask,
                sampled_rgb, sampled_annotation.binary_mask,
                foreground_mask_value=i)
            object_category_foreground_val.append(i)

        # Seems ok to build the result
        result = ImageWithAnnotation()
        result.rgb_image = current_image_rgb

        # Build the annotation
        result.annotation_list = []
        assert len(object_category_list) == len(object_category_foreground_val)
        for i in range(len(object_category_list)):
            # Get the mask
            this_mask = current_image_mask.copy()
            this_mask[current_image_mask != object_category_foreground_val[i]] = 0
            this_mask[current_image_mask == object_category_foreground_val[i]] = 1

            # Build the annotation
            this_annotation = AnnotationEntry()
            this_annotation.binary_mask = this_mask
            this_annotation.category_name = object_category_list[i]
            this_annotation.bbox_topleft, this_annotation.bbox_bottomright = mask2bbox(this_mask)

            # Insert into the list
            result.annotation_list.append(this_annotation)

        # OK
        return result

    def _get_random_input_entry(self) -> ImageWithAnnotation:
        """
        Currently only for single object, but should be OK
        for multiple database (only need to modified this function)
        :return:
        """
        return self._config.db_input.get_random_entry()

    def _get_random_image_and_annotation(self) -> (np.ndarray, AnnotationEntry):
        image_entry = self._get_random_input_entry()
        image_rgb = image_entry.rgb_image.copy()
        sampled_annotation = PatchPasteDatabase._sample_from_annotation_list(image_entry)
        return image_rgb, sampled_annotation

    @staticmethod
    def _sample_from_annotation_list(img_entry: ImageWithAnnotation) -> AnnotationEntry:
        if len(img_entry.annotation_list) == 1:
            return img_entry.annotation_list[0]
        elif len(img_entry.annotation_list) > 1:
            rand_idx = random.randint(0, len(img_entry.annotation_list) - 1)
            return img_entry.annotation_list[rand_idx]
        else:
            raise RuntimeError('There is no annotation in this entry')

    @staticmethod
    def _merge_image(
            background_image: np.ndarray, background_mask: np.ndarray,
            foreground_image: np.ndarray, foreground_mask: np.ndarray,
            foreground_mask_value: int = 2) -> (np.ndarray, np.ndarray):
        # create the new foreground image and mask
        three_channel_foreground_mask = np.zeros_like(foreground_image)
        for i in range(3):
            three_channel_foreground_mask[:, :, i] = foreground_mask
        new_foreground_image = foreground_image * three_channel_foreground_mask
        new_foreground_mask = foreground_mask * foreground_mask_value

        # create the new background image and mask
        temporary_three_channel_foreground_mask = three_channel_foreground_mask.copy()
        # this is needed because the mask is no longer just binary
        # (instead it has values corresponding to different objects)
        temporary_three_channel_foreground_mask[temporary_three_channel_foreground_mask > 0] = 1
        three_channel_foreground_mask_complement = np.ones_like(
            three_channel_foreground_mask) - temporary_three_channel_foreground_mask
        new_background_image = background_image * three_channel_foreground_mask_complement
        new_background_mask = background_mask * three_channel_foreground_mask_complement[:, :, 1]

        merged_image = new_background_image + new_foreground_image
        merged_mask = new_background_mask + new_foreground_mask

        return merged_image, merged_mask


# The debugger code
def test_patch_paste_db():
    import os
    import cv2
    import utils.imgproc as imgproc
    # Constuct the spartan db
    from dataproc.spartan_singleobj_database import SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/wei/data/pdc'
    config.scene_list_filepath = '/home/wei/Coding/fill_it/config/mugs_flat.txt'
    config.category_name_key = 'mug'
    database = SpartanSingleObjMaskDatabase(config)

    # Construct the patch db
    patch_db_config = PatchPasteDatabaseConfig()
    patch_db_config.db_input = database
    patch_db_config.nominal_size = 10
    patch_db = PatchPasteDatabase(patch_db_config)

    # Get one image entry and test it
    image_entry = patch_db[0]

    # Save the result
    tmp_dir = 'tmp/patch'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # Save the rgb image
    cv2.imwrite(os.path.join(tmp_dir, 'rgb.png'), image_entry.rgb_image)
    for idx in range(len(image_entry.annotation_list)):
        annotation = image_entry.annotation_list[idx]
        print(annotation.category_name)
        mask = annotation.binary_mask
        mask_name = 'mask-%d.png' % idx
        mask_path = os.path.join(tmp_dir, mask_name)
        cv2.imwrite(mask_path, imgproc.get_visible_mask(mask))


if __name__ == '__main__':
    test_patch_paste_db()
