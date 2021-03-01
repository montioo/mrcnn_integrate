from dataproc.spartan_singleobj_database import SpartanSingleObjMaskDatabaseConfig, SpartanSingleObjMaskDatabase
from dataproc.abstract_db import AbstractMaskDatabase
from dataproc.patch_paste_db import PatchPasteDatabase, PatchPasteDatabaseConfig
from dataproc.coco_formatter import COCODatasetFormatter, COCODatasetFormatterConfig
from typing import List


def build_singleobj_database() -> (SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig):
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/wei/data/pdc'
    config.scene_list_filepath = '/home/wei/Code/mankey_ros/mankey/config/spartan_data/mugs_all.txt'
    config.category_name_key = 'mug'
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def make_singleobj_database(scene_list_path: str, category_name: str) -> (SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig):
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/pdc'
    config.scene_list_filepath = scene_list_path
    config.category_name_key = category_name
    config.pose_yaml_name = "stick_2_keypoint_image.yaml"
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def build_patch_database(
        database_in: List[AbstractMaskDatabase],
        nominal_size: int) -> (PatchPasteDatabase, PatchPasteDatabaseConfig):
    patch_db_config = PatchPasteDatabaseConfig()
    patch_db_config.db_input_list = database_in
    patch_db_config.nominal_size = nominal_size
    patch_db_config.max_instance = 2
    patch_db = PatchPasteDatabase(patch_db_config)
    return patch_db, patch_db_config


def build_stick_dataset_full():
    # Build the database
    raw_db, _ = make_singleobj_database("/pdc/logs_proto/stick_samples_2021-01-30_all.txt", "stick")
    patch_db, _ = build_patch_database([raw_db], 70000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = "stick_db"
    formatter_config.base_folder = "/pdc/logs_proto/coco/stick_db"
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])

def build_stick_dataset_train():
    # Build the database
    # raw_db, _ = make_singleobj_database("/pdc/logs_proto/stick_samples_2021-01-30_train.txt", "stick")
    raw_db, _ = make_singleobj_database("/pdc/logs_proto/stick_samples_2021-02-01_small_train.txt", "stick")
    patch_db, _ = build_patch_database([raw_db], 15 * 2000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = "stick_db_train"
    formatter_config.base_folder = "/pdc/logs_proto/coco/stick_db_train"
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])

def build_stick_dataset_val():
    # Build the database
    # raw_db, _ = make_singleobj_database("/pdc/logs_proto/stick_samples_2021-01-30_val.txt", "stick")
    raw_db, _ = make_singleobj_database("/pdc/logs_proto/stick_samples_2021-02-01_small_val.txt", "stick")
    patch_db, _ = build_patch_database([raw_db], 6 * 2000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = "stick_db_val"
    formatter_config.base_folder = "/pdc/logs_proto/coco/stick_db_val"
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])



if __name__ == '__main__':
    # build_stick_dataset()
    build_stick_dataset_val()
    build_stick_dataset_train()
