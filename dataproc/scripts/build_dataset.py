from dataproc.spartan_singleobj_database import SpartanSingleObjMaskDatabaseConfig, SpartanSingleObjMaskDatabase
from dataproc.abstract_db import AbstractMaskDatabase
from dataproc.patch_paste_db import PatchPasteDatabase, PatchPasteDatabaseConfig
from dataproc.coco_formatter import COCODatasetFormatter, COCODatasetFormatterConfig


def build_singleobj_database() -> (SpartanSingleObjMaskDatabase, SpartanSingleObjMaskDatabaseConfig):
    config = SpartanSingleObjMaskDatabaseConfig()
    config.pdc_data_root = '/home/wei/data/pdc'
    config.scene_list_filepath = '/home/wei/Coding/mankey/config/mugs_up_with_flat_logs.txt'
    config.category_name_key = 'mug'
    database = SpartanSingleObjMaskDatabase(config)
    return database, config


def build_patch_database(
        database_in: AbstractMaskDatabase,
        nominal_size: int) -> (PatchPasteDatabase, PatchPasteDatabaseConfig):
    patch_db_config = PatchPasteDatabaseConfig()
    patch_db_config.db_input = database_in
    patch_db_config.nominal_size = nominal_size
    patch_db_config.max_instance = 4
    patch_db = PatchPasteDatabase(patch_db_config)
    return patch_db, patch_db_config


def build_coco_dataset():
    # Build the database
    raw_db, _ = build_singleobj_database()
    patch_db, _ = build_patch_database(raw_db, 1000)

    # Build and formatter and run it
    formatter_config = COCODatasetFormatterConfig()
    formatter_config.db_name = 'mug_test'
    formatter_config.base_folder = '/home/wei/data/pdc/coco/mug_test'
    formatter = COCODatasetFormatter(formatter_config)
    formatter.process_db_list([patch_db])


if __name__ == '__main__':
    build_coco_dataset()
