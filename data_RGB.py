from dataset_RGB import *
from dataset_loader.RGBTdatasets import *

def get_training_data(cfg):
    return RGBT_TrainDataset(cfg)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
