from dataset_RGB import *
from dataset_loader.RGBTdatasets import *

def get_training_data(cfg, img_shape):
    return RGBT_TrainDataset(cfg, img_shape)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(cfg, img_shape):
    return RGBT_TestDataset(cfg, img_shape)
