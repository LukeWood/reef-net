import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.wandb_project_name = "reef-net"
    config.data_path = "data/tensorflow-great-barrier-reef"
    config.custom_path = "metadata/great-barrier-reef-custom-splits"
    config.train_path = "train.csv"
    config.val_path = "val.csv"
    config.batch_size = 2
    config.num_classes = 2
    config.input_shape = (720, 1280, 3)
    return config
