import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.wandb_project_name = "reef-net"
    config.data_path = "data/tensorflow-great-barrier-reef"
    config.custom_path = "metadata/great-barrier-reef-custom-splits"

    config.train_path = "train.csv"
    config.val_path = "val.csv"

    config.batch_size = 2
    config.num_classes = 1
    config.input_shape = (720, 1280, 3)


    epochs = 300
    steps_per_epoch = 1000
    validation_steps = 300

    return config
