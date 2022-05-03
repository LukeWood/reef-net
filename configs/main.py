import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.wandb_project_name = "reef-net"
    config.data_path = "tensorflow-great-barrier-reef"
    config.batch_size = 2
    config.input_shape = (720, 1280, 3)
    config.shuffle_buffer = 64

    return config
