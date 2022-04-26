import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.data_path = "tensorflow-great-barrier-reef"
    config.batch_size = 33333s
    return config
