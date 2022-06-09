class config_cartpole:
    def __init__(self, use_baseline, seed):
        self.env_name = "CartPole-v0"
        self.record = False
        baseline_str = "baseline" if use_baseline else "no_baseline"
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 2000  # number of steps used to compute each policy update
        self.max_ep_len = 200  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_breakout:
    def __init__(self, use_baseline, seed):
        self.env_name = "SuperMarioBros-v0"
        self.record = True
        baseline_str = "baseline" if use_baseline else "no_baseline"
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 2000  # number of steps used to compute each policy update
        self.max_ep_len = 200  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.99  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class config_mario:
    def __init__(self, use_baseline, seed):
        self.env_name = "SuperMarioBros-v0"
        self.record = True
        baseline_str = "baseline" if use_baseline else "no_baseline"
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 2000  # number of steps used to compute each policy update
        self.max_ep_len = 200  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.99  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size



class config_pendulum:
    def __init__(self, use_baseline, seed):
        self.env_name = "InvertedPendulum-v2"
        self.record = False
        baseline_str = "baseline" if use_baseline else "no_baseline"
        seed_str = "seed=" + str(seed)

        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 10000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class config_cheetah:
    def __init__(self, use_baseline, seed):
        self.env_name = "HalfCheetah-v2"
        self.record = False
        baseline_str = "baseline" if use_baseline else "no_baseline"
        seed_str = "seed=" + str(seed)

        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 50000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.9  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


def get_config(env_name, baseline, seed=15):
    if env_name == "cartpole":
        return config_cartpole(baseline, seed)
    elif env_name == "pendulum":
        return config_pendulum(baseline, seed)
    elif env_name == "cheetah":
        return config_cheetah(baseline, seed)
    elif env_name == "mario":
        return config_mario(baseline, seed)
