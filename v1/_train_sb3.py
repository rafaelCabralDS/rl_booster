from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from easydict import EasyDict
import torch.nn as nn
from booster_env.env import BoosterEnv
from utils.boostertracker import Sb3BoosterTracker
from utils.utils import create_training_folder
from utils.yaml_utils import *
from typing import Callable
from stable_baselines3.common.utils import get_linear_fn


def _policyGenerator(config: EasyDict):
    if config.layers is None or config.nodes is None:
        return None  # fallback to the models default
    layers = config.layers
    nodes = config.nodes
    net_arch = [nodes for _ in range(layers)]

    activation_fn = config.activation or 'tanh'
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return dict(net_arch=net_arch, activation_fn=activation_fn)


def _PPO(config: EasyDict, env):
    lr_end_episodes = 1000
    end_fraction = lr_end_episodes/train_config["episodes"]

    model_ = PPO(
        config.policy,
        env,
        policy_kwargs=_policyGenerator(config),
        verbose=config.verbose,
        learning_rate=get_linear_fn(config.lr_start, config.lr_end, 0.025),  #config.lr or 3e-4, #
        gamma=config.gamma or 0.99,
        normalize_advantage=config.normalize_advantage,
        device=config.device or "auto",
        batch_size=config.batch_size or 64,
        n_epochs=config.n_epochs or 10,
        n_steps=config.buffer_size or 2048,
        use_sde=config.use_sde or False,
        # tensorboard_log=config.log_dir,
    )

    if config.model is not None:
        print("Initializing previous trained model at ", config.model)
        model_.set_parameters(os.path.join(config.model, "agent.zip"))

    return model_


if __name__ == "__main__":
    set_log_level(logging.INFO)

    env_config = load_config("./env.yaml")
    train_config = load_config("./train.yaml")

    # Override the train config save dir with the actual folder
    training_folder = create_training_folder(os.path.join(train_config["save_dir"], "training"), "sb3_ppo")
    train_config["save_dir"] = training_folder

    show_dict(env_config, depth=2)
    show_dict(train_config, depth=2)

    save_config(train_config["save_dir"], env_config, "env.yaml")
    save_config(train_config["save_dir"], train_config, "train.yaml")

    env = BoosterEnv(config=EasyDict(env_config))
    env = Monitor(env, allow_early_resets=False)

    model = _PPO(config=EasyDict(train_config), env=env)
    steps = train_config["episodes"] * env_config["max_steps"]

    new_logger = configure(train_config["save_dir"], ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=steps, callback=Sb3BoosterTracker(env_config, train_config))