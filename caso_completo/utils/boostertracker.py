import abc
from abc import ABC
import torch
import numpy as np
from utils.eval import eval_sb3_agent
from utils.episode_recorder import BoosterRecorder
from utils.utils import clear_folder
import pandas as pd
import os
from utils.plotter import plot_iterations, plot_loss
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Union, Optional, Dict, Any
import gymnasium as gym
from easydict import EasyDict
import optuna
from booster_env.env import BoosterEnv


class Callback(ABC):

    def __init__(self, env_config: Dict, train_config: Dict):
        # self.logger = None
        self._log_df = pd.DataFrame()

        self._max_iter = train_config["max_iter"] or np.inf

        self.env_config = env_config
        self.save_dir = train_config["save_dir"] or "."
        self._save_freq = train_config["save_freq"] or 10
        self._eval_freq = train_config["eval_freq"] or 50

        self.num_timesteps = 0
        self._rollouts = 0
        self.episodes = 0
        self.episode = BoosterRecorder()
        self.max_reward = -np.Inf
        self.mean_reward = 0
        # self.rewards_history = []  # rewards over steps (not episodes)
        self.episodes_reward = []
        self.termination_history = []

    def _on_step(self, state, reward, done, obs, actions) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: If the callback returns False, training is aborted early.
        """
        self.num_timesteps += 1
        # self.rewards_history.append(reward)
        self.episode.on_step(state, actions, reward, done, obs)

        if done:
            self.on_episode_end()
        return True

    def on_episode_start(self):
        self.episodes += 1
        self.episode = BoosterRecorder()

    def on_episode_end(self):
        self.episodes_reward.append(self.episode.total_reward)
        self.termination_history.append(self.episode.termination_reason)

        # Save best model
        if self.episode.total_reward > self.max_reward:
            best_dir = os.path.join(self.save_dir, "training_best_ep")
            clear_folder(best_dir)
            self.max_reward = self.episode.total_reward
            self.save_model(best_dir)
            self.episode.to_csv(os.path.join(best_dir, f"{self.episodes}.csv"))
            self.episode.plot(best_dir)

    def on_rollout_end(self) -> bool:
        self._rollouts += 1
        self._read_log_file()

        # Eval
        if self._rollouts % self._eval_freq == 0:
            eval_dir = os.path.join(self.save_dir, f"{self._rollouts}")
            os.makedirs(eval_dir)
            self.save_model(eval_dir)

            self.eval_model(
                agent=self.agent(),
                env_config=self.env_config,
                verbose=False,
                n=30,
                render=False,
                save_dir=eval_dir
            )

        # Plot training
        if self._rollouts % self._save_freq == 0:
            self.save_model(self.save_dir)
            plot_iterations(self.save_dir, self._log_df["returns"], self._log_df["lengths"])
            if "loss" in self._log_df.columns.values: plot_loss(self.save_dir, self._log_df["loss"])

        if self._rollouts > self._max_iter:
            self.save_model(self.save_dir)
            plot_iterations(self.save_dir, self._log_df["returns"], self._log_df["lengths"])
            plot_loss(self.save_dir, self._log_df["loss"])
            print(f"Max update iterations of {self._max_iter} reached. Training is now done")
            return False
        return True

    @abc.abstractmethod
    def agent(self):
        pass

    @abc.abstractmethod
    def eval_model(self, agent, env_config, verbose, n, render, save_dir):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass

    def _read_log_file(self):
        pass


class Sb3BoosterTracker(Callback, BaseCallback):

    def __init__(self, env_config, train_config):
        super().__init__(env_config, train_config)

    def on_step(self) -> bool:
        self.num_timesteps += 1
        obs = self.locals['infos'][0]
        r = self.locals['rewards'][0]
        # state = self.locals["obs_tensor"][0]
        done = self.locals["dones"][0]
        s = obs["state"]

        # self.rewards_history.append(reward)
        self.episode.on_step(s, obs["action"], r, done, obs)

        if done:
            self.on_episode_end()
            self.on_episode_start()

        if self._rollouts > self._max_iter:
            print(f"Max update iterations of {self._max_iter} reached. Training is now done")
            return False

        return True

    def eval_model(self, agent, env_config, verbose, n, render, save_dir):
        eval_sb3_agent(agent, save_dir, env_config, render, n, verbose)

    def save_model(self, path):
        self.model.save(os.path.join(path, "agent.zip"))

    def agent(self):
        return self.model.policy

    def _read_log_file(self):

        # Read the csv file
        path = os.path.join(self.save_dir, "progress.csv")

        try:
            self._log_df = pd.read_csv(path)
        except:  # Create a placeholder
            self._log_df = pd.DataFrame(columns=["returns", "lengths", "loss"])
            return

        self._log_df = self._log_df.rename(columns={
            "rollout/ep_rew_mean": "returns",
            "rollout/ep_len_mean": "lengths",
            "train/loss": "loss",
        })


class SuccessRateTrialCallback(BaseCallback):

    def __init__(
            self,
            eval_env: EasyDict | gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 100,
            eval_freq: int = 10000,
            verbose: int = 0,
            max_iter: int = 5000,
            # callback_after_eval: Optional[BaseCallback] = None
    ):
        super().__init__(verbose=verbose, )

        self.trial = trial
        self.eval_idx = 0
        self.eval_freq = eval_freq
        self.eval_env = eval_env if isinstance(eval_env, gym.Env) else BoosterEnv(eval_env, render=False)
        self.n_eval_episodes = n_eval_episodes
        self.success_rate = 0.0
        self._recorder = BoosterRecorder()
        self.episodes_count = 0
        self.is_pruned = False
        self.mean_rew = 0
        self.max_iter = max_iter
        self.iteration = 0
        self._last_progress = 0.0
        self.best_ep : BoosterRecorder= None

    def eval_and_report(self) -> bool:
        self.success_rate, self.mean_rew, std_rew = eval_sb3_agent(
            self.model.policy,
            None,
            env_=self.eval_env,
            render=False,
            n=self.n_eval_episodes,
            verbose=False
        )
        print("Eval mean reward is ", self.mean_rew, "and rate is", self.success_rate)


        self.trial.report(self.mean_rew, self.eval_idx)
        self.eval_idx += 1
        # self.trial.report(self.success_rate, self.eval_idx)

        # Prune trial if needed.
        #if self.trial.should_prune():
        #    return True
        return False

    def _on_step(self) -> bool:

        obs = self.locals['infos'][0]
        r = self.locals['rewards'][0]
        done = self.locals["dones"][0]
        s = obs["state"]
        a = obs["action"]

        self._recorder.on_step(s, a, r, done, obs)

        if done:
            # save best episode
            if self.best_ep is None or self._recorder.total_reward > self.best_ep.total_reward:
                self.best_ep = self._recorder

            self.episodes_count += 1
            del self._recorder
            self._recorder = BoosterRecorder()

        # Eval
        if done and self.eval_freq > 1 and self.episodes_count % self.eval_freq == 0:
            self.is_pruned = self.eval_and_report()
            if self.is_pruned:
                return False

        if self.iteration > self.max_iter:
            self.eval_and_report()
            return False

        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1

        progress = round(self.iteration / self.max_iter, 1)
        if progress % 0.1 == 0 and progress > self._last_progress:
            self._last_progress = progress
            print(f'[UPDATE] Trial {self.trial.number} progress is at {progress*100}%')
        super()._on_rollout_end()

    def _on_training_end(self) -> None:
        # Final evaluation
        # self.eval_and_report()
        super()._on_training_end()
