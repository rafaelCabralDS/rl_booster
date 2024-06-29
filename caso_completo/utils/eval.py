import time

from easydict import EasyDict
from booster_env.env import BoosterEnv
from utils.episode_recorder import BoosterRecorder
from utils.utils import *
from utils.plotter import *
from stable_baselines3.common.policies import BasePolicy
from gymnasium import Env

def eval_sb3_agent(agent: BasePolicy, save_dir, env_: EasyDict | Env | str | dict, render=True, n=30, verbose=True):
    episodes = []
    best_ep = None

    if isinstance(env_, Env):
        env = env_
    elif isinstance(env_, EasyDict) or isinstance(env_, str):
        env = BoosterEnv(env_, render=render)
    elif isinstance(env_, dict):
        env = BoosterEnv(EasyDict(env_), render=render)
    else:
        raise Exception("Environment is not valid")


    for i in range(n):

        # Initialize new episode
        s, _ = env.reset()
        episodes.append(BoosterRecorder())

        while not episodes[-1].done:
            action, _ = agent.predict(s)
            s, r, done, _, obs = env.step(action)
            episodes[-1].on_step(s, action, r, done, obs)
            if render:
                time.sleep(1/60)

        episodes[-1].terminate(obs["termination_cause"])
        # Save "best" episode
        if best_ep is None or episodes[-1].total_reward > best_ep.total_reward:
            best_ep = episodes[-1]

        if verbose:
            print(f"Successfully lands: {len(where(episodes, lambda x: x.success))} | "
                  f"{len(where(episodes, lambda x: x.success)) * 100 / (i + 1)}%\r", flush=True)

    if save_dir is not None:
        eval_dir = f"{save_dir}/eval"
        best_dir = f"{eval_dir}/best"
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

        plot_episodes_rewards(eval_dir, list(map(lambda x: x.total_reward, episodes)))
        plot_terminations(eval_dir, list(map(lambda x: x.termination_reason, episodes)))
        best_ep.plot(best_dir)
        best_ep.to_csv(f"{best_dir}/episode.csv")

    categories, counts = np.unique(list(map(lambda x: x.termination_reason, episodes)), return_counts=True)

    try:
        env.close()
        success_rate = counts[list.index(categories.tolist(), 'Landed')]
    except:
        success_rate = 0

    return success_rate, np.mean(list(map(lambda x: x.total_reward, episodes))), np.std(list(map(lambda x: x.total_reward, episodes)))