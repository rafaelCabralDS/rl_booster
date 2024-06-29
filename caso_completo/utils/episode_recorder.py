from utils.plotter import *
import pandas as pd
from booster_env.constants import *

class BoosterRecorder:

    def __init__(self):
        self._actions_history = []
        self._state_history = []
        self.episode_rewards = []
        self.episode_steps = 0
        self.done = False
        self.obs_history = []
        self.termination_reason = None

    def __getitem__(self, i):
        return self.to_frame().iloc[i]

    def on_step(self, state, action, reward, done, obs):
        # assert self.done == False, "on_step called after the episode is done"
        self.episode_steps += 1
        self._state_history.append(state)
        self._actions_history.append(action)
        self.episode_rewards.append(reward)
        self.obs_history.append(obs)
        self.done = done

        if done:
            self.terminate(obs["termination_cause"])

    def terminate(self, termination_reason):
        self.done = True
        self.termination_reason = termination_reason

    @property
    def success(self):
        return self.termination_reason == "Landed"

    @property
    def total_reward(self): return sum(self.episode_rewards)

    @property
    def mean_reward(self): return sum(self.episode_rewards) / self.episode_steps

    def to_csv(self, filename: str):
        self.to_frame().to_csv(filename)

    def to_frame(self) -> pd.DataFrame:
        ep = pd.concat([
            pd.DataFrame(self.state_history),
            pd.DataFrame(self.actions_history),
            pd.Series(self.episode_rewards, name="reward"),
            pd.Series([*([None]*(len(self._state_history)-1)), self.termination_reason], name="Termination"),
        ], axis=1)

        return ep

    @property
    def state_history(self):
        return list(map(lambda state: {
            "x": state[XX],
            "y": state[YY],
            "Vx": state[X_DOT],
            "Vy": state[Y_DOT],
            "alpha": state[ALPHA],
            "w": state[ALPHA_DOT],
            "fuel": state[FUEL],
        }, self._state_history))

    @property
    def actions_history(self):
        return list(map(lambda action: {
            "main_engine": action[0],
            "side_engine": action[1],
            "gimbal": action[2],
        }, self._actions_history))


    def get_att_history(self, att: str):
        return list(map(lambda x: x[att], self.state_history))

    def get_action_att_history(self, att: str):
        return list(map(lambda x: x[att], self.actions_history))

    def get_obs_att_history(self, att: str):
        return list(map(lambda x: x[att], self.obs_history))

    def plot(self, save_dir):

        plot_mission(save_dir,
                     X=(self.get_att_history("x"), self.get_att_history("y")),
                     V=(self.get_att_history("Vy"), self.get_att_history("Vx")),
                     THETA=self.get_att_history("alpha")
                     )


        plot_velocity(save_dir, Vy=self.get_att_history("Vy"), Vx=self.get_att_history("Vx"))
        plot_pitch_angle(save_dir, self.get_att_history("alpha"), self.get_att_history("w"))
        plot_trajectory(save_dir, self.get_att_history("x"), self.get_att_history("y"))
        plot_rewards(save_dir, self.episode_rewards)
        plot_control(save_dir,
                     self.get_action_att_history("main_engine"),
                     self.get_action_att_history("side_engine"),
                     self.get_obs_att_history("nozzle_angle"),
                     )
