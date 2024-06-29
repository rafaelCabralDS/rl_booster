import numpy as np
from easydict import EasyDict
from .constants import *


class Reward:

    def __init__(self, env_config: EasyDict):

        self._ready = False
        self.previous_shaping = None
        self.current = None
        self.previous = None

        self._reward_fn = None
        self._version = env_config.reward_version
        self.X_target = (LAUNCH_PAD_CENTER, LAUNCH_PAD_HEIGHT + GROUND_HEIGHT)

        self.R_burn = env_config.reward.get("main_engine_burn", 0)
        self.R_ignition = env_config.reward.get("engine_startup", 0)
        self.time_penalization = env_config.reward.get("time_penalization", 0)
        self.R_termination = env_config.reward.get("termination_reward", 100)
        self.R_v = env_config.reward.get("velocity", 0)  # env_config.reward.velocity or 0
        self.R_s = env_config.reward.get("position", 0) or 0
        self.fuel_penalization = env_config.reward.get("fuel_penalization", 0)
        self.R_w = env_config.reward.get("w", 0) or 0
        self.R_alpha = env_config.reward.get("angle", 0) or 0
        self.R_x = env_config.reward.get("x", 0) or 0
        self.R_y = env_config.reward.get("y", 0) or 0
        self.R_Vx = env_config.reward.get("Vx", 0) or 0
        self.R_Vy = env_config.reward.get("Vy", 0) or 0
        self.R_side = env_config.reward.get('side_engine_burn', 0)
        self.trajectory_penalization = env_config.reward.get("trajectory_penalization", 0)

        try:
            exec(f'self._reward_fn = self._{self._version}')
        except:
            raise Exception("Reward version not implemented")



    def _v5(self, **kwargs):

        """
            V (Vx, Vy) [m/s]
            X (x_cg - x_target, y_cg - y_target) [m]
        """

        V, X, alpha = kwargs["V"], kwargs["X"], abs(kwargs["alpha"])

        v = np.linalg.norm(V) / np.linalg.norm(self.V0)
        s = np.linalg.norm(X) / np.linalg.norm(self.X0)

        reward = 0
        shaping = - (self.R_v * v) \
                  - (self.R_s * s) \
                  - (self.R_alpha * alpha)

        terminated_successfully = kwargs["terminated_successfully"]
        if terminated_successfully is not None:
            r_ter = shaping/3
            return ((1 if terminated_successfully else -1) * self.R_termination) + r_ter



        if self.previous_shaping is not None:
            reward = shaping - self.previous_shaping
        self.previous_shaping = shaping

        # Para fins de estabilidade
        # reward = np.clip(reward, -10, 10)

        m_power = np.clip(kwargs["action"][0], 0, 1)
        prev_m_power = np.clip(kwargs["action_prev"][0], 0, 1)
        s_power = abs(kwargs["action"][1]) if (abs(kwargs["action"][1]) > 0.5) else 0
        # d_nozzle = kwargs["action"][2]

        # -> cubic root to relatively penalize harder low values of control
        reward -= self.R_burn * (np.cbrt(abs(m_power)))  # penalize the use of engines
        reward -= self.R_burn * np.cbrt(abs(s_power))  # Side engine burn penalization
        # reward -= 0.3 * abs(d_nozzle) # nozzle angle changes

        # penalize engine transition from OFF to ON # https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
        reward -= self.R_ignition if (m_power > 0 and prev_m_power == 0) else 0

        # if self.X0[1] > 1000 and X[1] <= 1000:
        #    # Penalizar o uso de motores laterais com menos de 1km para incentivar a traetoria desejada
        #    reward -= 0.5 * s_power

        if kwargs["fuel"] <= 0:
            reward -= self.fuel_penalization

        return reward

    def _v6(self, **kwargs):

        """
            V (Vx, Vy) [m/s]
            X (x_cg - x_target, y_cg - y_target) [m]
        """

        V, X, alpha = kwargs["V"], kwargs["X"],  abs(kwargs["alpha"])

        v = np.linalg.norm(V) / np.linalg.norm(self.V0)
        s = np.linalg.norm(X) / np.linalg.norm(self.X0)

        reward = 0
        shaping = - (self.R_v * v) \
                  - (self.R_s * s) \
                  - (self.R_alpha * alpha)

        terminated_successfully = kwargs["terminated_successfully"]
        if terminated_successfully is not None:
            r_ter = shaping / 2
            return ((1 if terminated_successfully else -1) * self.R_termination) + r_ter

        if self.previous_shaping is not None:
            reward = shaping - self.previous_shaping
        self.previous_shaping = shaping


        # Para fins de estabilidade
        # reward = np.clip(reward, -10, 10)

        m_power = np.clip(kwargs["action"][0], 0, 1)
        prev_m_power = np.clip(kwargs["action_prev"][0], 0, 1)
        #s_power = abs(kwargs["action"][1]) if (abs(kwargs["action"][1]) > 0.5) else 0
        #d_nozzle = kwargs["action"][2]


        # -> cubic root to relatively penalize harder low values of control
        reward -= self.R_burn * np.cbrt(abs(m_power))  # penalize the use of engines
        #reward -= self.R_side * np.cbrt(abs(s_power)) # Side engine burn penalization
        #reward -= self.R_gimbal * np.cbrt(abs(d_nozzle)) # nozzle angle changes


        # penalize engine transition from OFF to ON # https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
        reward -= self.R_ignition if (m_power > 0 and prev_m_power == 0) else 0


        if kwargs["fuel"] <= 0:
            reward -= self.fuel_penalization

        # Creates tree direction zones:
        # <- left | ----- neutral ----- | right ->
        # If the x velocity zone changes from the last step, penalize it
        Vx_threshold = 1
        if np.sign(self.V0[0]) != np.sign(V[0]) and abs(V[0]) > Vx_threshold:
            reward -= self.trajectory_penalization


        return reward

    def _v7(self, **kwargs):

        """
            V (Vx, Vy) [m/s]
            X (x_cg - x_target, y_cg - y_target) [m]
        """

        V, X, alpha = kwargs["V"], kwargs["X"], abs(kwargs["alpha"])

        vx = abs(V[0]) / abs(self.V0[0])
        vy = abs(V[1]) / abs(self.V0[1])
        s = np.linalg.norm(X) / np.linalg.norm(self.X0)

        reward = 0
        shaping = - (self.R_Vx * vx) \
                  - (self.R_Vy * vy) \
                  - (self.R_s * s) \
                  - (self.R_alpha * alpha)

        terminated_successfully = kwargs["terminated_successfully"]
        if terminated_successfully is not None:
            r_ter = shaping / 3
            return ((1 if terminated_successfully else -1) * self.R_termination) + r_ter

        if self.previous_shaping is not None:
            reward = shaping - self.previous_shaping
        self.previous_shaping = shaping

        # Para fins de estabilidade
        # reward = np.clip(reward, -10, 10)

        m_power = np.clip(kwargs["action"][0], 0, 1)
        prev_m_power = np.clip(kwargs["action_prev"][0], 0, 1)
        s_power = abs(kwargs["action"][1]) if (abs(kwargs["action"][1]) > 0.5) else 0
        #d_nozzle = kwargs["action"][2]

        # -> cubic root to relatively penalize harder low values of control
        reward -= self.R_burn * np.cbrt(abs(m_power))  # penalize the use of engines
        reward -= self.R_side * np.cbrt(abs(s_power))  # Side engine burn penalization
        #reward -= self.R_gimbal * np.cbrt(abs(d_nozzle))  # nozzle angle changes

        # penalize engine transition from OFF to ON # https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
        reward -= self.R_ignition if (m_power > 0 and prev_m_power == 0) else 0

        if kwargs["fuel"] <= 0:
            reward -= self.fuel_penalization

        # Creates tree direction zones:
        # <- left | ----- neutral ----- | right ->
        # If the x velocity zone changes from the last step, penalize it
        Vx_threshold = 1
        if np.sign(self.V0[0]) != np.sign(V[0]) and abs(V[0]) > Vx_threshold:
            reward -= self.trajectory_penalization

        return reward

    def calculate(self, **kwargs):

        if not self._ready:
            raise Exception("Reset function was not called")

        self.previous = self.current
        self.current = self._reward_fn(**kwargs)

    def reset(self, X0, V0):
        self._ready = True
        self.current = None
        self.previous = None
        self.previous_shaping = None
        self.X0 = X0
        self.V0 = V0
