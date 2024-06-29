from typing import Optional, Tuple, List, Any

from gymnasium import *
from gymnasium.core import RenderFrame, ActType

from utils.yaml_utils import load_config
from .painter import *
from .rewards import Reward
from .world import *


class BoosterEnv(Env, EzPickle):
    """
        Action Space =>
                        0. Main engine: -1..0 off, 0..+1 throttle from 57% to 100% power.
                        1. Side engine: -1..-0.5 (left) | -0.5..0.5 (off) | 0.5..1 (right)

        Observation Space =>
                             0. X relative position to the lauchpad [m], [-X_LIM,+X_LIM]
                             1. Y relative position to the lauchpad [m], [GROUND_HEIGHT,+Y_LIM]
                             2. Vx [m/s] [-inf,+inf]
                             3. Vy [m/s] [-inf,+inf]
                             4. Pitch angle [rad]
                             5. Fuel / Max Fuel [0-1]

    """

    def __init__(self,
                 config: Union[EasyDict, str],
                 render: Optional[bool] = False,
                 ):

        EzPickle.__init__(self, config)
        Env.__init__(self)

        self._loadEnvConfig(config)
        # self.seed = seed
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.render_mode = "human" if (self.config["render"] or render) else "rgb_Array"
        self.painter = TrackPainter(render_mode=self.render_mode)
        self.world: World = None

        self.termination_cause = None
        self.initial_state = None
        self.previousState = None
        self.currentAction = None
        self.previousAction = None

        self._is_env_ready = False
        self.reward = Reward(self.config)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)

        if self.world is not None:
            self.world.destroy()



        # Add initial state noise
        self.initial_state = EasyDict(dict(self.config.initial_condition))
        for key, value in dict(self.config.initial_condition).items():
            mean, sigma = [float(v) for v in value.strip().split("+-")]
            self.initial_state[key] = np.clip(noisy(mean, sigma), mean - sigma, mean + sigma)

        # Random direction
        initial_direction = [-1, 1][np.random.randint(0, 2)]
        self.initial_state["Vx"] *= -initial_direction
        self.initial_state["alpha"] *= -initial_direction
        self.initial_state["x"] = LAUNCH_PAD_CENTER + (initial_direction*(LAUNCH_PAD_CENTER-self.initial_state["x"]))

        self.world = World(
            np_random=self.np_random,
            initial_state=self.initial_state,
            wind_power=self.config.wind,
            turbulence=self.config.turbulence,
            use_drag=self.config.drag,
        )

        # Set rewards helper class
        self.reward.reset(
            X0=self.world.xy_target_pos,
            V0=(self.world.state.Vx, self.world.state.Vy)
        )

        self.previousAction = [0, 0]
        self.currentAction = [0, 0]
        self._is_env_ready = True
        self.previousState = self.state
        self.termination_cause = None

        return np.array(self.state, dtype=np.float32), {}

    @property
    def state(self) -> ObsType:
        state = self.world.state
        expanded_state = (
            self.world.xy_target_pos[0],
            self.world.xy_target_pos[1] , # Relative to target
            state.Vx,
            state.Vy,
            state.angle,
            state.fuel / FIRST_STAGE_FUEL_CAPACITY,
        )
        return np.array(expanded_state, dtype=np.float32)

    @property
    def obs(self):
        """Other environment useful variables + latest action"""
        return {
            "t": self.world.state.t,
            "termination_cause": self.termination_cause,
            "drag": self.world.state.drag,
            "turbulence": self.world.state.turbulence,
            "wind": self.world.state.wind,
            "action": self.currentAction,
            "F": self.world.state.F,
            "nozzle_angle": self.world.state.nozzle_angle,
            "state": self.state,
            "w": self.world.state.w,
        }

    def step(
            self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:

        if not self._is_env_ready:
            raise Exception("Environment is not ready. Call reset()")


        self.world.step(action)

        terminated, self.termination_cause = self._eval_termination()

        # Update the records
        self.previousAction = self.currentAction
        self.currentAction = action
        self.previousState = self.state

        self.reward.calculate(
            terminated_successfully=terminated,
            X=(self.state[XX], self.state[YY]),
            V=(self.state[X_DOT], self.state[Y_DOT]),
            alpha=self.state[ALPHA],
            #w=self.state[ALPHA_DOT],
            action=action,
            action_prev=self.previousAction,
            previous_state=self.previousState,
            # legs_on_contact=self.world.state.legs_on_contact,
            step=self.world.state.t / (dt * K_time),
            fuel=self.world.state.fuel,
        )


        self.render()

        return self.state, self.reward.current, terminated is not None, False, self.obs

    def _eval_termination(self):

        # Time Limit
        step = self.world.state.t / (dt * K_time)
        if step > self.config.max_steps:
            return False, "Time Limit"

        # Eval contact
        if self.world.contact:

            exploded = any([
                abs(self.previousState[Y_DOT]) >= Y_LAND_VELOCITY_LIMIT, # Y velocity limit,
                abs(self.previousState[X_DOT]) >= X_LAND_VELOCITY_LIMIT,  # X velocity limit,
                abs(self.previousState[ALPHA]) >= TILT_LAND_ANGLE,
                abs(self.previousState[XX]) >= LAUNCH_PAD_RADIUS, # Launchpad
            ])

            if exploded:
                return False, "Explosion"
            else:
                return True, "Landed"

        # Eval stability

        if abs(self.state[XX]) > X_LIMIT: return False, "X limit"
        if abs(self.state[YY]) > Y_LIMIT: return False, "Y limit"
        if abs(self.state[ALPHA]) > TILT_ANGLE: return False, "Tilted"
        #if abs(self.state[ALPHA_DOT]) > ANGULAR_VELOCITY_LIMIT: return False, "W"

        # Keep going
        return None, None

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "human":
            self.painter.paint(world=self.world)
        return

    def close(self):
        if self.painter is not None:
            self.painter.dispose()
        if self.world is not None:
            self.world.destroy()

    def _loadEnvConfig(self, config: Union[EasyDict, str]):

        if isinstance(config, EasyDict):
            self.config = config
        elif isinstance(config, str):
            self.config = EasyDict(load_config(config))
        elif isinstance(config, dict):
            self.config = EasyDict(config)
        else:
            raise Exception("Could not read the config file")

