import logging
import math
import numpy as np
from Box2D import *
from gymnasium.core import ObsType

from .constants import *
from .util_methods import *
from easydict import EasyDict
from gymnasium.utils import seeding, EzPickle


class World(EzPickle):

    def __init__(self,
                 np_random: np.random.Generator,
                 initial_state: EasyDict,
                 wind_power: float = 5000.0,
                 turbulence: float = 5000.0,
                 use_drag=True):
        EzPickle.__init__(self, np_random, initial_state, wind_power, turbulence, use_drag)
        self.np_random = np_random
        self.initial_state = initial_state

        self.world = Box2D.b2World(gravity=(0, -EARTH_GRAVITY))
        self.world._contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world._contactListener_keepref

        self.terrain = self._createTerrain()
        self.launch_pad = self._createLaunchPad()
        self.particles = []
        self.booster = self._createBooster()

        self._step = 0

        self._windStep = np.random.randint(-1000, 1000)
        self._windPower = wind_power
        self._turbulenceStep = np.random.randint(-1000, 1000)
        self._turbulencePower = turbulence
        self._useDrag = use_drag

        self.contact = False
        self.state = EasyDict()
        self._setInitialState(initial_state)

    @property
    def xy_target_pos(self):
        return target_distance((self.state.x, self.state.y))[0]

    def step(self, action) -> EasyDict:
        """
        :param action:
        :return: If it have landed or not
        """
        self._step += 1

        self.contact = self.legs[0].ground_contact or self.legs[1].ground_contact
        action[0] = np.clip(action[0], -1, 1)
        #action[1] = np.clip(action[1], -1, 1)

        # If there is any contact, turn off all engines
        if not self.contact:
            self._fireMainEngine(action[0], 0)
            #self._fireSideThruster(action[1])

        if self.state.y >= 100:
            #self._applyTurbulence()
            #self._applyWind()
            self._applyDrag()

        self.__updateParticles__()
        self.world.Step(dt, 6 * 60, 6 * 60)

        # Update the state variables
        self.set_state()

        return self.state

    def set_state(self):
        # Update the state variables
        x, y = self.fuselage.worldCenter
        Vx, Vy = self.fuselage.linearVelocity
        self.state.t = self._step * dt * K_time  # [s]
        self.state.x = x * SI_UNITS_SCALE  # [m]
        self.state.y = y * SI_UNITS_SCALE  # [m]
        self.state.angle = self.fuselage.angle * K_angle  # [rad]
        self.state.Vx = Vx * K_velocity  # [m/s]
        self.state.Vy = Vy * K_velocity  # [m/s]
        self.state.w = self.fuselage.angularVelocity * K_w  # [rad/s]
        self.state.legs_on_contact = sum(int(leg.ground_contact) for leg in self.legs)
        self.contact = self.state.legs_on_contact > 0

    def _setInitialState(self, state: EasyDict):

        # Convert meters the scales
        x = state.x / SI_UNITS_SCALE
        y = state.y / SI_UNITS_SCALE
        alpha = state.alpha / K_angle
        Vx = state.Vx / K_velocity
        Vy = state.Vy / K_velocity
        w = state.w / K_w
        fuel = state.fuel_ratio * FIRST_STAGE_FUEL_CAPACITY

        # Set the bodies
        self.fuselage.position = (x, y)
        self.fuselage.linearVelocity = (Vx, Vy)
        self.fuselage.angularVelocity = w
        self.fuselage.angle = alpha
        self.state.fuel = fuel
        self.state.nozzle_angle = 0.0
        self.state.legs_on_contact = 0
        self.state.F = (0, 0)

        # Update the state variables
        self.set_state()
        self._applyDrag()
        self._applyWind()
        self._applyTurbulence()

    # ------------------------------------- CONTROL ------------------------------------- #

    def _fireMainEngine(self, power: float, d_alpha: float):

        """
            Versao simplificada
        """

        self._mainEngineBurning = (self.state.fuel > 0 and power > 0)
        if not self._mainEngineBurning:
            return

        # !! self.state.nozzle_angle [deg]
        self.state.nozzle_angle += MAX_GIMBAL_VELOCITY * d_alpha * dt * K_time
        self.state.nozzle_angle = np.clip(self.state.nozzle_angle, -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)

        # thrust_angle = self.fuselage.angle + np.deg2rad(self.state.nozzle_angle)
        # sin = round(np.sin(thrust_angle), 1)
        # cos = round(np.cos(thrust_angle), 1)

        # Force magnitude and vector
        thrust_dispersion = self.np_random.uniform(0.95, 1)
        thrust = N_ENGINES * M1D_MAX_THRUST * (M1D_THRESHOLD + (0.43 * power))
        thrust_x = 0#thrust * (-sin + thrust_dispersion)
        thrust_y = thrust * thrust_dispersion#* (cos - thrust_dispersion)
        self.state.F = (thrust_x, abs(thrust_y))
        #self.nozzle.ApplyForceToCenter((thrust_x, thrust_y), True)
        self.fuselage.ApplyForceToCenter((thrust_x, thrust_y), True)

        consumedFuel = N_ENGINES * (m_dot(
            F=thrust / N_ENGINES,
            Ve=M1D_Ve,
            Pe=M1D_Pe,
            _Pa=Pa,
            mixRatio=M1D_PHI,
            Ae=NOZZLE_AREA)[0] * dt * K_time)  # rp1 + lox

        self.fuselage.mass -= consumedFuel
        self.state.fuel -= consumedFuel

        # Visual Only

        impulse_pos = self.nozzle.worldCenter
        m = consumedFuel/75  # power * consumedFuel / 100
        p = self.__createParticle__(
            mass=m,
            x=self.nozzle.worldCenter[0],
            y=self.nozzle.worldCenter[1],
            ttl=1,
            radius=3 + 1.8 * (power ** 2)
        )

        o = power * 100
        ox, oy = 0, -o
        p.ApplyForce((ox, oy), impulse_pos, True)


    def _fireSideThruster(self, side_prob: float):
        """
            :param side_prob The probability of firing the engine
            The side booster are discrete actions for now,
            -1 left side
            1 right side
        """

        if abs(side_prob) < 0.5: return
        side = np.sign(side_prob)
        p = abs(side_prob)
        side_index = 0 if side == -1 else 1

        thruster = self.sideThrusters[side_index]
        angle = round(self.fuselage.angle, 2)
        sin = side * np.sin(angle)
        cos = side * np.cos(angle)

        position_dispersion = [0, 0]  # [self.np_random.uniform(-0.05, +0.05) / SI_UNITS_SCALE for _ in range(2)]
        x = thruster.worldCenter[0] * (1 + position_dispersion[0])
        y = thruster.worldCenter[1] * (1 + position_dispersion[1])

        # Force magnitude and vector
        thrust_dispersion = 0  # self.np_random.uniform(-0.005, +0.005)
        thrust_y = DRACO_THRUST * p * (-sin + thrust_dispersion)
        thrust_x = DRACO_THRUST * p * (cos - thrust_dispersion)

        self.sideThrusters[side_index].ApplyForce((-thrust_x, -thrust_y), (x, y), True)

        # Visual particles
        p = self.__createParticle__(DRACO_THRUST, x, y, 1, radius=2.5)
        p.ApplyLinearImpulse((thrust_x / 200, thrust_y / 200), (x, y), True)

    def _applyWind(self):
        # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
        # which is proven to never be periodic, k = 0.01
        wind_mag = (np.tanh(np.sin(0.02 * self._windStep)+ (np.sin(np.pi * 0.01 * self._windStep)))
                * self._windPower
        )
        self._windStep += 1
        self.state.wind = wind_mag
        self.fuselage.ApplyForceToCenter(
            (wind_mag, 0.0),
            True,
        )

    def _applyTurbulence(self):
        # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
        # which is proven to never be periodic, k = 0.01
        turbulence = (
                np.tanh(
                    np.sin(0.02 * self._turbulenceStep)
                    + (np.sin(np.pi * 0.01 * self._turbulenceStep))
                )
                * self._turbulencePower
        )
        self._turbulenceStep += 1
        self.state.turbulence = turbulence
        self.fuselage.ApplyTorque(turbulence, True)

    def _applyDrag(self):
        self.state.drag = 0

    # --------------------------------------------- BUILD ------------------------------- #

    def _createTerrain(self) -> b2Body:
        terrain_coordinates = [(0, 0), (0, GROUND_HEIGHT), (X_LIMIT, GROUND_HEIGHT), (X_LIMIT, 0)]
        terrain_coordinates = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in terrain_coordinates]
        terrain = self.world.CreateStaticBody(shapes=b2PolygonShape(vertices=terrain_coordinates))
        terrain.CreateEdgeFixture(
            vertices=terrain_coordinates,
            density=0,
            friction=1
        )
        return terrain

    def _createLaunchPad(self) -> b2Body:
        self.launchPadConstraints = EasyDict()

        self.launchPadConstraints.center = (LAUNCH_PAD_CENTER, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)
        self.launchPadConstraints._bottomLeft = (LAUNCH_PAD_CENTER - LAUNCH_PAD_RADIUS, GROUND_HEIGHT)
        self.launchPadConstraints._bottomRight = (LAUNCH_PAD_CENTER + LAUNCH_PAD_RADIUS, GROUND_HEIGHT)
        self.launchPadConstraints._topLeft = (LAUNCH_PAD_CENTER - LAUNCH_PAD_RADIUS, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)
        self.launchPadConstraints._topRight = (LAUNCH_PAD_CENTER + LAUNCH_PAD_RADIUS, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT)

        pad_vertices = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in [
            self.launchPadConstraints._bottomLeft, self.launchPadConstraints._topLeft,
            self.launchPadConstraints._topRight, self.launchPadConstraints._bottomRight,
        ]]

        launch_pad = self.world.CreateStaticBody(shapes=b2PolygonShape(vertices=pad_vertices))
        launch_pad.CreateEdgeFixture(
            vertices=pad_vertices,
            density=0,
            friction=3
        )
        return launch_pad

    def _createBooster(self):

        BOOSTER_POLY = [
            (-BOOSTER_RADIUS, 0), (+BOOSTER_RADIUS, 0),
            (+BOOSTER_RADIUS, +BOOSTER_HEIGHT), (-BOOSTER_RADIUS, +BOOSTER_HEIGHT)
        ]
        BOOSTER_POLY = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in BOOSTER_POLY]

        LEGS_MASS = 2000
        INITIAL_MASS = (BOOSTER_EMPTY_MASS - LEGS_MASS) + (FIRST_STAGE_FUEL_CAPACITY * self.initial_state.fuel_ratio)
        FUSELAGE_DENSITY = INITIAL_MASS / (BOOSTER_HEIGHT * BOOSTER_RADIUS * 2) * (SI_UNITS_SCALE ** 2)
        initial_x, initial_y = self.initial_state.x / SI_UNITS_SCALE, self.initial_state.y / SI_UNITS_SCALE

        self.fuselage: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=BOOSTER_POLY),
                density=FUSELAGE_DENSITY,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0  # 0 (rigid) 0.99 (bouncy)
            ),
        )

        # ------------------------------------- LEGS ---------------------------------------------

        self.legs = []

        LEG_HEIGHT = 12  # BOOSTER_HEIGHT / 3
        LEG_WIDTH = 1
        LEG_ANGLE = np.deg2rad(45)

        for legDirection in [-1, 1]:

            leg_poly = [
                (0, 0),  # A
                (LEG_HEIGHT * np.sin(LEG_ANGLE) + LEG_WIDTH, LEG_HEIGHT * np.cos(LEG_ANGLE) + LEG_WIDTH),  # B
                (LEG_HEIGHT * np.sin(LEG_ANGLE) + LEG_WIDTH, LEG_HEIGHT * np.cos(LEG_ANGLE)),  # C
                (LEG_WIDTH, 0)  # D
            ]

            leg_poly = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in leg_poly]

            leg: b2Body = self.world.CreateDynamicBody(
                position=(initial_x, initial_y),
                angle=np.deg2rad(270) if legDirection == 1 else 0,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(vertices=leg_poly),
                    density=(2000 / (LEG_HEIGHT * LEG_WIDTH) * (SI_UNITS_SCALE ** 2)),
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x005
                )
            )
            leg.ground_contact = False

            joint = b2WeldJointDef(
                bodyA=self.fuselage,
                bodyB=leg,
                localAnchorA=(legDirection * BOOSTER_RADIUS / SI_UNITS_SCALE, 5.2 / SI_UNITS_SCALE),
                localAnchorB=leg_poly[1] if legDirection == -1 else leg_poly[0],
            )

            leg.joint = self.world.CreateJoint(joint)

            self.legs.append(leg)

            # ---------------------------------- SIDE BOOSTERS ------------------------------- #
            self.sideThrusters = []
            for side in [-1, 1]:
                w, h = 0.6, 1.2
                thruster = self.world.CreateDynamicBody(
                    position=(initial_x, initial_y),
                    angle=0,
                    fixtures=b2FixtureDef(
                        shape=b2PolygonShape(box=(w / SI_UNITS_SCALE, h / SI_UNITS_SCALE)),
                        density=FUSELAGE_DENSITY,
                        friction=0.1,
                        categoryBits=0x0050,
                        maskBits=0x001,  # collide only with ground
                        restitution=0.0  # 0 (rigid) 0.99 (bouncy)
                    ),
                )
                thruster.joint = self.world.CreateJoint(
                    b2WeldJointDef(
                        bodyA=self.fuselage,
                        bodyB=thruster,
                        localAnchorA=(
                            (side * (BOOSTER_RADIUS + w / 3)) / SI_UNITS_SCALE,
                            (BOOSTER_HEIGHT * 0.9) / SI_UNITS_SCALE),
                        localAnchorB=(0, 0),
                        referenceAngle=0,
                    )
                )
                self.sideThrusters.append(thruster)

            # ------------------------------------- NOZZLE -------------------------------------------
        NOZZLE_HEIGHT = 2
        NOZZLE_POLY = [
            (-BOOSTER_RADIUS * 0.8, NOZZLE_HEIGHT / 2), (BOOSTER_RADIUS * 0.8, NOZZLE_HEIGHT / 2),
            (NOZZLE_RADIUS, -NOZZLE_HEIGHT / 2), (-NOZZLE_RADIUS, -NOZZLE_HEIGHT / 2)
        ]
        NOZZLE_POLY = [(x / SI_UNITS_SCALE, y / SI_UNITS_SCALE) for x, y in NOZZLE_POLY]

        self.nozzle: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=NOZZLE_POLY),
                density=FUSELAGE_DENSITY,
                friction=0.0,
                categoryBits=0x0040,
                maskBits=0x003,  # collide only with ground
                restitution=0.0  # 0 (rigid) 0.99 (bouncy)
            ),
        )

        self.world.CreateWeldJoint(
            bodyA=self.fuselage,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, (NOZZLE_HEIGHT / 2) / SI_UNITS_SCALE),  # (0, (NOZZLE_HEIGHT / 2) / SI_UNITS_SCALE),
            referenceAngle=0
        )

        return [self.fuselage, self.nozzle] + self.legs + self.sideThrusters

    def __createParticle__(self, mass, x, y, ttl, radius=3.0):
        """
            Particles represents the engine thrusts
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius / SI_UNITS_SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0
            )
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self.__destroyParticles__(False)
        return p

    def __updateParticles__(self):
        for obj in self.particles:
            obj.ttl -= 0.1
        self.__destroyParticles__(False)

    def __destroyParticles__(self, destroyAll=False):
        while self.particles and (destroyAll or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def destroy(self):
        self.__destroyParticles__()

        if self.world is not None:
            self.world.DestroyBody(self.terrain)
            self.world.DestroyBody(self.launch_pad)
            self.world.contactListener = None
            for obj in self.booster:
                self.world.DestroyBody(obj)

        self.terrain = None
        self.particles = None
        self.launch_pad = None
        self.world = None
        self.booster = None


# Default implementation
class ContactDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
                self.env.fuselage == contact.fixtureA.body
                or self.env.fuselage == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False
