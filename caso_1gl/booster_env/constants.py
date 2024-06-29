import numpy as np
from enum import Enum

RENDER_FPS = 240
SI_UNITS_SCALE = 100
dt = 1/RENDER_FPS
# --- Constantes de conversão unidades box2d para SI (Validadas experimentalmente) ---- #
K_time = 10 #(RENDER_FPS/4)
K_velocity = 10 #SI_UNITS_SCALE / K_time
K_w = 0.04293#2.519  #40.0 / RENDER_FPS
K_angle = 1#0.4018

# Define the state variables
class State(Enum):
    y = 0
    Vy = 1
    fuel = 2


YY = State.y.value
Y_DOT = State.Vy.value
FUEL = State.fuel.value





" # ----------------------------------- BOOSTER CONSTANTS ------------------------------ # "

FIRST_STAGE_FUEL_CAPACITY = 395600 # [kg] (RP1 + LOX)
BOOSTER_EMPTY_MASS = 25600 # [kg]
BOOSTER_RADIUS = 1.85
BOOSTER_HEIGHT = 41.2

# Nozzle (Merlin-1D)
NOZZLE_RADIUS = 0.46 # [m[
NOZZLE_AREA = 0.9 # [m2]

# Main Engine (1 Merlin-1D)
MAX_GIMBAL_ANGLE = 10 # [deg]
MAX_GIMBAL_VELOCITY = 15 # [3deg/s]



M1D_PHI = 2.38 # LOX / RP-1
#M1D_ISP = 283 #
N_ENGINES = 3
M1D_MAX_THRUST = 845000 # [N] 100%
M1D_THRESHOLD = 0.57 # < 57% de potencia nao ativa
M1D_MIN_THRUST = M1D_MAX_THRUST*M1D_THRESHOLD # [N] 100%


# --- Atenção! Os parâmetros de desempenho do motor foram calculados a partir do
# software CEARUN para o motor Merlin 1D. Não alterar!
M1D_M = 3.713
M1D_Ve = 803.7 # [m/s]
M1D_Pe = 65400 # [Pa] = 0.654 bar
M1D_ISP = 304 # 7% > ISP real


# Side Engines (Draco)
DRACO_THRUST = 25e3


# Limits
X_LIMIT = 5000
Y_LIMIT = 5000 #600
TILT_ANGLE = np.deg2rad(60) # rad
TILT_LAND_ANGLE = np.deg2rad(23) # rad https://space.stackexchange.com/questions/8771/how-stable-would-a-falcon-9-first-stage-be-after-it-has-landed-on-a-drone-ship
ANGULAR_VELOCITY_LIMIT = 10 # rad/s
ANGULAR_VELOCITY_LAND_LIMIT = 5 # rad/s
Y_LAND_VELOCITY_LIMIT = 5 #10
X_LAND_VELOCITY_LIMIT = 5 #10 #5


GROUND_HEIGHT = 10
LAUNCH_PAD_CENTER = X_LIMIT/2
LAUNCH_PAD_HEIGHT = 2
LAUNCH_PAD_RADIUS = 85/2

# ----- Utils

EARTH_GRAVITY = 9.81
Pa = 10e5

# ---- Render
VIEWPORT_SIZE = 600
PIXELS_UNITS_SCALE = 100#80

