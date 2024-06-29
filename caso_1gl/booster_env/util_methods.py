import os
from typing import Dict
import yaml
import logging
import numpy as np
from .constants import LAUNCH_PAD_CENTER, LAUNCH_PAD_HEIGHT, GROUND_HEIGHT
from Box2D import b2Body

def mean_lists_error(list_a, list_b):
    import numpy as np
    a = np.array(list_a)
    b = np.array(list_b)

    # Calculate the absolute differences between corresponding elements
    errors = np.abs(a - b)

    # Calculate the mean of the absolute differences
    mean_error_value = np.mean(errors)

    return mean_error_value

def noisy(mean, sigma):
    import numpy as np
    return np.random.normal(mean, sigma)

def target_distance(X):
    """
        Params: [float, float] X,Y coordinates from the gravity center of the booster (In the (0,0) origin referential)
        Returns: (float) Relative coordinates to the target
                 (float) Absolute euclidian distance to the target
    """
    X_TARGET = (LAUNCH_PAD_CENTER, LAUNCH_PAD_HEIGHT+GROUND_HEIGHT)
    rel_pos = (X[0] - X_TARGET[0], X[1] - X_TARGET[1])
    return rel_pos, np.linalg.norm(rel_pos)

def calculateCG(bodies: list[b2Body]):
    composite_mass = 0.0
    center_of_gravity = (0.0, 0.0)

    for body in bodies:
        body_mass = body.mass
        body_center = body.worldCenter
        composite_mass += body_mass
        center_of_gravity = (
            center_of_gravity[0] + body_mass * body_center[0],
            center_of_gravity[1] + body_mass * body_center[1]
        )

    center_of_gravity = (
        center_of_gravity[0] / composite_mass,
        center_of_gravity[1] / composite_mass
    )
    return center_of_gravity

def m_dot(F, Ve, Pe, _Pa, Ae, mixRatio):

    """
    :param F:
    :param Ve:
    :param Pe:
    :param _Pa:
    :param Ae:
    :param mixRatio:
    :return: total mass flow rate, fuel mass flow, oxidizer mass flow
    """
    _m_dot = (F - ((Pe - _Pa) * Ae)) / Ve
    return _m_dot, _m_dot / (1 + mixRatio), _m_dot * (mixRatio / (1 + mixRatio))