
        Action Space =>
                        0. Main engine: -1..0 off, 0..+1 throttle from 57% to 100% power.
                        1. Side engine: -1..-0.5 (left) | -0.5..0.5 (off) | 0.5..1 (right)
                        2. Delta gimbal angle -1..1

        Observation Space =>
                             0. X relative position to the lauchpad [m], [-X_LIM,+X_LIM]
                             1. Y relative position to the lauchpad [m], [GROUND_HEIGHT,+Y_LIM]
                             2. Vx [m/s] [-inf,+inf]
                             3. Vy [m/s] [-inf,+inf]
                             4. Pitch angle [rad]
                             5. w [rad/s]
                             6. Fuel / Initial Fuel [0-1]
                             7. Previous main engine power [0-1]
                             8. Previous nozzle angle [rad]

-----------------------------------------------------------------------------------------------------------------------

        Stability Criteria =>
                              0. X_LIMIT > XX > -X_LIMIT
                              1. YY > Y_LIMIT
                              2. ALPHA > TILT ANGLE (60deg)
                              3. w (alpha_dot) > ANGULAR_VELOCITY_LIMIT
                              4. step > MAX_STEPS
        Termination Criteria =>
                              0. Vy > Y_LAND_VELOCITY_LIMIT
                              1. Vx > X_LAND_VELOCITY_LIMIT
                              2. ALPHA > TILT_LAND_ANGLE (23deg)
                              3. LAUNCH_PAD_RADIUS >= XX >= -LAUNCH_PAD_RADIUS