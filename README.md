        Action Space =>
                        0. Main engine: -1..0 off, 0..+1 throttle from 57% to 100% power.

        Observation Space =>
                             1. Y relative position to the lauchpad [m], [GROUND_HEIGHT,+Y_LIM]
                             3. Vy [m/s] [-inf,+inf]
                             6. Fuel / Max Fuel [0-1]

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
