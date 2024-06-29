import os.path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import numpy as np

from booster_env.constants import RENDER_FPS, K_time, Y_LIMIT, X_LIMIT, LAUNCH_PAD_CENTER, GROUND_HEIGHT, \
    LAUNCH_PAD_HEIGHT, LAUNCH_PAD_RADIUS
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_iterations(save_dir, episodic_returns, episodic_lengths):
    assert len(episodic_returns) == len(episodic_lengths)
    iterations = np.arange(len(episodic_returns))
    sns.lineplot(
        x=iterations,
        y=episodic_returns,  # df["Episodic Return"].expanding().mean(),
        label="Episodic Return"
    )
    plt.twinx(plt.gca())
    sns.lineplot(
        x=iterations,  # df["Iteration"],
        y=episodic_lengths,  # df["Average Episodic Length"],
        label="Episodic Length",
        color="r",
    )

    plt.xlabel("Iteration")
    plt.savefig(f"{save_dir}/training.png")
    plt.close()


def plot_loss(save_dir, loss):
    iterations = np.arange(len(loss))
    sns.lineplot(
        x=iterations,  # df["Iteration"],
        y=loss,  # df["Average Loss"],
    )
    plt.title("Agent loss")
    plt.xlabel("Iteration")
    plt.savefig(f"{save_dir}/agent_loss.png")
    plt.close()


def plot_episodes_rewards(save_dir, rewards):
    episodes = np.arange(len(rewards))
    sns.scatterplot(x=episodes, y=rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"{save_dir}/rewards.png")
    plt.close()


def plot_terminations(save_dir, terminations: List[str]):
    palette_color = seaborn.color_palette('hls', 8)

    # plotting data on chart
    plt.title("Terminations")
    categories, counts = np.unique(terminations, return_counts=True)
    plt.pie(counts, labels=categories, colors=palette_color,
            # explode=explode,
            autopct='%.0f%%')
    plt.savefig(os.path.join(save_dir, "terminations"))
    plt.close()


def plot_rewards(save_dir, rewards):
    sns.lineplot(x=np.arange(len(rewards)), y=rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Rewards")

    plt.savefig(f"{save_dir}/rewards.png")
    plt.close()


def plot_pitch_angle(save_dir, angle, w):
    t = [((K_time * t) / RENDER_FPS) for t in range(len(angle))]
    sns.lineplot(x=t, y=[np.rad2deg(a) for a in angle], label="theta")
    plt.ylabel("Pitch Angle [deg]")
    plt.xlabel("Time [s]")

    plt.twinx(plt.gca())
    sns.lineplot(x=t, y=w, label="theta_dot", color='y', )
    plt.ylabel("Angular velocity [rad/s]")

    plt.title("Pitch angle and angular velocity")
    plt.savefig(f"{save_dir}/pitch_angle.png")
    plt.close()


def plot_control(save_dir, main_engine, side_engine, nozzle_angle):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 12))
    fig.suptitle("Agent control")

    t = [((K_time * t) / RENDER_FPS) for t in range(len(main_engine))]

    def side_output(p):
        if abs(p > 0.5): return p
        return None

    # Main Engine
    axs[0].plot(t, [p if p > 0 else 0 for p in main_engine])
    axs[0].set_title("Main Engine Power")
    axs[0].grid(True)

    # Gimbal
    nozzle_angle = [np.rad2deg(round(a, 2)) for a in nozzle_angle]
    axs[1].plot(t, nozzle_angle)
    axs[1].grid(True)
    axs[1].set_title("Nozzle Angle [deg]")

    # Side Engine

    side_engine = list(filter(lambda x: x[1] is not None, [(t_, p) for t_, p in zip(t, map(side_output, side_engine))]))
    side_engine = np.array(side_engine)

    if len(side_engine) > 0:
        axs[2].scatter(side_engine[:, 0], side_engine[:, 1])
    axs[2].set_title("Side Engine")
    axs[2].grid(True)
    axs[2].set_ylim(-1.1, 1.1)

    plt.savefig(f"{save_dir}/agent_control.png")
    plt.close()


def plot_velocity(save_dir, Vx: Optional = None, Vy: Optional = None):
    assert Vx is not None or Vy is not None
    if Vx is not None and Vy is not None:
        assert len(Vx) == len(Vy)

    if Vx is not None:
        steps = np.arange(len(Vx))
        time = list(map(lambda x: (K_time * x) / RENDER_FPS, steps))
        sns.lineplot(x=time, y=Vx, label="Vx")

    if Vy is not None:
        steps = np.arange(len(Vy))
        time = list(map(lambda x: (K_time * x) / RENDER_FPS, steps))
        sns.lineplot(x=time, y=Vy, label="Vy")

    plt.title("Velocity Profile")
    plt.xlabel("Time [s]")
    plt.ylabel("V [m/s]")
    plt.savefig(f"{save_dir}/velocity_profile.png")
    plt.close()


def plot_trajectory(save_dir, x, y):
    sns.scatterplot(x=x, y=y)

    plt.scatter(x=[-LAUNCH_PAD_RADIUS, LAUNCH_PAD_RADIUS],
                y=[GROUND_HEIGHT + LAUNCH_PAD_HEIGHT, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT], color='red', marker='x',
                label="Launchpad")
    plt.scatter(x=[x[0]], y=[y[0]], color='b', marker='o', label="Initial Position")
    plt.legend()

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory")
    plt.xlim(-max([*x, abs(min(x))]) * 1.2, max([*x, abs(min(x))]) * 1.2)
    plt.ylim(0, max(y) * 1.2)

    plt.savefig(f"{save_dir}/trajectory_profile.png")
    plt.close()


def plot_mission(save_dir, X, V, THETA):
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))

    # Trajectory
    x, y = X
    sns.scatterplot(x=x, y=y, markers='-',ax=axs[0])

    axs[0].scatter(x=[-LAUNCH_PAD_RADIUS, LAUNCH_PAD_RADIUS],
                   y=[GROUND_HEIGHT + LAUNCH_PAD_HEIGHT, GROUND_HEIGHT + LAUNCH_PAD_HEIGHT],
                   color='red',
                   marker='x',
                   label="Launchpad")
    axs[0].scatter(x=[x[0]], y=[y[0]], color='b', marker='o', label="Initial Position")
    axs[0].legend()
    axs[0].set(
        xlabel="x [m]",
        ylabel="y [m]",
        title="Trajectory",
        xlim=(-max([*x, abs(min(x))]) * 1.2, max([*x, abs(min(x))]) * 1.2),
        ylim=(0, max(y) * 1.2)
    )

    # Velocity
    Vx, Vy = V
    time = list(map(lambda t: (K_time * t) / RENDER_FPS, np.arange(len(Vx))))
    sns.lineplot(x=time, y=Vx, label="Vx", ax=axs[1])
    sns.lineplot(x=time, y=Vy, label="Vy", ax=axs[1])
    axs[1].legend()
    axs[1].set(
        xlabel="Time [s]",
        ylabel="V [m/s]",
        title="Velocity Profile",
    )

    # Angle
    t = [((K_time * t) / RENDER_FPS) for t in range(len(THETA))]
    sns.lineplot(x=t, y=[np.rad2deg(a) for a in THETA], label="theta", ax=axs[2])
    axs[2].set(
        xlabel="Time [s]",
        ylabel="Pitch Angle [deg]",
        title="Pitch Angle"
    )

    plt.savefig(os.path.join(save_dir, "mission.png"))
    plt.close()
