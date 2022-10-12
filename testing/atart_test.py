# https://www.gymlibrary.dev/
# https://www.gymlibrary.dev/environments/atari/
# https://www.gymlibrary.dev/environments/atari/breakout/

# pip install gym[atari,accept-rom-license]
# https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym

"""
Actions
================
0 - NOOP
1 - FIRE
2 - RIGHT
3 - LEFT

"""


import gym
import numpy as np

import matplotlib.pyplot as plt

SEED = False

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
# env = gym.make("ALE/Breakout-v5", render_mode="human", obs_type="grayscale")

if SEED:
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
else:
    observation, info = env.reset()

fig = plt.figure()

NUM_FRAMES = 4

# each axis is a graph
# ax1 = plt.subplot2grid((2, 1), (0, 0))
# ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

# axs = []

# obses = []

ROWS = 1
COLUMNS = NUM_FRAMES

for i in range(NUM_FRAMES):
    action = env.action_space.sample()
    print("action", action)
    observation, reward, terminated, truncated, info = env.step(action)

    print(observation.shape)  # (210, 160, 3)

    # obses.append(observation)

    ax = fig.add_subplot(ROWS, COLUMNS, (i+1))
    ax.title.set_text(f"frame: {i}\naction: {action}")
    plt.imshow(observation)

    # ax = plt.subplot2grid((NUM_FRAMES, 1), (0, (i+1)))
    # ax.imshow(observation)
    # axs.append(ax)

    # ax1.plot(times, accuracies, label="acc")
    # ax1.plot(times, val_accs, label="val_acc")
    # ax1.legend(loc=2)

    # ax2.plot(times, losses, label="loss")
    # ax2.plot(times, val_losses, label="val_losses")
    # ax2.legend(loc=3)

    # plt.show()

    # plt.imshow(observation)

    if terminated or truncated:
        observation, info = env.reset()

# fig = plt.figure()


# ax1 = plt.subplot2grid((NUM_FRAMES, 1), (0, 1))
# ax1.plot(observation)
# axs.append(ax)

plt.show()

env.close()


# class RelativePosition(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

#     def observation(self, obs):
#         return obs["target"] - obs["agent"]
