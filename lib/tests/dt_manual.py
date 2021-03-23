"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys
import cv2

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img
env = DuckietownEnv(
    seed=0,
    map_name='loop_obstacles',
    domain_rand=False,
    frame_skip=1,
    distortion=False,
    camera_rand=False,
    dynamics_rand=False)

env.reset()
env.render()

image_idx = 0
iterator = 0

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global iterator
    global image_idx
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    #print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    #print(obs)
    #if iterator % 10 == 0:
    obs_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    str_image_idx = str(image_idx).zfill(4)
    cv2.imwrite(f'test_images/{str_image_idx}.jpg', obs_rgb)
    image_idx += 1

    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()
    iterator += 1


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
