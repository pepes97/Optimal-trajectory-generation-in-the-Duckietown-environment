
import argparse
import sys

import gym
from pyglet import app, clock
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator

env = DuckietownEnv(seed=0,
                        map_name='loop_obstacles',
                        camera_rand=False,
                        frame_skip=1,
                        domain_rand=False,
                        dynamics_rand=False,
                        distortion=False)
env.reset()
env.render()

assert isinstance(env.unwrapped, Simulator)


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env.reset()
        env.render()
        return
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Camera movement
    env.unwrapped.cam_offset = env.unwrapped.cam_offset.astype("float64")
    cam_offset, cam_angle = env.unwrapped.cam_offset, env.unwrapped.cam_angle
    if symbol == key.W:
        cam_angle[0] -= 5
    elif symbol == key.S:
        cam_angle[0] += 5
    elif symbol == key.A:
        cam_angle[1] -= 5
    elif symbol == key.D:
        cam_angle[1] += 5
    elif symbol == key.Q:
        cam_angle[2] -= 5
    elif symbol == key.E:
        cam_angle[2] += 5
    elif symbol == key.UP:
        cam_offset[0] += 0.1
    elif symbol == key.DOWN:
        cam_offset[0] -= 0.1
    elif symbol == key.LEFT:
        cam_offset[2] -= 0.1
    elif symbol == key.RIGHT:
        cam_offset[2] += 0.1
    elif symbol == key.O:
        cam_offset[1] += 0.1
    elif symbol == key.P:
        cam_offset[1] -= 0.1
    elif symbol == key.T:
        cam_offset[0] = -0.8
        cam_offset[1] = 2.7
        cam_offset[2] = -1.2
        cam_angle[0] = 90
        cam_angle[1] = -40
        cam_angle[2] = 0
    elif symbol == key.H:
        print(f"offset: {cam_offset}")
        print(f"angle: {cam_angle}")

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage depencency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     try:
    #         from experiments.utils import save_img
    #         save_img('screenshot.png', img)
    #     except BaseException as e:
    #         print(str(e))


def update(dt):
    env.render("free_cam")


# Main event loop
clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
app.run()

env.close()