import itertools
from gym.envs.registration import register
import numpy as np

for v in ['1']:
    register(id='PlaygroundNavigation-v' + v,
             entry_point='src.playground_env.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=50)

    register(id='PlaygroundNavigationHuman-v' + v,
             entry_point='src.playground_env.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=50,
             kwargs=dict(human=True, render_mode=True))

    register(id='PlaygroundNavigationRender-v' + v,
             entry_point='src.playground_env.playgroundnavv' + v + ':PlayGroundNavigationV' + v,
             max_episode_steps=50,
             kwargs=dict(human=False, render_mode=True))
