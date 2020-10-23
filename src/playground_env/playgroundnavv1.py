from __future__ import division
import gym
from gym import spaces
import numpy as np
import pygame
from src.playground_env.objects import generate_objects
from src.playground_env.env_params import get_env_params


class PlayGroundNavigationV1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    '''
        Playground Environment:
        set reward_screen to True to visualize modular reward function predictions
        set viz_data_collection to True to visualize Social Partner interactions 
    '''

    def __init__(self,
                 max_timesteps=50,
                 random_init=False,
                 human=False,
                 reward_screen=False,
                 viz_data_collection=False,
                 display=True,
                 agent_step_size=0.15,
                 agent_initial_pos=(0, 0),
                 agent_initial_pos_range=0.6,
                 max_nb_objects=3,  # number of objects in the scene
                 random_nb_obj=False,
                 admissible_actions=('Move', 'Grasp', 'Grow'),  # which types of actions are admissible
                 admissible_attributes=('colors', 'categories', 'types'),
                 # , 'relative_sizes', 'shades', 'relative_shades', 'sizes', 'relative_positions'),
                 # which object attributes
                 # can be used
                 min_max_sizes=((0.2, 0.25), (0.25, 0.3)),  # ranges of sizes of objects (small and large ones)
                 agent_size=0.05,  # size of the agent
                 epsilon_initial_pos=0.3,  # epsilon to sample initial positions
                 screen_size=800,  # size of the visualization screen
                 next_to_epsilon=0.3,  # define the area to qualify an object as 'next to' another.
                 attribute_combinations=False,
                 obj_size_update=0.04,
                 render_mode=False
                 ):

        self.params = get_env_params(max_nb_objects=max_nb_objects,
                                     admissible_actions=admissible_actions,
                                     admissible_attributes=admissible_attributes,
                                     min_max_sizes=min_max_sizes,
                                     agent_size=agent_size,
                                     epsilon_initial_pos=epsilon_initial_pos,
                                     screen_size=screen_size,
                                     next_to_epsilon=next_to_epsilon,
                                     attribute_combinations=attribute_combinations,
                                     obj_size_update=obj_size_update,
                                     render_mode=render_mode
                                     )
        self.adm_attributes = self.params['admissible_attributes']
        self.adm_abs_attributes = [a for a in self.adm_attributes if 'relative' not in a]

        self.attributes = self.params['attributes']
        self.categories = self.params['categories']
        self.screen_size = self.params['screen_size']

        self.viz_data_collection = viz_data_collection
        self.show_imagination_bubble = False
        self.reward_screen = reward_screen
        self.first_action = False
        self.SP_feedback = False
        self.known_goals_update = False
        self.known_goals_descr = []
        self.display = display
        self.circles = [[x * 3, 200, x * 4] for x in range(50)]

        self.random_init = random_init
        self.max_timesteps = max_timesteps

        # Dimensions of action and observations spaces
        self.dim_act = 3
        self.max_nb_objects = self.params['max_nb_objects']
        self.random_nb_obj = random_nb_obj
        self.nb_obj = self.params['max_nb_objects']
        self.dim_obj = self.params['dim_obj_features']
        self.dim_body = self.params['dim_body_features']
        self.inds_objs = [np.arange(self.dim_body + self.dim_obj * i_obj, self.dim_body + self.dim_obj * (i_obj + 1))
                          for i_obj in range(self.nb_obj)]

        self.half_dim_obs = self.max_nb_objects * self.dim_obj + self.dim_body
        self.dim_obs = int(2 * self.half_dim_obs)

        # We define the spaces
        self.action_space = spaces.Box(low=-np.ones(self.dim_act),
                                       high=np.ones(self.dim_act),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(self.dim_obs),
                                            high=np.ones(self.dim_obs),
                                            dtype=np.float32)

        # Agent parameters
        self.agent_step_size = agent_step_size
        self.agent_initial_pos = agent_initial_pos
        if self.display:
            self.agent_initial_pos_range = agent_initial_pos_range
        else:
            self.agent_initial_pos_range = 0.8

        # rendering
        self.human = human
        self.render_mode = render_mode
        self.logits_concat = (0 for _ in range(self.nb_obj))
        if self.render_mode:
            pygame.init()
            if self.display:
                if self.reward_screen:
                    self.viewer = pygame.display.set_mode((self.screen_size + 300, self.screen_size))
                else:
                    self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
            else:
	            if self.reward_screen:
	                self.viewer = pygame.Surface((self.screen_size + 300, self.screen_size))
	            else:
	                self.viewer = pygame.Surface((self.screen_size, self.screen_size))
            self.viewer_started = False
        self.background = None

        self.reset()

        # We set to None to rush error if reset not called
        self.observation = None
        self.initial_observation = None
        self.done = None

    def regularize_type_and_attribute(self, object):
        if object['categories'] is None and object['types'] is not None:
            for k in self.categories.keys():
                if object['types'] in self.categories[k]:
                    object['categories'] = k
        elif object['categories'] is not None and object['types'] is None:
            object['types'] = np.random.choice(self.categories[object['categories']])
        elif object['categories'] is None and object['types'] is None:
            object['categories'] = np.random.choice(list(self.categories.keys()))
            object['types'] = np.random.choice(self.categories[object['categories']])
        elif object['categories'] is not None and object['types'] is not None:
            if object['types'] not in self.categories[object['categories']]:
                object['types'] = np.random.choice(self.categories[object['categories']])
        return object.copy()

    def complete_and_check_objs(self, objects_decr):
        objects_decr = [self.regularize_type_and_attribute(o) for o in objects_decr]
        for o in objects_decr:
            for k in o.keys():
                if o[k] is None:
                    o[k] = np.random.choice(self.attributes[k])
        return objects_decr.copy()

    def reset_with_goal(self, goal_str):
        words = goal_str.split(' ')
        objs = []

        if words[0] == 'Grow':
            obj_to_be_grown = dict(zip(self.adm_abs_attributes, [None for _ in range(len(self.adm_abs_attributes))]))
            obj_supply = dict(zip(self.adm_abs_attributes, [None for _ in range(len(self.adm_abs_attributes))]))

            # first add the object that should be grown
            for w in words[1:]:
                for k in self.adm_abs_attributes:
                    if w in self.attributes[k]:
                        obj_to_be_grown[k] = w
            if obj_to_be_grown['categories'] is None and obj_to_be_grown['types'] is None:
                # if only attributes are proposed, sample a grownable object type
                obj_to_be_grown['categories'] = np.random.choice(['animal', 'plant'])
            objs.append(obj_to_be_grown.copy())

            # now sample the supply
            if obj_to_be_grown['categories'] in ['living_thing', 'plant'] or obj_to_be_grown['types'] in \
                    self.categories['plant']:
                obj_supply.update(dict(types='water',
                                       categories='supply'))
            else:
                obj_supply.update(dict(categories='supply'))
            objs.append(obj_supply.copy())

        else:
            obj = dict(zip(self.adm_abs_attributes, [None for _ in range(len(self.adm_abs_attributes))]))
            for w in words[1:]:
                for k in self.adm_abs_attributes:
                    if w in self.attributes[k]:
                        obj[k] = w
            objs.append(obj.copy())

        return self.reset_scene(objs)

    def reset(self):
        if self.random_nb_obj:
            self.nb_obj = np.random.randint(2, self.max_nb_objects)
            self.half_dim_obs = self.nb_obj * self.dim_obj + self.dim_body
            self.dim_obs = int(2 * self.half_dim_obs)

        self.first_action = False
        self.logits_concat = (0 for _ in range(self.nb_obj))
        self.SP_feedback = False
        self.known_goals_update = False
        return self.reset_scene()

    def reset_scene(self, objects=None):

        self.agent_pos = self.agent_initial_pos

        if self.random_init:
            self.agent_pos += np.random.uniform(-self.agent_initial_pos_range, self.agent_initial_pos_range, 2)
            self.gripper_state = np.random.choice([-1, 1])
        else:
            self.gripper_state = -1

        self.objects = self.sample_objects(objects)

        # Print objects
        self.object_grasped = False
        for obj in self.objects:
            self.object_grasped = obj.update_state(self.agent_pos,
                                                   self.gripper_state > 0,
                                                   self.objects,
                                                   self.object_grasped,
                                                   np.zeros([self.dim_act]))

        # construct vector of observations
        self.observation = np.zeros(self.dim_obs)
        self.observation[:self.half_dim_obs] = self.observe()
        self.initial_observation = self.observation[:self.half_dim_obs].copy()
        self.env_step = 0
        self.done = False
        return self.observation.copy()

    def get_pixel_coordinates(self, xpos, ypos):
        return ((xpos + 1) / 2 * (self.params['screen_size'] * 2 / 3) + 1 / 6 * self.params['screen_size']).astype(
            np.int), \
               ((-ypos + 1) / 2 * (self.params['screen_size'] * 2 / 3) + 1 / 6 * self.params['screen_size']).astype(
                   np.int)

    def sample_objects(self, objects_to_add):
        object_descr = objects_to_add if objects_to_add is not None else []
        while len(object_descr) < self.nb_obj:
            object = dict()
            for k in self.adm_abs_attributes:
                object[k] = np.random.choice(self.attributes[k])
            object_descr.append(object)
        object_descr = self.complete_and_check_objs(object_descr)
        objects_ids = [self.get_obj_identifier(o) for o in object_descr]
        objects = generate_objects(object_descr, self.params)
        return objects

    def get_obj_identifier(self, object):
        id_str = ''
        for k in sorted(list(object.keys())):
            id_str += '{}:{} '.format(k, object[k])
        return id_str

    def observe(self):

        obj_features = np.array([obj.get_features() for obj in self.objects]).flatten()
        obs = np.concatenate([self.agent_pos,  # size 2
                              np.array([self.gripper_state]),
                              obj_features,
                              ])

        return obs.copy()

    def step(self, action):
        # actions
        # 0 = x
        # 1 = y
        # 2 = gripper

        """
        Run one timestep of the environment's dynamics.
        """
        action = np.array(action).clip(-1, 1)

        if np.sum(action) != 0:
            self.first_action = True

        # Update the agent position
        self.agent_pos = np.clip(self.agent_pos + action[:2] * self.agent_step_size, -1.2, 1.2)

        # Update the gripper state
        if self.human:
            if action[2] > 0:
                self.gripper_state = 1 if self.gripper_state == -1 else -1
        else:
            if action[2] > 0.:
                new_gripper = 1
            else:
                new_gripper = -1
            self.gripper_change = new_gripper == self.gripper_state
            self.gripper_state = new_gripper

        for obj in self.objects:
            self.object_grasped = obj.update_state(self.agent_pos,
                                                   self.gripper_state > 0,
                                                   self.objects,
                                                   self.object_grasped,
                                                   action)

        self.observation[:self.half_dim_obs] = self.observe()
        self.observation[self.half_dim_obs:] = self.observation[:self.half_dim_obs] - self.initial_observation

        self.env_step += 1
        if self.env_step == self.max_timesteps:
            self.done = True

        return self.observation.copy(), 0, False, {}

    def render(self, goal_str, mode='human', close=False):

        background_color = [220, 220, 220]
        FONT = pygame.font.Font(None, 25)
        self.viewer.fill(background_color)
        self.shapes = {}
        self.anchors = {}
        self.patches = {}

        # OBJECTS
        for object in self.objects:
            object.update_rendering(self.viewer)

        # REWARD SCREEN
        if self.reward_screen:
            pygame.draw.rect(self.viewer, pygame.Color('darkgray'), (800, 0, 300, 800))
            goal_txt_surface = FONT.render(goal_str, True, pygame.Color('black'))
            self.viewer.blit(goal_txt_surface, (800 + 150 - goal_txt_surface.get_width() // 2, 50))

            cross_icon = pygame.image.load(self.params['img_path'] + 'cross.png')
            cross_icon = pygame.transform.scale(cross_icon, (50, 50))

            tick_icon = pygame.image.load(self.params['img_path'] + 'tick.png')
            tick_icon = pygame.transform.scale(tick_icon, (50, 50))

            if any(logit > 0.5 for logit in self.logits_concat):
                self.viewer.blit(tick_icon, (800 + 125, 75))
            else:
                self.viewer.blit(cross_icon, (800 + 125, 75))
            for i_obj, object in enumerate(self.objects):
                object_surface = object.surface
                object_surface = pygame.transform.scale(object_surface, (80, 80))
                self.viewer.blit(object_surface, (900, 150 + 200 * i_obj))
                circle_img = pygame.Surface((20, 20))
                for x in self.circles:
                    pygame.draw.circle(circle_img, (255 - x[2], 255 - x[2], 255 - x[2]), (10, 10), 8)
                    circle_img.set_colorkey(0)
                    self.viewer.blit(circle_img, (860 + x[0], 255 + 200 * i_obj))
                # pygame.draw.rect(self.viewer, pygame.Color('white'), (880, 255 + 200*i_obj, 120,20))
                x = self.logits_concat[i_obj]

                pygame.draw.rect(self.viewer, pygame.Color('darkred'), (860 + int(x * 160), 252.5 + 200 * i_obj, 3, 25))

        # GRIPPER
        x, y = self.get_pixel_coordinates(self.agent_pos[0], self.agent_pos[1])
        # TODO don't load in rendering this is stupid
        size_gripper_pixels = 55
        size_gripper_closed_pixels = 45
        gripper_icon = pygame.image.load(self.params['img_path'] + 'hand_open.png')
        gripper_icon = pygame.transform.scale(gripper_icon, (size_gripper_pixels, size_gripper_pixels))
        closed_gripper_icon = pygame.image.load(self.params['img_path'] + 'hand_closed.png')
        closed_gripper_icon = pygame.transform.scale(closed_gripper_icon,
                                                     (size_gripper_closed_pixels, size_gripper_pixels))
        if self.gripper_state == 1:
            left = int(x - size_gripper_closed_pixels // 2)
            top = int(y - size_gripper_closed_pixels // 2)
            self.viewer.blit(closed_gripper_icon, (left, top))
        else:
            left = int(x - size_gripper_pixels // 2)
            top = int(y - size_gripper_pixels // 2)
            self.viewer.blit(gripper_icon, (left, top))

        # IMAGINATION BUBBLE
        if self.show_imagination_bubble:
            if self.first_action == False:
                txt_surface = FONT.render(goal_str, True, pygame.Color('black'))

                speech_bubble_icon = pygame.image.load(self.params['img_path'] + 'bubble.png')
                speech_bubble_icon = pygame.transform.scale(speech_bubble_icon,
                                                            (txt_surface.get_width() + 50, 120))
                off_set_bubble = int(1.2 * size_gripper_pixels)
                bubble_x = x - off_set_bubble // 2
                bubble_y = y - 2 * off_set_bubble
                self.viewer.blit(speech_bubble_icon, (bubble_x, bubble_y))
                self.viewer.blit(txt_surface, (bubble_x + 25, bubble_y + 20))

        if self.viz_data_collection:
            # KNOWN GOALS
            known_goals_txt = FONT.render('Known Goals', True, pygame.Color('darkblue'))
            known_goals_icon = pygame.image.load(self.params['img_path'] + 'known_goals_box.png')
            known_goals_icon = pygame.transform.scale(known_goals_icon,
                                                      (300, 35 + 25 * len(self.known_goals_descr)))
            self.viewer.blit(known_goals_icon, (50, 50))
            self.viewer.blit(known_goals_txt, (75, 60))
            for i, descr in enumerate(self.known_goals_descr):
                goal_txt_surface = FONT.render(descr, True, pygame.Color('black'))
                self.viewer.blit(goal_txt_surface, (100, 85 + 25 * i))

            if self.SP_feedback == True:
                # SOCIAL PEER
                SP_head_icon = pygame.image.load(self.params['img_path'] + 'SP_head.png')
                SP_head_icon = pygame.transform.scale(SP_head_icon, (80, 80))
                SP_x = 50
                SP_y = 700
                self.viewer.blit(SP_head_icon, (SP_x, SP_y))
                SP_txt_surface = FONT.render('You ' + 'g' + self.SP_goal_descr[1:], True, pygame.Color('black'))
                SP_bubble_icon = pygame.image.load(self.params['img_path'] + 'SP_bubble.png')
                SP_bubble_icon = pygame.transform.scale(SP_bubble_icon,
                                                        (SP_txt_surface.get_width() + 50, 80))
                self.viewer.blit(SP_bubble_icon, (SP_x + 70, SP_y - 25))
                self.viewer.blit(SP_txt_surface, (SP_x + 100, SP_y))

                ## KNOWN GOALS UPDATE
                if self.known_goals_update == True:
                    if self.SP_goal_descr not in self.known_goals_descr:
                        known_goals_icon = pygame.transform.scale(known_goals_icon,
                                                                  (300, 35 + 25 * (1 + len(
                                                                      self.known_goals_descr))))
                        self.viewer.blit(known_goals_icon, (50, 50))
                        self.viewer.blit(known_goals_txt, (75, 60))
                        for i, descr in enumerate(self.known_goals_descr):
                            goal_txt_surface = FONT.render(descr, True, pygame.Color('black'))
                            self.viewer.blit(goal_txt_surface, (100, 85 + 25 * i))

                        if len(self.known_goals_descr) > 1:
                            goal_txt_surface = FONT.render(self.SP_goal_descr, True, pygame.Color('black'))
                            self.viewer.blit(goal_txt_surface, (
                                100,
                                SP_y - int(self.progress_goal_move * (SP_y - 85 - 25 * (len(self.known_goals_descr))))))
                            print(self.progress_goal_move)
                        else:
                            goal_txt_surface = FONT.render(self.SP_goal_descr, True, pygame.Color('black'))
                            self.viewer.blit(goal_txt_surface,
                                             (100, SP_y - int(self.progress_goal_move * (SP_y - 100)) - 15))
        if self.display:
            pygame.display.update()
            pygame.time.wait(50)

    def set_SP_feedback(self, goal_descr):
        self.SP_feedback = True
        self.SP_goal_descr = goal_descr

    def update_known_goal_position(self, x):
        self.known_goals_update = True
        self.progress_goal_move = x / 10

    def update_known_goals_list(self):
        if self.SP_goal_descr not in self.known_goals_descr:
            self.known_goals_descr.append(self.SP_goal_descr)

    def set_logits_concat(self, logits_concats):
        self.logits_concat = logits_concats

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None
