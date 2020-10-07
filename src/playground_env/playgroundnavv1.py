from __future__ import division
import gym
from gym import spaces

from src.playground_env.objects import *
from src.playground_env.reward_function import *


class PlayGroundNavigationV1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    '''
        Playground Environement:
        set reward_screen to True to visualize modular reward function predictions
        set viz_data_collection to True to visualize Social Partner interactions 
    '''
    def __init__(self,
                 n_timesteps=50,
                 epsilon=0.1,
                 random_init=False,
                 render=False,
                 human=False,
                 reward_screen=False,
                 viz_data_collection=False
                 ):

        self.viz_data_collection = viz_data_collection
        self.reward_screen = reward_screen
        self.first_action = False
        self.SP_feedback = False
        self.known_goals_update = False
        self.known_goals_descr = []
        self.logits_concat = [0, 0, 0]

        self.circles = [[x * 3, 200, x * 4] for x in range(50)]

        self.random_init = random_init
        self.epsilon = 1
        self.n_timesteps = n_timesteps
        self.human = human
        self.render_mode = render

        self.n_act = 3
        self.nb_obj = N_OBJECTS_IN_SCENE
        self.dim_obj = DIM_OBJ
        self.inds_objs = [np.arange(n_inds_before_obj_inds + self.dim_obj * i_obj,
                                    n_inds_before_obj_inds + self.dim_obj * (i_obj + 1)) for i_obj in
                          range(self.nb_obj)]
        self.n_half_obs = self.nb_obj * self.dim_obj + n_inds_before_obj_inds
        self.inds_grasped_obj = np.array([])
        self.n_obs = self.n_half_obs * 2

        # We define the spaces
        self.action_space = spaces.Box(low=-np.ones(self.n_act),
                                       high=np.ones(self.n_act),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.ones(self.n_obs),
                                            high=np.ones(self.n_obs),
                                            dtype=np.float32)

        # Maint agent
        self.pos_step_size = 0.15
        self.pos_init = [0., 0.]
        self.pos_init_random_random = 0.6

        if self.render_mode:
            pygame.init()
            if self.reward_screen:
                self.viewer = pygame.display.set_mode((SCREEN_SIZE + 300, SCREEN_SIZE))
            else:
                self.viewer = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            self.viewer_started = False
        self.background = None

        self.reset()

        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.initial_observation = None
        self.done = None

        self.info = dict(is_success=0)

    def change_n_objs(self, n_objs):
        self.nb_obj = n_objs
        self.inds_objs = [np.arange(n_inds_before_obj_inds + self.dim_obj * i_obj,
                                    n_inds_before_obj_inds + self.dim_obj * (i_obj + 1)) for i_obj in
                          range(self.nb_obj)]
        self.n_half_obs = self.nb_obj * self.dim_obj + n_inds_before_obj_inds
        self.n_obs = self.n_half_obs * 2

    def set_state(self, state):

        assert state.size == self.n_half_obs, 'N_OBJECTS_IN_SCENE is not right'
        current_state = state[:state.shape[0] // 2]
        self.pos = current_state[:2]
        self.gripper_state = current_state[2]

        self.initial_observation = current_state - state[state.shape[0] // 2:]
        obj_features = []
        for i_obj in range(self.nb_obj):
            obj_features.append(current_state[self.inds_objs[i_obj]])
        self.set_objects(obj_features)
        self.object_grasped = np.any(np.array(obj_features)[:, -1] == 1)

    def set_objects(self, features):
        objects = []
        objects_ids = []
        objects_types = []
        for obj_feat in features:
            type = get_object_type_and_categories(obj_feat)[0]
            color, shade = get_object_color_and_shade(obj_feat, True, True)
            size = get_size(obj_feat)[0]
            obj_id = get_obj_identifier(type, color, shade, size)
            if obj_id not in objects_ids:
                objects.append(build_object(type, color, shade, size, len(objects), objects, self.render_mode))
                objects_ids.append(obj_id)
                objects_types.append(type)
                objects[-1].update_color(color, shade, obj_feat[color_inds])
                objects[-1].update_size(obj_feat[size_inds])
                objects[-1].update_position(obj_feat[position_inds])

        self.objects = objects
        self.objects_ids = objects_ids
        self.objects_types = objects_types
        for obj in self.objects:
            obj.give_ref_to_obj_list(self.objects)
            obj.update_all_attributes()

    def reset_with_goal(self, goal_str):
        words = goal_str.split(' ')
        objs = []

        if words[0] == 'Grow':
            obj = dict(type=None,
                       category=None,
                       color=None,
                       size=None,
                       shade=None)
            if words[2] in thing_colors + thing_shades + thing_sizes and words[3] == 'thing':
                obj['category'] = np.random.choice(['animal', 'plant'])
                if words[2] in thing_colors:
                    obj['color'] = words[2]
                elif words[2] in thing_sizes:
                    obj['size'] = words[2]
                elif words[2] in thing_shades:
                    obj['shade'] = words[2]
            else:
                for w in words:
                    if w in things:
                        obj['type'] = w
                    elif w in group_names:
                        obj['category'] = w
                    elif w in thing_colors:
                        obj['color'] = w
                    elif w in thing_sizes:
                        obj['size'] = w
                    elif w in thing_shades:
                        obj['shade'] = w
            objs.append(obj)
            if obj['category'] in ['living_thing', 'plant'] or obj['type'] in plants:
                obj = dict(type='water',
                           category='supply',
                           color=None,
                           size=None,
                           shade=None)
            else:
                obj = dict(type=None,
                           category='supply',
                           color='green',
                           size=None,
                           shade=None)
            objs.append(obj.copy())

        else:
            obj = dict(type=None,
                       category=None,
                       color=None,
                       size=None,
                       shade=None)
            for w in words:
                if w in things:
                    obj['type'] = w
                elif w in group_names:
                    obj['category'] = w
                elif w in thing_colors:
                    obj['color'] = w
                elif w in thing_sizes:
                    obj['size'] = w
                elif w in thing_shades:
                    obj['shade'] = w
            objs.append(obj)

        return self.reset_scene(objs)

    def reset(self):
        self.first_action = False
        self.logits_concat = [0, 0, 0]
        self.SP_feedback = False
        self.known_goals_update = False
        return self.reset_scene()

    def reset_scene(self, objects=None):

        self.pos = self.pos_init

        if self.random_init:
            self.pos += np.random.uniform(-self.pos_init_random_random, self.pos_init_random_random, 2)
            self.gripper_state = np.random.choice([-1, 1])  # self.arm_rest_state[3]
        else:
            self.gripper_state = -1

        self.objects, self.objects_ids, self.objects_types = self.sample_objects(objects)

        self.object_grasped = False

        # update attributes
        for obj in self.objects:
            obj.give_ref_to_obj_list(self.objects)
            obj.update_all_attributes()

        # Print objects
        for obj in self.objects:
            self.object_grasped = obj.update_state(self.pos,
                                                   self.gripper_state > 0,
                                                   self.objects,
                                                   self.object_grasped,
                                                   np.zeros([10]))


        # construct vector of observations
        self.observation = np.zeros(self.n_obs)
        self.observation[:self.n_half_obs] = self.observe()
        self.initial_observation = self.observation[:self.n_half_obs].copy()
        self.steps = 0
        self.done = False
        return self.observation.copy()

    def sample_objects(self, objects_to_add):
        objects = []
        objects_ids = []
        objects_types = []
        if objects_to_add is not None:
            for object in objects_to_add:
                if object['type'] is not None:
                    type = object['type']
                elif object['category'] is not None:
                    type = np.random.choice(groups[group_names.index(object['category'])])
                else:
                    type = np.random.choice(things)
                if object['size'] is not None:
                    size = object['size']
                else:
                    size = np.random.choice(thing_sizes)
                if object['color'] is not None:
                    color = object['color']
                else:
                    color = np.random.choice(thing_colors)
                if object['shade'] is not None:
                    shade = object['shade']
                else:
                    shade = np.random.choice(thing_shades)
                obj_id = get_obj_identifier(type, color, shade, size)
                if obj_id not in objects_ids:
                    objects.append(build_object(type, color, shade, size, len(objects), objects, self.render_mode))
                    objects_ids.append(obj_id)
                    objects_types.append(type)

        while len(objects) < self.nb_obj:
            type = np.random.choice(things)
            color = np.random.choice(thing_colors)
            shade = np.random.choice(thing_shades)
            size = np.random.choice(thing_sizes)
            obj_id = get_obj_identifier(type, color, shade, size)
            if obj_id not in objects_ids:
                objects.append(build_object(type, color, shade, size, len(objects), objects, self.render_mode))
                objects_ids.append(obj_id)
                objects_types.append(type)

        return objects, objects_ids, objects_types

    def observe(self):

        obj_features = np.array([obj.features for obj in self.objects]).flatten()
        obs = np.concatenate([self.pos,  # size 2
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

        # use fake actions so as to reuse objects from v0
        fake_action = np.zeros([10])
        fake_action[:3] = action[:3].copy()  # give gripper actions

        # Update the arm position
        self.pos = np.clip(self.pos + action[:2] * self.pos_step_size, -1.2, 1.2)

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
            self.object_grasped = obj.update_state(self.pos,
                                                   self.gripper_state > 0,
                                                   self.objects,
                                                   self.object_grasped,
                                                   fake_action)

        for obj in self.objects:
            obj.update_all_attributes()


        self.observation[:self.n_half_obs] = self.observe()
        self.observation[self.n_half_obs:] = self.observation[:self.n_half_obs] - self.initial_observation

        self.steps += 1
        if self.steps == self.n_timesteps:
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

            cross_icon = pygame.image.load(IMAGE_PATH + 'cross.png')
            cross_icon = pygame.transform.scale(cross_icon, (50, 50)).convert_alpha()

            tick_icon = pygame.image.load(IMAGE_PATH + 'tick.png')
            tick_icon = pygame.transform.scale(tick_icon, (50, 50)).convert_alpha()

            if any(logit > 0.5 for logit in self.logits_concat):
                self.viewer.blit(tick_icon, (800 + 125, 75))
            else:
                self.viewer.blit(cross_icon, (800 + 125, 75))
            for i_obj, object in enumerate(self.objects):
                object_surface = object.surface
                object_surface = pygame.transform.scale(object_surface, (80, 80)).convert_alpha()
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
        x, y = get_pixel_coordinates(self.pos[0], self.pos[1])
        # TODO don't load in rendering this is stupid
        size_gripper_pixels = 55
        size_gripper_closed_pixels = 45
        gripper_icon = pygame.image.load(IMAGE_PATH + 'hand_open.png')
        gripper_icon = pygame.transform.scale(gripper_icon, (size_gripper_pixels, size_gripper_pixels)).convert_alpha()
        closed_gripper_icon = pygame.image.load(IMAGE_PATH + 'hand_closed.png')
        closed_gripper_icon = pygame.transform.scale(closed_gripper_icon,
                                                     (size_gripper_closed_pixels, size_gripper_pixels)).convert_alpha()
        if self.gripper_state == 1:
            left = int(x - size_gripper_closed_pixels // 2)
            top = int(y - size_gripper_closed_pixels // 2)
            self.viewer.blit(closed_gripper_icon, (left, top))
        else:
            left = int(x - size_gripper_pixels // 2)
            top = int(y - size_gripper_pixels // 2)
            self.viewer.blit(gripper_icon, (left, top))

        # IMAGINATION BUBBLE
        if self.first_action == False:
            txt_surface = FONT.render(goal_str, True, pygame.Color('black'))

            speech_bubble_icon = pygame.image.load(IMAGE_PATH + 'bubble.png')
            speech_bubble_icon = pygame.transform.scale(speech_bubble_icon,
                                                        (txt_surface.get_width() + 50, 120)).convert_alpha()
            off_set_bubble = int(1.2 * size_gripper_pixels)
            bubble_x = x - off_set_bubble // 2
            bubble_y = y - 2 * off_set_bubble
            self.viewer.blit(speech_bubble_icon, (bubble_x, bubble_y))
            self.viewer.blit(txt_surface, (bubble_x + 25, bubble_y + 20))

        if self.viz_data_collection:
            # KNOWN GOALS
            known_goals_txt = FONT.render('Known Goals', True, pygame.Color('darkblue'))
            known_goals_icon = pygame.image.load(IMAGE_PATH + 'known_goals_box.png')
            known_goals_icon = pygame.transform.scale(known_goals_icon,
                                                      (300, 35 + 25 * len(self.known_goals_descr))).convert_alpha()
            self.viewer.blit(known_goals_icon, (50, 50))
            self.viewer.blit(known_goals_txt, (75, 60))
            for i, descr in enumerate(self.known_goals_descr):
                goal_txt_surface = FONT.render(descr, True, pygame.Color('black'))
                self.viewer.blit(goal_txt_surface, (100, 85 + 25 * i))

            if self.SP_feedback == True:
                # SOCIAL PEER
                SP_head_icon = pygame.image.load(IMAGE_PATH + 'SP_head.png')
                SP_head_icon = pygame.transform.scale(SP_head_icon, (80, 80)).convert_alpha()
                SP_x = 50
                SP_y = 700
                self.viewer.blit(SP_head_icon, (SP_x, SP_y))
                SP_txt_surface = FONT.render('You ' + 'g' + self.SP_goal_descr[1:], True, pygame.Color('black'))
                SP_bubble_icon = pygame.image.load(IMAGE_PATH + 'SP_bubble.png')
                SP_bubble_icon = pygame.transform.scale(SP_bubble_icon,
                                                        (SP_txt_surface.get_width() + 50, 80)).convert_alpha()
                self.viewer.blit(SP_bubble_icon, (SP_x + 70, SP_y - 25))
                self.viewer.blit(SP_txt_surface, (SP_x + 100, SP_y))

                ## KNOWN GOALS UPDATE
                if self.known_goals_update == True:
                    if self.SP_goal_descr not in self.known_goals_descr:
                        known_goals_icon = pygame.transform.scale(known_goals_icon,
                                                                  (300, 35 + 25 * (1 + len(
                                                                      self.known_goals_descr)))).convert_alpha()
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
