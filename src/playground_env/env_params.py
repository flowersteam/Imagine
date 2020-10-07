import numpy as np


N_OBJECTS_IN_SCENE = 3
ENV_ID = 'big'
admissible_actions = ['Gripper', 'Grasp', 'Grow']
n_inds_before_obj_inds = 3  #

MODES = [1, 2]

controllers = ['animal_switch', 'light_controller', 'tv_button']
furnitures = ['door', 'chair', 'desk', 'lamp', 'table']
if 'big' in ENV_ID:
    furnitures += ['cupboard', 'sink', 'window', 'sofa', 'carpet']
plants = ['cactus', 'carnivorous', 'flower', 'tree', 'bush']
if 'big' in ENV_ID:
    plants += ['grass', 'algae', 'tea', 'rose', 'bonsai']
animals = ['dog', 'cat', 'cameleon', 'human', 'fly']
if 'big' in ENV_ID:
    animals += ['parrot', 'mouse', 'lion', 'pig', 'cow']
living_things = animals + plants
supply = ['food', 'water']
things = living_things + furnitures + supply
group_names = ['animal', 'plant', 'living_thing', 'furniture', 'supply']
groups = [animals, plants, living_things, furnitures, supply]
all_non_attributes = ['animal', 'plant', 'living_thing', 'furniture', 'supply', 'controller', 'thing'] + living_things + supply + controllers

thing_colors = ['red', 'blue', 'green']
thing_shades = ['light', 'dark']
thing_sizes = ['big', 'small']

if 'plant' not in ENV_ID:
    words_to_remove_from_train = ['red tree', 'green dog', 'blue door'] + \
                                 ['flower'] + \
                                 ['Grasp {} animal'.format(c) for c in thing_colors + ['any']] +  \
                                 ['Grow {} {}'.format(c, p) for c in thing_colors + ['any'] for p in plants + ['plant', 'living_thing']] + \
                                 ['Grasp {} fly'.format(c) for c in thing_colors + ['any']]


n_things = len(things)
n_things_combinatorial = things * len(thing_colors) * len(thing_shades) * len(thing_sizes)

DIM_OBJ = n_things + 7
# get indices of attributes in object feature vector
color_inds = np.arange(n_things + 3, n_things + 6)
size_inds = np.array(n_things + 2)
position_inds = np.arange(n_things, n_things + 2)
type_inds = np.arange(0, n_things)

min_max_sizes = [[0.2, 0.25], [0.25, 0.3]]
GRIPPER_SIZE = 0.05
EPSILON = 0.3  # epsilon for initial positions
SCREEN_SIZE = 800
RATIO_SIZE = int(SCREEN_SIZE * 2/3 / 2)
NEXT_TO_EPSILON = 0.3

ATTRIBUTE_LIST = ['color', 'category', 'type']