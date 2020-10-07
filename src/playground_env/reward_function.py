from src.playground_env.extract_from_state import *
from src.playground_env.env_params import *
from src.playground_env.descriptions import train_descriptions, test_descriptions, extra_descriptions



def sample_descriptions_from_state(state, modes=MODES):

    descriptions = []

    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    # extract attributes of objects
    current_absolute_attributes, current_relative_attributes = get_attributes_from_state(current_state)

    if 'Gripper' in admissible_actions:
        # descriptions about the gripper
        agent_pos = current_state[:2]
        if agent_pos[0] < -0.05:
            descriptions.append('Go left')
        elif agent_pos[0] > 0.05:
            descriptions.append('Go right')
        if agent_pos[1] < -0.05:
            descriptions.append('Go bottom')
        elif agent_pos[1] > 0.05:
            descriptions.append('Go top')
        if agent_pos[0] < - 0.25:
            if agent_pos[1] < -0.25:
                descriptions.append('Go bottom left')
            elif agent_pos[1] > 0.25:
                descriptions.append('Go top left')
        elif agent_pos[0] > 0.25:
            if agent_pos[1] < -0.25:
                descriptions.append('Go bottom right')
            elif agent_pos[1] > 0.25:
                descriptions.append('Go top right')
        else:
            if agent_pos[1] < 0.25 and agent_pos[1] > -0.25 and agent_pos[0] < 0.25 and agent_pos[0] > -0.25:
                descriptions.append('Go center')

    # deal with Grasp
    # get grasped objects
    if 'Grasp' in admissible_actions:
        obj_id_grasped_current = get_grasped_obj_ids(current_state)
        verb = 'Grasp'
        for obj_id in obj_id_grasped_current:
            if 1 in modes:
                for attr in current_absolute_attributes[obj_id] + current_relative_attributes[obj_id]:
                    if attr in thing_sizes + thing_shades + thing_colors:
                        descriptions.append('{} any {} thing'.format(verb, attr))
                    else:
                        descriptions.append('{} any {}'.format(verb, attr))

            # add combination of attribute and object type
            if 2 in modes:
                # add combination of attribute and object type
                object_types = []
                for attr in current_absolute_attributes[obj_id]:
                    if attr in things + group_names:
                        object_types.append(attr)
                for obj_type in object_types:
                    for attr in current_absolute_attributes[obj_id]:
                        if attr not in all_non_attributes:
                            descriptions.append('{} {} {}'.format(verb, attr, obj_type))

    # get grasped objects
    if 'Grow' in admissible_actions:
        obj_grown = get_grown_obj_ids(initial_state, current_state)
        verb = 'Grow'
        for obj_id in obj_grown:
            if 1 in modes:
                for attr in current_absolute_attributes[obj_id] + current_relative_attributes[obj_id]:
                    if attr in thing_sizes + thing_shades + thing_colors:
                        descriptions.append('{} any {} thing'.format(verb, attr))
                    else:
                        descriptions.append('{} any {}'.format(verb, attr))

            # add combination of attribute and object type
            if 2 in modes:
                # add combination of attribute and object type
                object_types = []
                for attr in current_absolute_attributes[obj_id]:
                    if attr in things + group_names:
                        object_types.append(attr)
                for obj_type in object_types:
                    for attr in current_absolute_attributes[obj_id]:
                        if attr not in all_non_attributes:
                            descriptions.append('{} {} {}'.format(verb, attr, obj_type))

    # deal with furniture stuff
    if 'Grow' in admissible_actions:
        obj_on_supply = get_obj_ids_on_supply(initial_state, current_state)
        verb = 'Attempt grow'
        for obj_id in obj_on_supply:
            if 1 in modes:
                for attr in current_absolute_attributes[obj_id] + current_relative_attributes[obj_id]:
                    if attr in plants + furnitures + ['plant', 'furniture']:
                        descriptions.append('{} any {}'.format(verb, attr))

            # add combination of attribute and object type
            if 2 in modes:
                # add combination of attribute and object type
                object_types = []
                for attr in current_absolute_attributes[obj_id]:
                    if attr in things + group_names:
                        object_types.append(attr)
                for obj_type in object_types:
                    for attr in current_absolute_attributes[obj_id]:
                        if attr not in all_non_attributes:
                            if obj_type in plants + furnitures + ['plant', 'furniture']:
                                descriptions.append('{} {} {}'.format(verb, attr, obj_type))

    train_descr = []
    test_descr = []
    extra_descr = []
    for descr in descriptions:
        if descr in train_descriptions:
            train_descr.append(descr)
        if descr in test_descriptions:
            test_descr.append(descr)
        if descr in extra_descriptions:
            extra_descr.append(descr)
    return train_descr.copy(), test_descr.copy(), extra_descr.copy()

def get_reward_from_state(state, goal):
    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    # extract attributes of objects
    current_absolute_attributes, current_relative_attributes = get_attributes_from_state(current_state)

    words = goal.split(' ')
    reward = False

    # Deal with gripper goals
    if words[0] == 'Go':
        # descriptions about the gripper
        agent_pos = current_state[:2]
        if words[1] == 'top':
            if len(words) > 2:
                if agent_pos[1] > 0.25:
                    if agent_pos[0] < -0.25 and words[2] == 'left':
                        return True
                    elif agent_pos[0] > 0.25 and words[2] == 'right':
                        return True
                    else:
                        return False
                else:
                    return False
            elif agent_pos[1] > 0.05:
                return True
            else:
                return False
        elif words[1] == 'bottom':
            if len(words) > 2:
                if agent_pos[1] < -0.25:
                    if agent_pos[0] < -0.25 and words[2] == 'left':
                        return True
                    elif agent_pos[0] > 0.25 and words[2] == 'right':
                        return True
                    else:
                        return False
                else:
                    return False
            elif agent_pos[1] < -0.05:
                return True
            else:
                return False
        elif words[1] == 'left':
            if len(words) > 2:
                if agent_pos[0] < -0.25:
                    if agent_pos[1] < -0.25 and words[2] == 'bottom':
                        return True
                    elif agent_pos[1] > 0.25 and words[2] == 'top':
                        return True
                    else:
                        return False
                else:
                    return False
            elif agent_pos[0] < -0.05:
                return True
            else:
                return False
        elif words[1] == 'right':
            if len(words) > 2:
                if agent_pos[0] > 0.25:
                    if agent_pos[1] < -0.25 and words[2] == 'bottom':
                        return True
                    elif agent_pos[1] > 0.25 and words[2] == 'top':
                        return True
                    else:
                        return False
                else:
                    return False
            elif agent_pos[0] > 0.05:
                return True
            else:
                return False
        elif words[1] == 'center':
            if agent_pos[1] < 0.25 and agent_pos[1] > -0.25 and agent_pos[0] < 0.25 and agent_pos[0] > -0.25:
                return True
            else:
                return False


    # Deal with grasped
    if words[0] == 'Grasp':
        # get grasped objects
        obj_id_grasped_current = get_grasped_obj_ids(current_state)
        # check whether the goal refers to any of the grasped objects
        for id in obj_id_grasped_current:
            # in that case only the attribute matters
            if words[1] == 'any':
                if words[2] in current_absolute_attributes[id] + current_relative_attributes[id]:
                        return True
            # in other cases, check whether the last word is an object type
            elif words[2] in current_absolute_attributes[id] and (words[2] in things or words[2] in group_names):
                # check that the attributes corresponds to that object
                if words[1] in current_absolute_attributes[id]:
                    return True

    # Deal with grow
    if words[0] == 'Grow':
        # get grasped objects
        obj_id_grown = get_grown_obj_ids(initial_state, current_state)
        # check whether the goal refers to any of the grasped objects
        for id in obj_id_grown:
            # in that case only the attribute matters
            if words[1] == 'any':
                if words[2] in current_absolute_attributes[id] + current_relative_attributes[id]:
                        return True
            # in other cases, check whether the last word is an object type
            elif words[2] in current_absolute_attributes[id] and (words[2] in things or words[2] in group_names):
                # check that the attributes corresponds to that object
                if words[1] in current_absolute_attributes[id]:
                    return True

    return reward





def supply_on_furniture(state, goal):
    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    # extract attributes of objects
    initial_absolute_attributes, initial_relative_attributes = get_attributes_from_state(initial_state)
    current_absolute_attributes, current_relative_attributes = get_attributes_from_state(current_state)

    words = goal.split(' ')
    reward = False


    # Deal with grow
    if words[0] == 'Grow':
        # get grasped objects
        obj_id_water = get_obj_ids_on_water(initial_state, current_state).tolist()
        obj_id_food = get_obj_ids_on_food(initial_state, current_state).tolist()
        # check whether the goal refers to any of the grasped objects
        for id in obj_id_water + obj_id_food:
            # in that case only the attribute matters
            if words[1] == 'any':
                if words[2] in current_absolute_attributes[id] + current_relative_attributes[id]:
                        return True
            # in other cases, check whether the last word is an object type
            elif words[2] in current_absolute_attributes[id] and (words[2] in things or words[2] in group_names):
                # check that the attributes corresponds to that object
                if words[1] in current_absolute_attributes[id]:
                    return True

    return reward

def food_on_furniture(state, goal):
    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    # extract attributes of objects
    initial_absolute_attributes, initial_relative_attributes = get_attributes_from_state(initial_state)
    current_absolute_attributes, current_relative_attributes = get_attributes_from_state(current_state)

    words = goal.split(' ')
    reward = False


    # Deal with grow
    if words[0] == 'Grow':
        # get grasped objects
        obj_id_food = get_obj_ids_on_food(initial_state, current_state).tolist()
        # check whether the goal refers to any of the grasped objects
        for id in obj_id_food:
            # in that case only the attribute matters
            if words[1] == 'any':
                if words[2] in current_absolute_attributes[id] + current_relative_attributes[id]:
                        return True
            # in other cases, check whether the last word is an object type
            elif words[2] in current_absolute_attributes[id] and (words[2] in things or words[2] in group_names):
                # check that the attributes corresponds to that object
                if words[1] in current_absolute_attributes[id]:
                    return True

    return reward


def water_on_furniture(state, goal):
    current_state = state[:len(state) // 2]
    initial_state = current_state - state[len(state) // 2:]
    assert len(current_state) == len(initial_state)

    # extract attributes of objects
    initial_absolute_attributes, initial_relative_attributes = get_attributes_from_state(initial_state)
    current_absolute_attributes, current_relative_attributes = get_attributes_from_state(current_state)

    words = goal.split(' ')
    reward = False


    # Deal with grow
    if words[0] == 'Grow':
        # get grasped objects
        obj_id_water = get_obj_ids_on_water(initial_state, current_state).tolist()
        # check whether the goal refers to any of the grasped objects
        for id in obj_id_water:
            # in that case only the attribute matters
            if words[1] == 'any':
                if words[2] in current_absolute_attributes[id] + current_relative_attributes[id]:
                        return True
            # in other cases, check whether the last word is an object type
            elif words[2] in current_absolute_attributes[id] and (words[2] in things or words[2] in group_names):
                # check that the attributes corresponds to that object
                if words[1] in current_absolute_attributes[id]:
                    return True

    return reward




