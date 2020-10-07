from src.playground_env.env_params import *
from src.playground_env.color_generation import *



def get_object_type_and_categories(features):
    type_encoding = features[type_inds]
    ind = type_encoding.tolist().index(1)
    type_obj = things[ind]
    out = []
    if 'type' in ATTRIBUTE_LIST:
        out.append(type_obj)
    if 'category' in ATTRIBUTE_LIST:
        if type_obj in living_things:
            out.append('living_thing')
        if type_obj in supply:
            out.append('supply')
        if type_obj in animals:
            out.append('animal')
        if type_obj in plants:
            out.append('plant')
        if type_obj in controllers:
            out.append('controller')
        if type_obj in furnitures:
            out.append('furniture')
    return out


def get_object_color_and_shade(features, color=None, shade=None):
    rgb = features[color_inds]
    for c in ['blue', 'green', 'red', 'dark']:
        for s in ['light', 'dark']:
            color_class = Color(c, s)
            if color_class.contains(rgb):
                out = []
                if 'color' in ATTRIBUTE_LIST or color:
                    out.append(c)
                if 'shape' in ATTRIBUTE_LIST or shade:
                    out.append(s)
                return out
    raise NotImplementedError('Color not found')


def get_size(features):
    if 'size' in ATTRIBUTE_LIST:
        size = features[size_inds]
        if size < min_max_sizes[0][1]:
            return ['small']
        else:
            return ['big']
    else:
        return []


def get_darkest(all_features):
    shades = []
    for feature in all_features:
        shades.append(feature[color_inds].mean())
    min_val = np.min(shades)
    obj_id = np.atleast_1d(np.argwhere(shades == min_val).squeeze()).tolist()
    return obj_id


def get_lightest(all_features):
    shades = []
    for feature in all_features:
        shades.append(feature[color_inds].mean())
    max_val = np.max(shades)
    obj_id = np.atleast_1d(np.argwhere(shades == max_val).squeeze()).tolist()
    return obj_id


def get_biggest(all_features):
    sizes = []
    for feature in all_features:
        sizes.append(feature[size_inds])
    max_val = np.max(sizes)
    obj_id = np.atleast_1d(np.argwhere(sizes == max_val).squeeze()).tolist()
    return obj_id


def get_smallest(all_features):
    sizes = []
    for feature in all_features:
        sizes.append(feature[size_inds])
    min_val = np.min(sizes)
    obj_id = np.atleast_1d(np.argwhere(sizes == min_val).squeeze()).tolist()
    return obj_id


def get_absolute_location(features):
    position = features[position_inds]
    attr = []
    if 'absolute_location' in ATTRIBUTE_LIST:
        if position[0] < 0:
            attr.append('left')
        else:
            attr.append('right')
        if position[1] < 0:
            attr.append('bottom')
        else:
            attr.append('top')
    return attr


def get_relative_location(all_features):
    positions = []
    for feature in all_features:
        positions.append(feature[position_inds])
    sizes = []
    for feature in all_features:
        sizes.append(feature[size_inds])
    close_pairs = []
    for obj_id_1 in range(N_OBJECTS_IN_SCENE):
        for obj_id_2 in range(obj_id_1 + 1, N_OBJECTS_IN_SCENE):
            if np.linalg.norm(positions[obj_id_1] - positions[obj_id_2]) < NEXT_TO_EPSILON:
                close_pairs.append((obj_id_1, obj_id_2))
    return close_pairs


def get_attributes_from_state(state):
    all_objects_absolute_attributes = []

    # get absolute attributes for all objects
    for i_obj in range(N_OBJECTS_IN_SCENE):
        absolute_obj_attributes = []
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))

        features = state[inds]
        absolute_obj_attributes += get_object_type_and_categories(features)
        absolute_obj_attributes += get_object_color_and_shade(features)
        absolute_obj_attributes += get_size(features)
        absolute_obj_attributes += get_absolute_location(features)
        all_objects_absolute_attributes.append(absolute_obj_attributes)

    # get relative attributes for all objects
    all_objects_relative_attributes = [[] for _ in range(N_OBJECTS_IN_SCENE)]
    all_features = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))

        all_features.append(state[inds])
    if 'relative_shade' in ATTRIBUTE_LIST:
        ind_darkest = get_darkest(all_features)
        ind_lightest = get_lightest(all_features)
        ind_smallest = get_smallest(all_features)
        ind_biggest = get_biggest(all_features)
        for i in ind_darkest:
            all_objects_relative_attributes[i].append('darkest')
        for i in ind_lightest:
            all_objects_relative_attributes[i].append('lightest')
        for i in ind_biggest:
            all_objects_relative_attributes[i].append('biggest')
        for i in ind_smallest:
            all_objects_relative_attributes[i].append('smallest')

    if 'relative_position' in ATTRIBUTE_LIST:
        pairs = get_relative_location(all_features)
        for p in pairs:
            att_1 = all_objects_absolute_attributes[p[1]].copy()
            att_0 = all_objects_absolute_attributes[p[0]].copy()
            for att in att_1:
                if att not in ['left', 'right', 'top', 'bottom']:
                    all_objects_relative_attributes[p[0]].append('next_to_' + att)
            for att in att_0:
                if att not in ['left', 'right', 'top', 'bottom']:
                    all_objects_relative_attributes[p[1]].append('next_to_' + att)

    return all_objects_absolute_attributes, all_objects_relative_attributes

def get_touched_obj_ids(state):
    hand_position = state[:2]
    hand_size = GRIPPER_SIZE
    touched_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))

        features = state[inds]
        position = features[position_inds]
        size = features[size_inds]
        if np.linalg.norm(position - hand_position) < (hand_size + size) / 2:
            touched_ids.append(i_obj)
    return np.array(touched_ids)

def get_grasped_obj_ids(state):
    grasped_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        features = state[inds]
        if features[-1] == 1:
            grasped_ids.append(i_obj)
    return np.array(grasped_ids)

def get_grown_obj_ids(initial_state, state):
    grown_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        size = state[inds][size_inds]
        initial_size = initial_state[inds][size_inds]
        if size > initial_size + 0.0001:
            grown_ids.append(i_obj)
    return np.array(grown_ids)

def get_transformed_obj_ids(initial_state, state):
    transformed_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        color = state[inds][color_inds]
        initial_color = initial_state[inds][color_inds]
        if not np.all(color == initial_color):
            transformed_ids.append(i_obj)
    return np.array(transformed_ids)

def get_obj_ids_on_food(initial_state, state):
    on_food_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        pos_obj = state[inds_obj][position_inds]
        size_obj = state[inds_obj][size_inds]

        for i_other_obj in range(N_OBJECTS_IN_SCENE):
            if i_other_obj != i_obj:
                inds_other_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_other_obj, n_inds_before_obj_inds + DIM_OBJ * (i_other_obj + 1))
                type_other_obj = get_object_type_and_categories(state[inds_other_obj])[0]
                if type_other_obj == 'food':
                    pos_other_obj = state[inds_other_obj][position_inds]
                    size_other_obj = state[inds_other_obj][size_inds]
                    if np.linalg.norm(pos_obj - pos_other_obj) < (size_obj+ size_other_obj) / 2:
                        on_food_ids.append(i_obj)
    return np.array(on_food_ids)

def get_obj_ids_on_water(initial_state, state):
    on_water_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        pos_obj = state[inds_obj][position_inds]
        size_obj = state[inds_obj][size_inds]

        for i_other_obj in range(N_OBJECTS_IN_SCENE):
            if i_other_obj != i_obj:
                inds_other_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_other_obj, n_inds_before_obj_inds + DIM_OBJ * (i_other_obj + 1))
                type_other_obj = get_object_type_and_categories(state[inds_other_obj])[0]
                if type_other_obj == 'water':
                    pos_other_obj = state[inds_other_obj][position_inds]
                    size_other_obj = state[inds_other_obj][size_inds]
                    if np.linalg.norm(pos_obj - pos_other_obj) < (size_obj+ size_other_obj) / 2:
                        on_water_ids.append(i_obj)
    return np.array(on_water_ids)

def get_obj_ids_on_supply(initial_state, state):
    on_supply_ids = []
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        pos_obj = state[inds_obj][position_inds]
        size_obj = state[inds_obj][size_inds]

        for i_other_obj in range(N_OBJECTS_IN_SCENE):
            if i_other_obj != i_obj:
                inds_other_obj = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_other_obj, n_inds_before_obj_inds + DIM_OBJ * (i_other_obj + 1))
                type_other_obj = get_object_type_and_categories(state[inds_other_obj])[0]
                if type_other_obj in ['water', 'food']:
                    pos_other_obj = state[inds_other_obj][position_inds]
                    size_other_obj = state[inds_other_obj][size_inds]
                    if np.linalg.norm(pos_obj - pos_other_obj) < (size_obj+ size_other_obj) / 2:
                        on_supply_ids.append(i_obj)
    return np.array(on_supply_ids)

def get_shifted_obs_ids(initial_state, state):
    eps = 0.15
    left = []
    right = []
    higher = []
    lower =[]
    for i_obj in range(N_OBJECTS_IN_SCENE):
        inds = np.arange(n_inds_before_obj_inds + DIM_OBJ * i_obj, n_inds_before_obj_inds + DIM_OBJ * (i_obj + 1))
        obj_position = state[inds][position_inds]
        initial_obj_position = initial_state[inds][position_inds]
        if obj_position[0] > initial_obj_position[0] + eps:
            right.append(i_obj)
        elif obj_position[0] < initial_obj_position[0] - eps:
            left.append(i_obj)
        if obj_position[1] > initial_obj_position[1] + eps:
            higher.append(i_obj)
        elif obj_position[1] < initial_obj_position[1] - eps:
            lower.append(i_obj)

    return left, right, higher, lower


def has_key(state):
    return state[-2] == 1

def has_door(state):
    return state[-1] == 1