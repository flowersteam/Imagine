import os

from src.playground_env.color_generation import *


def get_env_params(max_nb_objects=3,
                   admissible_actions=('Move', 'Grasp', 'Grow'),
                   admissible_attributes=('colors', 'categories', 'types'),
                   min_max_sizes=((0.2, 0.25), (0.25, 0.3)),
                   agent_size=0.05,
                   epsilon_initial_pos=0.3,
                   screen_size=800,
                   next_to_epsilon=0.3,
                   attribute_combinations=False,
                   obj_size_update=0.04,
                   render_mode=True
                   ):
    """
    Builds the set of environment parameters, and the set of function to extract information from the state.

    Parameters
    ----------
    max_nb_objects: int
         Maximum number of objects in the scene (effective number if it's not random).
    admissible_actions: tuple of str
        which types of actions are admissible
    admissible_attributes: tuple of str
        All admissible attributes, should be included in ('colors', 'categories', 'types', 'relative_sizes', 'shades', 'relative_shades', 'sizes', 'relative_positions')
    min_max_sizes: tuple of tuples
        Min and max sizes for the small and big objects respectively.
    agent_size: float
        Size of the agent.
    epsilon_initial_pos: float
        Range of initial position around origin.
    screen_size: int
        Screen size in pixels.
    next_to_epsilon: float
        Define which area corresponds to 'next to'.
    attribute_combinations: Bool
        Whether attributes should include combinations of two attributes.
    obj_size_update: float
        By how much should be updated the size of objects when the agent grows them.
    render_mode: Bool
        Whether to render the environment.

    Returns
    -------
    params: dict
    """

    # list objects and categories
    furnitures = ('door', 'chair', 'desk', 'lamp', 'table', 'cupboard', 'sink', 'window', 'sofa', 'carpet')
    plants = ('cactus', 'carnivorous', 'flower', 'tree', 'bush', 'grass', 'algae', 'tea', 'rose', 'bonsai')
    animals = ('dog', 'cat', 'chameleon', 'human', 'fly', 'parrot', 'mouse', 'lion', 'pig', 'cow')
    supplies = ('food', 'water')
    living_things = animals + plants
    categories = dict(animal=animals,
                      plant=plants,
                      furniture=furnitures,
                      living_thing=living_things,
                      supply=supplies)
    # List types
    types = ()
    for k_c in categories.keys():
        types += categories[k_c]
    types = tuple(sorted(list(set(types)))) # filters doubles, when some categories include others.
    nb_types = len(types)

    # List attributes
    colors = ('red', 'blue', 'green')
    shades = ('light', 'dark')
    sizes = ('big', 'small')
    positions = ('left', 'right', 'top', 'bottom')
    relative_shades = ('lightest', 'darkest')
    relative_sizes = ('smallest', 'biggest')
    relative_positions = ('leftest', 'rightest', 'highest', 'lowest')
    attributes = dict(types=types,
                      categories=tuple(categories.keys()),
                      colors=colors,
                      shades=shades,
                      sizes=sizes,
                      positions=positions,
                      relative_shades=relative_shades,
                      relative_sizes=relative_sizes,
                      relative_positions=relative_positions)

    # Get the list of admissible attributes
    name_attributes = ()
    adjective_attributes = ()
    for att_type in attributes.keys():
        if att_type in admissible_attributes:
            if att_type in ('types', 'categories'):
                name_attributes += attributes[att_type]
            else:
                adjective_attributes += attributes[att_type]


    for att in admissible_attributes:
        assert att in attributes.keys()

    # This defines the list of occurrences that should belong to the test set. All descriptions that contain them belong to the testing set.
    words_test_set_def = ('red tree', 'green dog', 'blue door') + \
                         ('flower',) + \
                         tuple('Grasp {} animal'.format(c) for c in colors + ('any',)) + \
                         tuple('Grow {} {}'.format(c, p) for c in colors + ('any',) for p in plants + ('plant', 'living_thing')) + \
                         tuple('Grasp {} fly'.format(c) for c in colors + ('any',))


    # get indices of attributes in object feature vector
    dim_body_features = 3
    agent_position_inds = np.arange(2)
    dim_obj_features = nb_types + 7
    type_inds = np.arange(0, nb_types)
    position_inds = np.arange(nb_types, nb_types + 2)
    size_inds = np.array(nb_types + 2)
    color_inds = np.arange(nb_types + 3, nb_types + 6)
    grasped_inds = np.array([nb_types + 6])

    params = dict(nb_types=nb_types,
                  max_nb_objects=max_nb_objects,
                  admissible_actions=admissible_actions,
                  admissible_attributes=admissible_attributes,
                  dim_body_features=dim_body_features,
                  agent_position_inds=agent_position_inds,
                  grasped_inds=grasped_inds,
                  attributes=attributes,
                  categories=categories,
                  name_attributes=name_attributes,
                  adjective_attributes=adjective_attributes,
                  words_test_set_def=words_test_set_def,
                  dim_obj_features=dim_obj_features,  # one-hot of things, 2D position, size, rgb code, grasped Boolean
                  color_inds=color_inds,
                  size_inds=size_inds,
                  position_inds=position_inds,
                  type_inds=type_inds,
                  min_max_sizes=min_max_sizes,
                  agent_size=agent_size,
                  epsilon_initial_pos=epsilon_initial_pos,
                  screen_size=screen_size,
                  ratio_size=int(screen_size / 2.4),
                  next_to_epsilon=next_to_epsilon,
                  attribute_combinations=attribute_combinations,
                  obj_size_update=obj_size_update,
                  render_mode=render_mode
                  )

    # # # # # # # # # # # # # # #
    # Define extraction functions
    # # # # # # # # # # # # # # #

    # global extraction functions
    def count_objects(state):
        return (state.size - params['dim_body_features']) // params['dim_obj_features']

    def get_obj_features(state, i_obj):
        inds = np.arange(dim_body_features + dim_obj_features * i_obj, dim_body_features + dim_obj_features * (i_obj + 1))
        return state[inds]

    # Attribute extraction functions
    def get_obj_type(all_obj_features, i_obj):
        obj_features = all_obj_features[i_obj]
        type_encoding = obj_features[type_inds]
        ind = type_encoding.tolist().index(1)
        obj_type = types[ind]
        return [obj_type]

    def get_obj_cat(all_obj_features, i_obj):
        obj_type = get_obj_type(all_obj_features, i_obj)[0]
        cats = []
        for k in categories.keys():
            if obj_type in categories[k]:
                cats.append(k)
        return cats

    def get_obj_color(all_obj_features, i_obj):
        obj_features = all_obj_features[i_obj]
        rgb = obj_features[color_inds]
        for c in colors:
            for s in shades:
                color_class = Color(c, s)
                if color_class.contains(rgb):
                    return [c]
        raise ValueError

    def get_obj_shade(all_obj_features, i_obj):
        obj_features = all_obj_features[i_obj]
        rgb = obj_features[color_inds]
        for c in colors:
            for s in shades:
                color_class = Color(c, s)
                if color_class.contains(rgb):
                    return [s]
        raise ValueError

    def get_obj_size(all_obj_features, i_obj):
        obj_features = all_obj_features[i_obj]
        size = obj_features[size_inds]
        if size < min_max_sizes[0][1]:
            return ['small']
        else:
            return ['big']

    def get_darkest_obj_id(all_obj_features):
        # list of obj_features
        shades = np.array(tuple(feature[color_inds].mean() for feature in all_obj_features))
        return np.argwhere(shades == np.min(shades)).flatten()

    def get_lightest_obj_id(all_obj_features):
        # list of obj_features
        shades = np.array(tuple(feature[color_inds].mean() for feature in all_obj_features))
        return np.argwhere(shades == np.max(shades)).flatten()

    def get_obj_relative_shades(all_obj_features, i_obj):
        out = []
        if i_obj in get_darkest_obj_id(all_obj_features):
            out.append('darkest')
        if i_obj in get_lightest_obj_id(all_obj_features):
            out.append('lightest')
        return out

    def get_biggest_obj_id(all_obj_features):
        # list of obj_features
        sizes = np.array(tuple(feature[size_inds] for feature in all_obj_features))
        return np.argwhere(sizes == np.max(sizes)).flatten()

    def get_smallest_obj_id(all_obj_features):
        # list of obj_features
        sizes = np.array(tuple(feature[size_inds] for feature in all_obj_features))
        return np.argwhere(sizes == np.min(sizes)).flatten()

    def get_obj_relative_sizes(all_obj_features, i_obj):
        out = []
        if i_obj in get_biggest_obj_id(all_obj_features):
            out.append('biggest')
        if i_obj in get_smallest_obj_id(all_obj_features):
            out.append('smallest')
        return out

    def get_obj_position(all_obj_features, i_obj):
        obj_features = all_obj_features[i_obj]
        position = obj_features[position_inds]
        attr = []
        if position[0] < 0:
            attr.append('left')
        else:
            attr.append('right')
        if position[1] < 0:
            attr.append('bottom')
        else:
            attr.append('top')
        return attr

    def get_leftest_obj_id(all_obj_features):
        # list of obj_features
        x_position = np.array(tuple(feature[position_inds[0]] for feature in all_obj_features))
        return np.argwhere(x_position == np.min(x_position)).flatten()
    
    def get_rightest_obj_id(all_obj_features):
        # list of obj_features
        x_position = np.array(tuple(feature[position_inds[0]] for feature in all_obj_features))
        return np.argwhere(x_position == np.max(x_position)).flatten()

    def get_highest_obj_id(all_obj_features):
        # list of obj_features
        y_position = np.array(tuple(feature[position_inds[1]] for feature in all_obj_features))
        return np.argwhere(y_position == np.max(y_position)).flatten()
    
    def get_lowest_obj_id(all_obj_features):
        # list of obj_features
        y_position = np.array(tuple(feature[position_inds[1]] for feature in all_obj_features))
        return np.argwhere(y_position == np.min(y_position)).flatten()
    
    def get_relative_position(all_obj_features, i_obj):
        out = []
        if i_obj in get_leftest_obj_id(all_obj_features):
            out.append('leftest')
        if i_obj in get_rightest_obj_id(all_obj_features):
            out.append('rightest')
        if i_obj in get_highest_obj_id(all_obj_features):
            out.append('highest')
        if i_obj in get_lowest_obj_id(all_obj_features):
            out.append('lowest')
        return out

    get_attributes_functions = dict(relative_shades=get_obj_relative_shades,
                                    relative_sizes=get_obj_relative_sizes,
                                    relative_positions=get_relative_position,
                                    positions=get_obj_position,
                                    colors=get_obj_color,
                                    shades=get_obj_shade,
                                    sizes=get_obj_size,
                                    types=get_obj_type,
                                    categories=get_obj_cat)
    assert sorted(list(get_attributes_functions.keys())) == sorted(list(attributes))

    # List all attributes of all objects from state
    def get_attributes_from_state(state):
        assert state.ndim == 1
        nb_objs = count_objects(state)
        all_obj_features = [get_obj_features(state, i_obj) for i_obj in range(nb_objs)]

        # get attributes for all objects
        all_objects_attributes = []
        for i_obj in range(nb_objs):
            obj_attributes = []
            for k in admissible_attributes:
                obj_attributes += get_attributes_functions[k](all_obj_features, i_obj)
            all_objects_attributes.append(obj_attributes)

        return all_objects_attributes.copy()

    get_attributes_functions['all_attributes'] = get_attributes_from_state

    # Extract absolute position of the agent
    def get_agent_position_attributes(state):
        agent_pos = state[agent_position_inds]
        out = []
        if agent_pos[0] < -0.05:
            out.append('left')
        elif agent_pos[0] > 0.05:
            out.append('right')
        if agent_pos[1] < -0.05:
            out.append('bottom')
        elif agent_pos[1] > 0.05:
            out.append('top')
        if agent_pos[0] < - 0.25:
            if agent_pos[1] < -0.25:
                out.append('bottom left')
            elif agent_pos[1] > 0.25:
                out.append('top left')
        elif agent_pos[0] > 0.25:
            if agent_pos[1] < -0.25:
                out.append('bottom right')
            elif agent_pos[1] > 0.25:
                out.append('top right')
        else:
            if agent_pos[1] < 0.25 and agent_pos[1] > -0.25 and agent_pos[0] < 0.25 and agent_pos[0] > -0.25:
                out.append('center')

        return out.copy()

    # Extract interactions with objects (touched, grasped, grown)
    def get_touched_obj_ids(state):
        nb_objs = count_objects(state)
        agent_position = state[agent_position_inds]
        touched_ids = []
        for i_obj in range(nb_objs):
            obj_features = get_obj_features(state, i_obj)
            position = obj_features[position_inds]
            size = obj_features[size_inds]
            if np.linalg.norm(position - agent_position) < ((agent_size + size) / 2):
                touched_ids.append(i_obj)
        return np.array(touched_ids)

    def get_grasped_obj_ids(state):
        nb_objs = count_objects(state)
        grasped_ids = []
        for i_obj in range(nb_objs):
            obj_features = get_obj_features(state, i_obj)
            if obj_features[grasped_inds] == 1:
                grasped_ids.append(i_obj)
        return np.array(grasped_ids)

    def get_grown_obj_ids(initial_state, state):
        nb_objs = count_objects(state)
        grown_ids = []
        for i_obj in range(nb_objs):
            initial_obj_features = get_obj_features(initial_state, i_obj)
            obj_features = get_obj_features(state, i_obj)
            initial_size = initial_obj_features[size_inds]
            size = obj_features[size_inds]
            if size > initial_size + 0.001:
                grown_ids.append(i_obj)
        return np.array(grown_ids)

    # Whether a supply is in contact with some non living thing object (to track funny behaviors where agents try to grow furtniture etc).
    def get_supply_contact_ids(state):
        nb_objs = count_objects(state)
        all_obj_features = [get_obj_features(state, i_obj) for i_obj in range(nb_objs)]
        supply_ids = []
        non_living_thing_ids = []
        for i_obj in range(nb_objs):
            obj_type = get_obj_type(all_obj_features, i_obj)[0]
            if obj_type in params['categories']['supply']:
                supply_ids.append(i_obj)
            elif obj_type not in params['categories']['living_thing']:
                non_living_thing_ids.append(i_obj)
        if len(supply_ids) > 0:
            supply_contact_ids = []
            for i_supply in supply_ids:
                for i_non_living in non_living_thing_ids:
                    if i_supply != i_non_living:
                        sizes = [all_obj_features[i_supply][size_inds], all_obj_features[i_non_living][size_inds]]
                        positions = [all_obj_features[i_supply][position_inds], all_obj_features[i_non_living][position_inds]]
                        if np.linalg.norm(positions[0] - positions[1]) < (sizes[0] + sizes[1]) / 2:
                            supply_contact_ids.append(i_non_living)
            return np.array(supply_contact_ids)
        else:
            return np.array([])

    get_interactions = dict(get_touched=get_touched_obj_ids,
                            get_grasped=get_grasped_obj_ids,
                            get_grown=get_grown_obj_ids,
                            get_supply_contact=get_supply_contact_ids)


    # extract category of a type attribute
    def find_category_of_attribute(attribute):
        for k in attributes.keys():
            if attribute in attributes[k]:
                return k
        return None

    # check whether an attribute is a relative attribute.
    def check_if_relative(attribute):
        attributes = [a for a in attribute.split(' ') if a != 'and']
        for a in attributes:
            if 'relative' in find_category_of_attribute(a):
                return True
        else:
            return False

    # Compute all combinations of attributes, in case you want to use up to two attributes
    def check_equal_cat(a, b):
        cat_a = find_category_of_attribute(a)
        cat_b = find_category_of_attribute(b)
        if cat_a is None or cat_b is None:
            raise ValueError
        if cat_a == cat_b:
            return True
        elif cat_a in cat_b or cat_b in cat_a:
            return True
        else:
            return False

    # combine two attributes to form a new one. Only works when combining non-relative attributes and adjective attributes.
    def combine_two(attribute_a, attribute_b):
        att_combinations = []
        for a in attribute_a:
            for b in attribute_b:
                # if different and not same category
                if a != b and not check_equal_cat(a, b) and not check_if_relative(a) and not check_if_relative(b):
                    att_combinations.append('{} and {}'.format(a, b))
        return tuple(att_combinations)

    params['extract_functions'] = dict(get_interactions=get_interactions,
                                       get_agent_position_attributes=get_agent_position_attributes,
                                       count_objects=count_objects,
                                       get_obj_features=get_obj_features,
                                       get_attributes_functions=get_attributes_functions,
                                       find_category_of_attribute=find_category_of_attribute,
                                       check_if_relative=check_if_relative,
                                       combine_two=combine_two)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    params['img_path'] = dir_path + '/icons/'
    return params
