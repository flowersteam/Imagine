from src.playground_env.env_params import get_env_params


def generate_all_descriptions(env_params):
    """
    Generates all possible descriptions from a set of environment parameters.

    Parameters
    ----------
    env_params: dict
        Dict of environment parameters from get_env_params function.

    Returns
    -------
    training_descriptions: tuple of str
        Tuple of descriptions that belong to the training set (descriptions that do not contain occurrences reserved to the testing set).
    test_descriptions: tuple of str
        Tuple of descriptions that belong to the testing set (that contain occurrences reserved to the testing set).
    extra_descriptions: tuple of str
        Other descriptions that we might want to track (here when the agent tries to grow furniture for instance).

    """

    p = env_params.copy()

    # Get the list of admissible attributes and split them by name attributes (type and categories) and adjective attributes.
    name_attributes = env_params['name_attributes']
    adjective_attributes = env_params['adjective_attributes']

    # function to check whether an attribute is a relative attribute
    check_if_relative = env_params['extract_functions']['check_if_relative']
    # function to generate attributes as combinations of non-relative attributes.
    combine_two = env_params['extract_functions']['combine_two']


    # Add combined attributes if needed.
    if p['attribute_combinations']:
        adjective_attributes += combine_two(adjective_attributes, adjective_attributes)


    all_descriptions = ()
    
    if 'Move' in p['admissible_actions']:
        move_descriptions = []
        for d in ['left', 'right', 'bottom', 'top']:
            move_descriptions.append('Go {}'.format(d))
        for d1 in ['left', 'right']:
            for d2 in ['top', 'bottom']:
                move_descriptions.append('Go {} {}'.format(d2, d1))
        move_descriptions.append('Go center')
        all_descriptions += tuple(move_descriptions)
    
    if 'Grasp' in p['admissible_actions']:
        grasp_descriptions = []
        for adj in adjective_attributes:
            quantifier = 'any'  # 'the' if check_if_relative(adj) else 'a'
            if not check_if_relative(adj):
                for name in name_attributes:
                    grasp_descriptions.append('Grasp {} {}'.format(adj, name))
                    # grasp_descriptions.append('Grasp {} {} {}'.format(quantifier, adj, name))
            grasp_descriptions.append('Grasp {} {} thing'.format(quantifier, adj))
        for name in name_attributes:
            # grasp_descriptions.append('Grasp a {}'.format(name))
            grasp_descriptions.append('Grasp any {}'.format(name))

        all_descriptions += tuple(grasp_descriptions)
    
    
    if 'Grow' in p['admissible_actions']:
        grow_descriptions = []
        list_exluded = p['categories']['furniture'] + p['categories']['supply'] + ('furniture', 'supply')
        for adj in adjective_attributes:
            if adj not in list_exluded:
                quantifier = 'any' #'the' if check_if_relative(adj) else 'a'
                if not check_if_relative(adj):
                    for name in name_attributes:
                        if name not in list_exluded:
                            grow_descriptions.append('Grow {} {}'.format(adj, name))
                            # grow_descriptions.append('Grow {} {} {}'.format(quantifier, adj, name))
                grow_descriptions.append('Grow {} {} thing'.format(quantifier, adj))
        for name in name_attributes:
            if name not in list_exluded:
                # grow_descriptions.append('Grow a {}'.format(name))
                grow_descriptions.append('Grow any {}'.format(name))

        all_descriptions += tuple(grow_descriptions)

    attempted_grow_descriptions = []
    if 'Grow' in p['admissible_actions']:
        list_exluded = p['categories']['living_thing'] + ('living_thing', 'animal', 'plant')
        for adj in adjective_attributes:
            if adj not in list_exluded:
                quantifier = 'any' #'the' if check_if_relative(adj) else 'a'
                if not check_if_relative(adj):
                    for name in name_attributes:
                        if name not in list_exluded:
                            # attempted_grow_descriptions.append('Attempted grow {} {} {}'.format(quantifier, adj, name))
                            attempted_grow_descriptions.append('Attempted grow {} {}'.format(adj, name))
                attempted_grow_descriptions.append('Attempted grow {} {} thing'.format(quantifier, adj))
        for name in name_attributes:
            if name not in list_exluded:
                # attempted_grow_descriptions.append('Attempted grow a {}'.format(name))
                attempted_grow_descriptions.append('Attempted grow any {}'.format(name))
    attempted_grow_descriptions = tuple(attempted_grow_descriptions)

    train_descriptions = []
    test_descriptions = []
    for descr in all_descriptions:
        to_remove = False
        for w in p['words_test_set_def']: # words_test_set_def is the set of occurrences that is reserved to the testing set.
            if w in descr:
                to_remove = True
                break
        if not to_remove:
            train_descriptions.append(descr)
        else:
            test_descriptions.append(descr)
    
    train_descriptions = tuple(sorted(train_descriptions))
    test_descriptions = tuple(sorted(test_descriptions))
    extra_descriptions = tuple(sorted(attempted_grow_descriptions))

    return train_descriptions, test_descriptions, extra_descriptions

if __name__ == '__main__':
    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
    