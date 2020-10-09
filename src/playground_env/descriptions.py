from src.playground_env.env_params import get_env_params



def generate_all_descriptions(env_params):

    p = env_params.copy()

    # Get the list of admissible attributes
    name_attributes = ()
    adjective_attributes = ()
    for att_type in p['attributes'].keys():
        if att_type in p['admissible_attributes']:
            if att_type in ('types', 'categories'):
                name_attributes += p['attributes'][att_type]
            else:
                adjective_attributes += p['attributes'][att_type]

    find_category_of_attribute = env_params['extract_functions']['find_category_of_attribute']
    check_if_relative = env_params['extract_functions']['check_if_relative']
    combine_two = env_params['extract_functions']['combine_two']


    # combine two
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

    if 'Grow' in p['admissible_actions']:
        attempted_grow_descriptions = []
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



    train_descriptions = []
    test_descriptions = []
    for descr in all_descriptions:
        to_remove = False
        for w in p['words_test_set_def']:
            if w in descr:
                to_remove = True
                break
        if not to_remove:
            train_descriptions.append(descr)
        else:
            test_descriptions.append(descr)
    
    train_descriptions = sorted(train_descriptions)
    test_descriptions = sorted(test_descriptions)
    extra_descriptions = sorted(attempted_grow_descriptions)

    return train_descriptions, test_descriptions, extra_descriptions

if __name__ == '__main__':
    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
    