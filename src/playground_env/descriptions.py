from src.playground_env.env_params import *


attributes = []
if 'category' in ATTRIBUTE_LIST: attributes += group_names
if 'type' in ATTRIBUTE_LIST: attributes += things
if 'size' in ATTRIBUTE_LIST: attributes += thing_sizes
if 'shade' in ATTRIBUTE_LIST: attributes += thing_shades
if 'color' in ATTRIBUTE_LIST: attributes += thing_colors
if 'relative_shade' in ATTRIBUTE_LIST: attributes += ['lighest', 'darkest']
if 'relative_size' in ATTRIBUTE_LIST: attributes += ['biggest', 'smallest']
if 'absolute_size' in ATTRIBUTE_LIST: attributes += ['left', 'right', 'top', 'bottom']
attributes_restricted = attributes.copy()
for a in attributes:
    if a in things or a in group_names:
        attributes_restricted.remove(a)


attributes_combinations = []

def combine_two(attribute_a, attribute_b):
    for a in attribute_a:
        for b in attribute_b:
            attributes_combinations.append('{} and {}'.format(a, b))
            attributes_combinations.append('{} and {}'.format(b, a))

def combine_three(attribute_a, attribute_b, attribute_c):
    for a in attribute_a:
        for b in attribute_b:
            for c in attribute_c:
                attributes_combinations.append('{}, {} and {}'.format(a, b, c))
                attributes_combinations.append('{}, {} and {}'.format(a, c, b))
                attributes_combinations.append('{}, {} and {}'.format(b, a, c))
                attributes_combinations.append('{}, {} and {}'.format(b, c, a))
                attributes_combinations.append('{}, {} and {}'.format(c, a, b))
                attributes_combinations.append('{}, {} and {}'.format(c, b, a))


# combine two
combine_two(thing_colors, thing_shades)
combine_two(thing_colors, thing_sizes)
combine_two(thing_colors, things)
combine_two(thing_colors, group_names)

combine_two(thing_shades, thing_sizes)
combine_two(thing_shades, things)
combine_two(thing_shades, group_names)

combine_two(thing_sizes, things)
combine_two(thing_sizes, group_names)


all_descriptions = []

if 'Gripper' in admissible_actions:
    gripper_descriptions = []
    for d in ['left', 'right', 'bottom', 'top']:
        gripper_descriptions.append('Go {}'.format(d))
    for d1 in ['left', 'right']:
        for d2 in ['top', 'bottom']:
            gripper_descriptions.append('Go {} {}'.format(d2, d1))
    gripper_descriptions.append('Go center')
    all_descriptions += gripper_descriptions

if 'Grasp' in admissible_actions:
    grasp_descriptions = []
    if 1 in MODES:
        for attribute in attributes:
            if attribute in thing_sizes + thing_shades + thing_colors :
                grasp_descriptions.append('Grasp any {} thing'.format(attribute))
            else:
                grasp_descriptions.append('Grasp any {}'.format(attribute))
    if 2 in MODES:
        for attribute in attributes_restricted:
            for type in things + group_names:
                grasp_descriptions.append('Grasp {} {}'.format(attribute, type))

    all_descriptions += grasp_descriptions


if 'Grow' in admissible_actions:
    grow_descriptions = []
    if 1 in MODES:
        for attribute in attributes:
            list_exluded = furnitures + supply + ['furniture', 'supply']
            if attribute not in list_exluded:
                if attribute in thing_sizes + thing_shades + thing_colors:
                    grow_descriptions.append('Grow any {} thing'.format(attribute))
                else:
                    grow_descriptions.append('Grow any {}'.format(attribute))
    if 2 in MODES:
        for attribute in attributes_restricted:
            for type in things + group_names:
                list_exluded = furnitures + supply + ['furniture', 'supply']
                if type not in list_exluded:
                    grow_descriptions.append('Grow {} {}'.format(attribute, type))

    all_descriptions += grow_descriptions

# add extra descriptions
extra_descriptions = []
if 'Grow' in admissible_actions:
    if 1 in MODES:
        for attribute in attributes:
            if attribute in plants + furnitures + ['plant', 'furniture']:
                extra_descriptions.append('Attempt grow any {}'.format(attribute))
    if 2 in MODES:
        for attribute in attributes_restricted:
            for type in plants + furnitures + ['plant', 'furniture']:
                extra_descriptions.append('Attempt grow {} {}'.format(attribute, type))



train_descriptions = []
test_descriptions = []
for descr in all_descriptions:
    to_remove = False
    for w in words_to_remove_from_train:
        if w in descr:
            to_remove = True
            break
    if not to_remove:
        train_descriptions.append(descr)
    else:
        test_descriptions.append(descr)

train_descriptions = sorted(train_descriptions)
test_descriptions = sorted(test_descriptions)
extra_descriptions = sorted(extra_descriptions)

stop = 1
