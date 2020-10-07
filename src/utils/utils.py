import random
from collections import Counter

import numpy as np
from tqdm import tqdm


def add_new_data(dataset_to_update, state, embedded_instructions, instruction_index, lab, true_lab=None):
    """
    Update the dataset as a list with a new list of length (state_dim + embedding_dim + 3)
    The last three dimensions are :
    - the label infered by the system according to the heuristic used,
    - the instruction index,
    - the true label given by the oracle but potentially unknown by the system,
    :param dataset_to_update: list
    :param state: np.array of dimension (1,state_dim)
    :param embedded_instructions: np.array of dimension (n_instruction, embedding_dim)
    :param instruction_index: int between 0 and n_instruction - 1
    :param lab: guessed label by the system according to the heuristic used 0 or 1
    :param true_lab: oracle label 0 or 1
    :return: void
    """
    new_data = state + embedded_instructions[instruction_index].tolist() + [lab, instruction_index, true_lab]
    dataset_to_update.append(new_data)


def get_discovered_instructions(current_discovered_instructions, labels):
    _, encountered_instruction = labels.nonzero()
    result = set(encountered_instruction).union(current_discovered_instructions)
    return sorted(result)


def update_dataset_from_exhaustive_feedback(dataset_to_update, states, labels, embedded_instructions,
                                            discovered_instructions):
    """

    :param dataset_to_update: list of list containing [state_feature, embedding_feature, label, instruciton_idx]
    :param states: np array of shape [n_state, state_feature]
    :param labels: np.array of shape [n_state, discovered_instruction]
    :param embedded_instructions: np.array of shape [discovered_instruction, embedding_dimension]
    """
    assert (len(states) == len(labels))
    for s, labs in zip(states, labels):
        s = s.tolist()
        for lab, idx in zip(labs, discovered_instructions):
            add_new_data(dataset_to_update, s, embedded_instructions, idx, lab, true_lab=lab)


def update_train_from_exhaustive_feedback(train_set, states, labels, embedded_instructions,
                                          discovered_instruction_record):
    current_discovered_instruction = [] if len(discovered_instruction_record) == 0 else discovered_instruction_record[
        -1]
    discovered_instruction = get_discovered_instructions(current_discovered_instruction, labels[:, -1])
    update_dataset_from_exhaustive_feedback(train_set,
                                            states[:, -1],
                                            labels[:, -1, discovered_instruction],
                                            embedded_instructions,
                                            discovered_instruction)
    return discovered_instruction


def get_most_complex_feedback(labels, oracle_complexity, instruction_complexity):
    """
    Get most complex positive feedback if it exists. Complexity is measured by the oracle according by grouping
    instructions by level of difficulty. If positive labels for same complexity task are available, the oracle chooses
    one randomly.
    Note : if an instruction has a positive label, it necessarily has been discovered ???
    :param labels:
    :param oracle_complexity:
    :param instruction_complexity:
    :return:
    """
    feedbacks_instruction = []
    for labs in labels:
        feedback = -1
        candidate_instruction = set(np.argwhere(labs == 1).flatten())
        for _, ins_list in reversed(oracle_complexity.items()):
            c = candidate_instruction.intersection(ins_list)
            if c:
                feedback = random.choice(tuple(c))
                instruction_complexity.update([feedback])
                break
        feedbacks_instruction.append(feedback)
    return feedbacks_instruction


def update_dataset_from_most_complex_feedback(dataset_to_update, states, labels, embedded_instructions,
                                              instruction_complexity, oracle_complexity):
    assert (len(states) == len(labels))
    feedbacks_instruction = get_most_complex_feedback(labels, oracle_complexity, instruction_complexity)
    for s, labs, feedback in zip(states, labels, feedbacks_instruction):
        s = s.tolist()
        for ins, freq in reversed(instruction_complexity.most_common()):
            if ins == feedback:
                add_new_data(dataset_to_update, s, embedded_instructions, ins, 1, true_lab=labs[ins])
                break
            else:
                add_new_data(dataset_to_update, s, embedded_instructions, ins, 0, true_lab=labs[ins])


def update_train_from_complex_feedback(train_set, states, labels, embedded_instructions, instruction_complexity_record,
                                       oracle_complexity, **kwargs):
    instruction_complexity = Counter() if len(instruction_complexity_record) == 0 else instruction_complexity_record[
        -1].copy()
    update_dataset_from_most_complex_feedback(train_set, states[:, -1],
                                              labels[:, -1],
                                              embedded_instructions,
                                              instruction_complexity,
                                              oracle_complexity)
    instruction_complexity_record.append(instruction_complexity)
    discovered_instruction = sorted(map(int, instruction_complexity.keys()))
    return discovered_instruction


def compute_training_set(train_update_func, states, labels, embedded_instructions, batch, max_episode, **kwargs):
    """

    :param train_update_func:
    :param states:
    :param labels:
    :param embedded_instructions:
    :param batch:
    :param max_episode:
    :param train_set_length_record:
    :param discovered_instruction_record:
    :param kwargs:
    :return:
    """
    train_set = []
    train_set_length = []
    discovered_instruction_record = []
    for episode in tqdm(range(max_episode)):
        # for episode in tqdm(range(len(states) // (num_workers * num_rollout))):
        episode_state = states[batch * episode: batch * (episode + 1)]
        episode_label = labels[batch * episode: batch * (episode + 1)]
        discovered_instruction = train_update_func(train_set=train_set, states=episode_state, labels=episode_label,
                                                   embedded_instructions=embedded_instructions,
                                                   discovered_instruction_record=discovered_instruction_record,
                                                   **kwargs)
        train_set_length.append(len(train_set))
        discovered_instruction_record.append(discovered_instruction)
    return np.array(train_set), train_set_length, discovered_instruction_record

