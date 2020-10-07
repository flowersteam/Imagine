import pickle
import sys
import json
import os
import numpy as np

sys.path.append('../..')
sys.path.append('../../..')

from src.utils.nlp_tools import Vocab, OneHotEncoder, analyze_descr
from src.utils.data_utils import create_test_set, create_train_set, \
    create_test_set_only_last_transition, pickle_dump
from src.imagine.goal_generator.descriptions import get_descriptions

ENV_NAME = 'big'

DATA_DIR = '../../data'

RAW_DATA_FILE = DATA_DIR + '/raw/dataset_playground.pk'

folder_out = DATA_DIR + '/processed/'

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

with open(RAW_DATA_FILE, 'rb') as fp:
    raw_data = pickle.load(fp)

train_descriptions, test_descriptions, all_descriptions = get_descriptions(ENV_NAME)
id2description_raw = raw_data['id2description']
id2description = {id: descr for id, descr in id2description_raw.items() if descr in all_descriptions}
ids_to_remove = [id for id in id2description_raw.keys() if id not in id2description.keys()]

id_train_descriptions = [id for id in id2description if id2description[id] in train_descriptions]
id_test_descriptions = [id for id in id2description if id2description[id] in test_descriptions]
split_descriptions, max_seq_length, word_set = analyze_descr(all_descriptions)
vocab = Vocab(word_set)

obs_raw = raw_data['obs']
descriptions_ids_raw = raw_data['descriptions_ids']
descriptions_ids = []
for d_ids in descriptions_ids_raw:
    new_d_ids = []
    for d in d_ids:
        new_d_ids.append([id for id in d if id not in ids_to_remove])
    descriptions_ids.append(new_d_ids)

obs = obs_raw
print(len(obs_raw[0][0]))

one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
id2one_hot = dict()
for id, sent in id2description.items():
    id2one_hot[id] = one_hot_encoder.encode(sent.lower().split(' '))

# Split Train Test
indices = set([i for i in range(len(obs))])
indices_test = set([10 * i + 9 for i in range(len(obs) // 10)])
indices_train = list(indices - indices_test)

size_train = len(indices_train) * 5 // 5
indices_train = indices_train[:size_train]

obs_train = [obs[i] for i in indices_train]
descriptions_ids_train = [descriptions_ids[i] for i in indices_train]

obs_test = [obs[i] for i in list(indices_test)]
descriptions_ids_test = [descriptions_ids[i] for i in list(indices_test)]
size_state = len(obs_test[0][0])

# Create Sets
print("Creating sets")
train_set = create_train_set(obs_train, descriptions_ids_train, id2description, size_state)
state_idx_buffer = train_set['state_idx_buffer']
state_train_list = train_set['states']

test_set = create_test_set(obs_test, descriptions_ids_test, id2description, size_state)
state_idx_reward_buffer = test_set['state_idx_reward_buffer']
state_test_list = test_set['states']

test_set_last_transition = create_test_set_only_last_transition(obs_test, descriptions_ids_test, id2description,
                                                                size_state)
state_idx_reward_buffer_last_transition = test_set_last_transition['state_idx_reward_buffer']
state_test_list_last_transition = test_set_last_transition['states']

print("Sampling test sets")
state_idx_buffer_sampled = dict()
for i in id_train_descriptions:
    state_idx_buffer_sampled[i] = state_idx_buffer[i]
train_set_sampled = dict(states=state_train_list, state_idx_buffer=state_idx_buffer_sampled)

state_idx_reward_buffer_sampled = {}
state_idx_reward_buffer_sampled_last_transition = {}
for id in id_train_descriptions:
    state_idx_reward_buffer_sampled[id] = state_idx_reward_buffer[id]
    state_idx_reward_buffer_sampled_last_transition[id] = state_idx_reward_buffer_last_transition[id]
test_set_state_generalization = dict(states=state_test_list, state_idx_reward_buffer=state_idx_reward_buffer_sampled)
test_set_state_generalization_last_transition = dict(states=state_test_list_last_transition,
                                                     state_idx_reward_buffer=state_idx_reward_buffer_sampled_last_transition)

count_pos_reward = {}
state_idx_reward_buffer_sampled_language = {}
state_idx_reward_buffer_sampled_language_last_transition = {}
for id in id_test_descriptions:
    state_idx_reward_buffer_sampled_language[id] = state_idx_reward_buffer[id]
    state_idx_reward_buffer_sampled_language_last_transition[id] = state_idx_reward_buffer_last_transition[id]
    count_pos_reward[id] = np.sum([state_idx_reward_buffer_sampled_language[id][i][1] for i in
                                   range(len(state_idx_reward_buffer_sampled_language[id]))])
test_set_language_generalization = dict(states=state_test_list,
                                        state_idx_reward_buffer=state_idx_reward_buffer_sampled_language)
test_set_language_generalization_last_transition = dict(states=state_test_list_last_transition,
                                                        state_idx_reward_buffer=state_idx_reward_buffer_sampled_language_last_transition)

description_data = dict(id2description=id2description, id2one_hot=id2one_hot, vocab=vocab,
                        max_seq_length=max_seq_length, encoder=one_hot_encoder)

# pickling everything
pickle_dump(description_data, folder_out + '/descriptions_data.pk')
pickle_dump(train_set_sampled, folder_out + '/train_set.pk')
pickle_dump(test_set_state_generalization, folder_out + '/test_set.pk')
pickle_dump(test_set_language_generalization, folder_out + '/test_set_language_generalization.pk')
pickle_dump(count_pos_reward, folder_out + '/count_pos_reward_test_set_language_generalization.pk')
pickle_dump(test_set_state_generalization_last_transition, folder_out + '/test_set_last_transition.pk')
pickle_dump(test_set_language_generalization_last_transition,
            folder_out + '/test_set_language_generalization_last_transition.pk')
