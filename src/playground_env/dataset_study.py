import sys
sys.path.append('../../')
import pickle
from src.playground_env.env_params import *


dataset_name = 'dataset_playground'
path = './'


with open(path + dataset_name + '.pk', 'rb') as f:
    dataset = pickle.load(f)

# obs_size = dataset[0]['obs'].shape[1]
# inds_without_grasped = np.concatenate([np.arange(0, obs_size // 2 - DIM_OBJ) , np.arange(obs_size // 2, obs_size - DIM_OBJ)])

all_obs = []
all_descriptions = []
count_descriptions = dict()
for ep in dataset:
    all_obs.append(ep['obs'])
    all_descriptions.append(ep['descriptions'])
    descr_ep = [d for desc in ep['descriptions'] for d in desc]
    for d in descr_ep:
        if d in count_descriptions.keys():
            count_descriptions[d] += 1
        else:
            count_descriptions[d] = 1
all_obs = np.array(all_obs)


descriptions = count_descriptions.keys()
description_id = dict()
id_description = dict()
for i, d in enumerate(descriptions):
    description_id[d] = i
    id_description[i] = d

all_descriptions_ids = []
for ep in dataset:
    ids = []
    for list_descr in ep['descriptions']:
        step_ids = []
        for d in list_descr:
            step_ids.append(description_id[d])
        ids.append(step_ids)
    all_descriptions_ids.append(ids)

print(count_descriptions)
dataset = dict(obs=all_obs,
               descriptions_ids=all_descriptions_ids,
               count_descriptions=count_descriptions,
               description2id=description_id,
               id2description=id_description)

a = []
for i in count_descriptions.values():
    a.append(i)
a = np.array(a)
print(a.min())
print(len(a))
with open(path + dataset_name + '_extracted.pk', 'wb') as f:
    pickle.dump(dataset, f)
obs = dataset['obs']  # shape (# eps, # steps, # features)
descriptions_ids = dataset['descriptions_ids']  # three levels of lists (ep, steps, description for step)
count_descriptions = dataset['count_descriptions']  # dictionary with number of occurences for each description
description2id = dataset['description2id']  # dictionary to convert descriptions to ids
id2description = dataset['id2description']  # dictionary to convert ids to descriptions
print(count_descriptions)

stop = 1