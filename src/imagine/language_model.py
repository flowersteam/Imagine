import numpy as np
import torch

class LanguageModelLSTM:
    def __init__(self, params):
        self.goal_dim = params['dims']['g_encoding']
        self.classifier = None
        self.reward_function = None
        self.size_encoding = None
        self.description2onehot = None

    def set_reward_function(self, classifier):
        self.classifier = classifier
        self.reward_function = classifier.reward_function
        self.size_encoding = self.reward_function.num_hidden_lstm
        self.description2onehot = classifier.goal_sampler.feedback2one_hot

    def encode(self, input_str):
        # input_str can be either:
        # - a string,
        # - a list of length rollout_batch_size, each element being a list of strings

        if input_str in self.classifier.goal_sampler.feedback2one_hot.keys():
            descriptions_one_hot = self.classifier.goal_sampler.feedback2one_hot[input_str]
        else:
            descriptions_one_hot = self.classifier.goal_sampler.one_hot_encoder.encode(input_str.lower().split(" "))

        descriptions_one_hot=np.asarray(descriptions_one_hot).reshape([1]+list(np.shape(descriptions_one_hot)))
        embeddings = self.classifier.reward_function.get_description_embedding(torch.tensor(descriptions_one_hot, dtype=torch.float32))

        return embeddings.squeeze().detach().numpy()
