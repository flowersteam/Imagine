import numpy as np

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
            instructions_one_hot = self.classifier.goal_sampler.feedback2one_hot[input_str]
        else:
            instructions_one_hot = self.classifier.goal_sampler.one_hot_encoder.encode(input_str.lower().split(" "))
        embeddings = self.classifier.sess.run(self.reward_function.get_instruction_embedding(),
                      feed_dict={self.reward_function.I: np.expand_dims(np.asarray(instructions_one_hot), 0)})
        return embeddings.squeeze()
