import torch
import torch.nn as nn
from src.or_module.train_or_module import OrNet
from torch.autograd import Variable


class Feedforward(torch.nn.Module):
    " A simple multilayer fully connected network"

    def __init__(self, input_size, layers_sizes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.layers_sizes = layers_sizes
        self.fc_layers = nn.ModuleList()
        self.activations = []

        size_tmp = self.input_size
        for i, size in enumerate(self.layers_sizes):
            self.activations.append(nn.ReLU() if i < len(self.layers_sizes) - 1 else None)
            fc = torch.nn.Linear(size_tmp, size)
            nn.init.kaiming_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fc_layers.append(fc)
            size_tmp = size

    def forward(self, input):
        for fc, activation in zip(self.fc_layers, self.activations):
            input = fc(input)
            if activation:
                input = activation(input)
        return input


class RewardFunctionModel(torch.nn.Module):
    '''
    The Reward Function DEEPSET model.

    Parameters
    ----------
    or_params_path : path to OR network pretrained weights
    body_size : Size of body (3 in Playground)
    obj_size : Size of object features vector (
    n_obj : Number of objects in the scene (defaut is 3 in Playground)
    state_size : State size (n_obj * obj_size + body_size) * 2
    voc_size : Size of the vocabulary used to construct language descriptions provided by the social Partner in the Environment
    sequence_length : Maximum length of sequences (number of tokens in descriptions provided by the social Partner)
    batch_size : Number of examples used to compute gradient for training.
    num_hidden_lstm : Size of the latent space of the LSTM.
    '''

    def __init__(self, or_params_path, body_size, obj_size, n_obj, state_size, voc_size, sequence_length, batch_size,
                 num_hidden_lstm):

        super().__init__()
        self.body_size = body_size
        self.obj_size = obj_size
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.state_size = state_size
        self.voc_size = voc_size
        self.sequence_length = sequence_length

        self.ff_size = 100
        self.num_hidden_lstm = num_hidden_lstm

        self.body_indices = [i for i in range(body_size)]
        self.delta_body_indices = [i + self.state_size // 2 for i in range(body_size)]
        self.objs_indices = []
        self.delta_objs_indices = []

        for i in range(n_obj):
            self.objs_indices.append([j for j in range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.delta_objs_indices.append([j + self.state_size // 2 for j in
                                            range(body_size + i * obj_size, body_size + i * obj_size + obj_size)])
            self.half_o = self.state_size // 2

        self.sub_state_size = 2 * (body_size + obj_size)

        self.lstm = nn.LSTM(self.voc_size, self.num_hidden_lstm, batch_first=True)
        self.fc_cast = Feedforward(self.num_hidden_lstm, [self.sub_state_size])
        self.fc_reward = Feedforward(self.sub_state_size, [self.ff_size, 1])
        self.or_model = OrNet(n_obj=n_obj)  # modify ornet with deep set
        self.or_model.load_state_dict(torch.load(or_params_path[self.n_obj]))
        self.sigmoid = torch.sigmoid

    def pred_from_h_lstm(self, s, h_lstm):
        '''
        Computes the reward for a given state s and a given language code h_lstm
        Parameters
        ----------
        s : Input state
        h_lstm : Input language encoding (Output from LSTM)

        Returns
        -------
        logits: the reward logit (the answer to the question: Does my state s corresponds to a configuration that
        fulfills the descriptions encoded in h_lstm)
        '''
        h_cast = self.sigmoid(self.fc_cast(h_lstm))
        # extract inputs for each object (and their delta)
        object_features = []
        for i in range(self.n_obj):
            obj_features = torch.cat(tensors=(s[:, self.objs_indices[i][0]: self.objs_indices[i][-1] + 1],
                                              s[:, self.objs_indices[i][0] + self.half_o: self.objs_indices[i][
                                                                                              -1] + 1 + self.half_o]),
                                     dim=1)
            object_features.append(obj_features)
        # extract body inputs and its delta
        state_body = torch.cat(tensors=(s[:, :self.objs_indices[0][0]],
                                        s[:, self.half_o: self.half_o + self.objs_indices[0][0]]), dim=1)

        object_features_attention = []
        for i in range(self.n_obj):
            obj_features_att = torch.mul(object_features[i], h_cast[:, int(state_body.shape[1]):])
            object_features_attention.append(obj_features_att)
        state_body_attention = torch.mul(state_body, h_cast[:, :int(state_body.shape[1])])

        # compute the reward output for each object

        logits_obj = []
        for i in range(self.n_obj):
            input_reward_func = torch.cat(tensors=(state_body_attention, object_features_attention[i]), dim=1)
            reward_current = self.sigmoid(self.fc_reward(input_reward_func))
            logits_obj.append(reward_current)
        logits_concat = torch.cat(tensors=logits_obj, dim=1)

        logits = self.or_model(logits_concat)

        return logits

    def forward(self, s, description):
        '''
        Forward pass of the model

        Parameters
        ----------
        s : Input state
        description : Input Description (one hot encoded)

        Returns
        -------
        The reward logit.
        '''
        h_lstm_seq, _ = self.lstm(description)
        h_lstm = h_lstm_seq[:, -1, :]

        return self.pred_from_h_lstm(s, h_lstm)

    def get_description_embedding(self, descr):
        '''

        Parameters
        ----------
        descr : a description (one hot encoded)

        Returns
        -------
        The description's code.
        '''
        self.eval()
        with torch.no_grad():
            h_lstm_seq, _ = self.lstm(descr)
        return h_lstm_seq[:, -1, :]

    def get_params(self):
        filtered_params = []
        for param, named_param in zip(self.parameters(), self.named_parameters()):
            if 'or_model' not in named_param[0]:
                print(named_param[0])
                filtered_params.append(param)
        return filtered_params

    def get_attention_vector(self,h_lstm):
        return self.sigmoid(self.fc_cast(h_lstm))

    def compute_logits_before_or(self, s, h_lstm):
        h_cast = self.sigmoid(self.fc_cast(h_lstm))
        # extract inputs for each object (and their delta)
        object_features = []
        for i in range(self.n_obj):
            obj_features = torch.cat(tensors=(s[:, self.objs_indices[i][0]: self.objs_indices[i][-1] + 1],
                                              s[:, self.objs_indices[i][0] + self.half_o: self.objs_indices[i][
                                                                                              -1] + 1 + self.half_o]),
                                     dim=1)
            object_features.append(obj_features)
        # extract body inputs and its delta
        state_body = torch.cat(tensors=(s[:, :self.objs_indices[0][0]],
                                        s[:, self.half_o: self.half_o + self.objs_indices[0][0]]), dim=1)

        object_features_attention = []
        for i in range(self.n_obj):
            obj_features_att = torch.mul(object_features[i], h_cast[:, int(state_body.shape[1]):])
            object_features_attention.append(obj_features_att)
        state_body_attention = torch.mul(state_body, h_cast[:, :int(state_body.shape[1])])

        # compute the reward output for each object

        logits_obj = []
        for i in range(self.n_obj):
            input_reward_func = torch.cat(tensors=(state_body_attention, object_features_attention[i]), dim=1)
            reward_current = self.sigmoid(self.fc_reward(input_reward_func))
            logits_obj.append(reward_current)

        return logits_obj