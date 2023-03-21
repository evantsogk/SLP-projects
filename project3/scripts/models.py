import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import init
from torch.autograd import Variable


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        # EX4
        num_emb, emb_dim = embeddings.shape
        self.embedding = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        # EX4
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=trainable_emb)

        # 4 - define a non-linear transformation of the representations
        # EX5
        self.fc = nn.Linear(emb_dim, 16)
        self.relu = nn.ReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.clf = nn.Linear(16, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.sum(embeddings, 1)  # EX6
        for i in range(lengths.shape[0]):
            representations[i] = representations[i] / lengths[i]

        # 3 - transform the representations to new ones.
        representations = self.relu(self.fc(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.clf(representations)  # EX6

        return logits


class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


class BasicLSTM(nn.Module):
    def __init__(self, embeddings, rnn_size, output_dim, num_layers, trainable_emb=False, bidirectional=False,
                 pooling=False, emb_attention=False, lstm_attention=False):
        super(BasicLSTM, self).__init__()
        self.scores = None  # attention scores
        self.bidirectional = bidirectional  # if lstm is bidirectional
        self.pooling = pooling  # add mean and max pooling to lstm outputs
        self.emb_attention = emb_attention  # apply attentions to embeddings
        self.lstm_attention = lstm_attention  # apply attention to lstm outputs

        # find feature size
        num_emb, emb_dim = embeddings.shape
        self.feature_size = rnn_size
        if pooling: self.feature_size = self.feature_size + 2 * emb_dim
        if emb_attention: self.feature_size = emb_dim

        # define the embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=trainable_emb)

        # apply attention to embeddings
        if emb_attention:
            self.emb_attention = SelfAttention(emb_dim, batch_first=True)

        # initialize the LSTM
        self.lstm = nn.LSTM(emb_dim, rnn_size, num_layers, batch_first=True)

        # apply lstm attention
        if lstm_attention:
            self.lstm_attention = SelfAttention(rnn_size, batch_first=True)

        # output layer
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        # embeddings
        embeddings = self.embedding(x)

        if not self.emb_attention:
            # lstm output
            x, _ = self.lstm(embeddings)

            if not self.lstm_attention:
                # lstm last output including the bidirectional case
                x = self.last_timestep(x, lengths, self.bidirectional)

                # mean and max pooling
                if self.pooling:
                    x = self.add_pooling(x, embeddings, lengths)

            else:  # lstm attention
                x, scores = self.lstm_attention(x, lengths)
                self.scores = scores

        else:  # embeddings attention
            x, scores = self.emb_attention(embeddings, lengths)
            self.scores = scores

        # output
        x = self.fc(x)

        return x

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx.long()).squeeze()

    @staticmethod
    def add_pooling(output, embeddings, lengths):
        """
        Concatenate LSTM output with mean and max pooling
        """
        mean_pooling = torch.sum(embeddings, 1)  # EX6
        for i in range(lengths.shape[0]):
            mean_pooling[i] = mean_pooling[i] / lengths[i]
        max_pooling, _ = torch.max(embeddings, 1)

        return torch.cat((output, mean_pooling, max_pooling), dim=-1)
