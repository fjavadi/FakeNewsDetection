import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gensim.models import Word2Vec
import numpy as np


class StatementModel(nn.Module):

    def __init__(self,sentences, vocab_size, hidden_size=300, number_filters=128, filter_sizes= [2,3,4]):
        super(StatementModel, self).__init__()

        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.number_filters= number_filters

        sents= [[word for word in sentence] for sentence in sentences]

        w2v = Word2Vec(sents, min_count=1, size=self.hidden_size)
        self.embeddings= nn.Embedding(self.vocab_size, self.hidden_size)
        self.filter_sizes=filter_sizes
        self.convs= [nn.Conv2d(1, self.number_filters, (filter_size, self.hidden_size)) for filter_size in filter_sizes]

    def forward(self, x):
        output= self.embeddings(x.statement.unsqueeze(0))
        output = [F.relu(conv(output.unsqueeze(0))).squeeze(3) for conv in self.convs]  # 1*1*W*1 -> 1*Co*W x [len(convs)]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]  # 1*Co*1 -> 1*Co x len(convs)
        output = torch.cat(output, 1)  # 1*len(convs)
        # print("siiize stament",output.size()) 1*384
        return output

class MetadataModel (nn.Module):
    def __init__(self,vocab_size,hidden_size=50, number_filters=10,filter_sizes=[3,5]):
        super(MetadataModel, self).__init__()

        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.number_filters= number_filters
        self.embeddings=nn.Embedding(self.vocab_size, self.hidden_size)
        self.filter_sizes=filter_sizes
        self.convs= [nn.Conv2d(1, self.number_filters, (filter_size, self.hidden_size)) for filter_size in filter_sizes]

    def forward(self, x):
        output= self.embeddings(x.metadata.unsqueeze(0))
        output = [F.relu(conv(output.unsqueeze(0))).squeeze(3) for conv in self.convs]  # 1*1*W*1 -> 1*Co*W x [len(convs)]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]  # 1*Co*1 -> 1*Co x len(convs)
        output = torch.cat(output, 1)  # 1*len(convs)
        return output

class HybridModel (nn.Module):
    def __init__(self,sentences, statement_word2num,metadata_word2num, dropout=0.8):
        super(HybridModel, self).__init__()

        self.statement_model= StatementModel(sentences, len(statement_word2num))
        self.metadata_model=MetadataModel(len(metadata_word2num))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(404, 6)

    def forward(self,x ):
        #statement
        statement_output = self.statement_model(x)


        #metadata
        metadata_output= self.metadata_model(x)

        #all_features
        # print("statement output , metadata output",statement_output.shape, metadata_output.shape)
        features = torch.cat((statement_output.squeeze(0).squeeze(0), metadata_output.squeeze(0).squeeze(0)))
        output=self.fc(self.dropout(features))
        return output.unsqueeze(0)

