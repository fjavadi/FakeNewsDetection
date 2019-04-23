from data import dataset_to_variable
from models import HybridModel
import torch.optim as optim
import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F

def train(use_cuda, sentences, train_data, statement_word2num, metadata_word2num,lr=0.001, epochs=10):
    print("Preparing train data ...")
    dataset_to_variable(train_data, use_cuda)

    print("Creating models...")
    hybrid_model= HybridModel(sentences, statement_word2num, metadata_word2num)
    if use_cuda: hybrid_model.cuda()

    optimizer = optim.Adam(hybrid_model.parameters(), lr=lr)
    display_interval= 100

    for epoch in range(epochs):
        step = 0
        random.shuffle(train_data)
        total_loss = 0
        for x in train_data:
            optimizer.zero_grad()
            prediction= hybrid_model(x)
            target = x.label

            #backward
            loss = F.cross_entropy(prediction, target)
            loss.backward()
            optimizer.step()

            step += 1
            if step % display_interval == 0:
                print('    ==> Iter: ' + str(step) + ' Loss: ' + str(loss))

            total_loss += loss.data.numpy()

        print('  ==> Epoch ' + str(epoch) + ' finished. Avg Loss: ' + str(total_loss / len(train_data)))

    return hybrid_model