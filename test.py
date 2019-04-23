from data import dataset_to_variable
from sklearn.metrics import confusion_matrix , accuracy_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def test(test_data, model,use_cuda):
    print("Preparing test data ...")
    dataset_to_variable(test_data, use_cuda)

    predictions= []
    for x in test_data:
        predict= model(x)
        predict= int(np.argmax(predict.data.numpy()))
        predictions.append(predict)

    return predictions

def evaluate(predictions, labels):

    cm = confusion_matrix(labels, predictions)
    cm= cm/cm.astype(np.float).sum(axis=0)
    acc=accuracy_score(labels, predictions)
    pr, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")

    return acc, cm, pr, recall, f1