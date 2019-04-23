from data import data_load, get_labels
from train import train
from test import test , evaluate


train_data, statement_word2num, metadata_word2num, sentences= data_load("liar_dataset/train.tsv", True)
print("sentences before train", len(sentences), sentences[0])
print("Train data loaded!")
test_data,_,_,_= data_load("liar_dataset/test.tsv", False, statement_word2num, metadata_word2num)
print("Test data loaded!")
model = train(False, sentences, train_data[:100], statement_word2num, metadata_word2num)
predictions= test(test_data, model,False )
acc, cm, precision, recall , f1 = evaluate(get_labels(test_data), predictions)
print("acc", acc)
print("cm", cm)
print("precision", precision)
print("recall", recall)
print("f1", f1)
file = open("result.txt", "w")
file.write(str(acc))
file.write(str(cm))
file.write(str(precision))
file.write(str(recall))
file.write(str(f1))
file.close()