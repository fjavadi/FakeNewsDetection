import torch

label_to_number = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

def preprocess(string):
	string= string.lower().strip()
	punctuations= ['(',')',':','%','$','\"','\'', '.',';',',']
	for punct in punctuations:
		string= string.replace(punct, " ")
	return string

def count_in_vocab(dict,  word, is_train):
    if is_train:
        if word not in dict:
            dict[word] = len(dict)
            return dict[word]
        else:
            return dict[word]
    else :
        if word not in dict:
            return dict["unk"]
        else:
            return dict[word]

def get_labels(dataset):
    labels=[]
    for x in dataset:
        labels.append(int(x.label))
    return labels

def dataset_to_variable(dataset, use_cuda):
    for i in range(len(dataset)):
        dataset[i].label= torch.LongTensor([dataset[i].label])
        dataset[i].statement = torch.LongTensor(dataset[i].statement)
        dataset[i].subject = torch.LongTensor(dataset[i].subject)
        dataset[i].speaker = torch.LongTensor([dataset[i].speaker])
        dataset[i].speaker_job = torch.LongTensor(dataset[i].speaker_job)
        dataset[i].state = torch.LongTensor([dataset[i].state])
        dataset[i].party = torch.LongTensor([dataset[i].party])
        dataset[i].context = torch.LongTensor(dataset[i].context)
        dataset[i].metadata = torch.LongTensor(dataset[i].metadata)

        if use_cuda:
            dataset[i].label.cuda()
            dataset[i].statement.cuda()
            dataset[i].subject.cuda()
            dataset[i].speaker.cuda()
            dataset[i].speaker_job.cuda()
            dataset[i].state.cuda()
            dataset[i].party.cuda()
            dataset[i].context.cuda()
            dataset[i].metadata.cuda()

class DataSample:
    def __init__(self, label, statement, subject, speaker, speaker_job, state, party, context):
        self.label = label_to_number[label]
        self.statement = preprocess(statement).split()
        self.sentence= preprocess(statement)
        self.subject = preprocess(subject).split()
        self.speaker = speaker.lower()
        self.speaker_job = speaker_job.lower().strip().split()
        self.state = state.lower()
        self.party = party.lower()
        self.context = preprocess(context).split()
        self.metadata = None


    def print(self):
        print("label", self.label)
        print("statement", self.statement)
        print("subject", self.subject)
        print("speaker", self.speaker)
        print("speaker job", self.speaker_job)
        print("state", self.state)
        print("party", self.party)
        print("context", self.context)
        print("meta data", self.metadata)
        return





def data_load(path,is_train,statement_word2num = {'unk': 0}, metadata_word2num = {"unk": 0}):
    data_samples = []
    sentences= []

    # read data
    file = open(path, "rb")
    lines = file.read()
    lines = lines.decode("utf-8")



    for line in lines.strip().split('\n'):
        # tmp[1]= label tmp[2]=statement tmp[3]=subjects tmp[4]= speaker tmp[5]=speaker_job  tmp[6]=state tmp[7]=party
        tmp_line = line.strip().split('\t')
        if len(tmp_line)!=14:
            tmp_line.append("none")
        # print(tmp_line)
        tmp = DataSample(tmp_line[1], tmp_line[2], tmp_line[3], tmp_line[4], tmp_line[5], tmp_line[6], tmp_line[7], tmp_line[13])
        # tmp.print()
        #statement
        if len(tmp.statement)<5:
            continue
        for i in range(len(tmp.statement)):
            tmp.statement[i] = count_in_vocab(statement_word2num, tmp.statement[i], is_train)
        #subject
        for i in range(len(tmp.subject)):
            tmp.subject[i] = count_in_vocab(metadata_word2num, tmp.subject[i],is_train)
        #speaker
        tmp.speaker = count_in_vocab(metadata_word2num, tmp.speaker,is_train)
        #speaker_job
        for i in range(len(tmp.speaker_job)):
            tmp.speaker_job[i] = count_in_vocab(metadata_word2num, tmp.speaker_job[i],is_train)
        #state
        tmp.state = count_in_vocab(metadata_word2num, tmp.state,is_train)
        #party
        tmp.party = count_in_vocab(metadata_word2num, tmp.party,is_train)
        #context
        for i in range(len(tmp.context)):
            tmp.context[i] = count_in_vocab(metadata_word2num, tmp.context[i],is_train)
        #metadata
        tmp.metadata = [tmp.speaker] + [tmp.state] + [tmp.party] + tmp.speaker_job + tmp.subject+tmp.context
        data_samples.append(tmp)
        sentences.append(tmp.sentence)
        # tmp.print()
    return data_samples, statement_word2num, metadata_word2num, sentences

