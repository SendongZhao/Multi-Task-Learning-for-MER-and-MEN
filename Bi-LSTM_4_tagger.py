import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import os

os.environ["OMP_NUM_THREADS"] = "3"

start = datetime.datetime.now()

#make every sentence to index sequence
def prepare_sequence(seq, to_ix):
	idxs = []
	for w in seq:
		if not to_ix.has_key(w):
			idxs.append(0)
		else:
			idxs.append(to_ix[w])
	tensor = torch.LongTensor(idxs)
	return Variable(tensor)

#Read data from file
def read_corpus(filename):
	data = []
	data_string_list = list(open(filename))

	print len(data_string_list)

	element_size = 0
	X = list()
	Y = list()

	for data_string in data_string_list:
		words = data_string.strip().split()
		if len(words) is 0:
			data.append((X,Y))
			X = list()
			Y = list()
		else:
			if element_size is 0:
				element_size = len(words)
			#print (words[0])
			#print (words[-1])
			X.append(words[0])
			Y.append(words[-1])
	if len(X)>0:
		data.append((X,Y))

	return data

training_data = read_corpus('train')
print('len of train data')
print(len(training_data))

testing_data = read_corpus('test')
print ('len of test data')
print (len(testing_data))

word_to_ix = {}
tag_to_ix = {}

for sent, tags in training_data:
	for word in sent:
		 if word not in word_to_ix:
		 	 word_to_ix[word] = len(word_to_ix)

	for tag in tags:
		if tag not in tag_to_ix:
			tag_to_ix[tag] = len(tag_to_ix)

print ("len of dictionary")
print (len(word_to_ix))
print ("len of tag")
print (len(tag_to_ix))

EMBEDDING_DIM = 200
HIDDEN_DIM = 200

#print (word_to_ix)
#print (tag_to_ix)

class LSTMTagger(nn.Module):
	"""docstring for LSTMTagger"""
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers = 1, bidirectional = True)
		
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (Variable(torch.randn(2, 1, self.hidden_dim // 2)),Variable(torch.randn(2, 1, self.hidden_dim // 2)))

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)

		lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)

		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

		tag_score = F.log_softmax(tag_space)

		_, tag_seq = torch.max(tag_score, 1)

		return tag_score, tag_seq


#training
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

loss_fn = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(20):

	print ("in epoch %d" % epoch)

	for sentence, tags in training_data:
	    
		model.zero_grad()

		model.hidden = model.init_hidden()

		sentence_in = prepare_sequence(sentence, word_to_ix)
		
		targets = prepare_sequence(tags, tag_to_ix)

		tag_scores, _ = model(sentence_in)

		loss = loss_fn(tag_scores, targets)

		#print (loss.data[0])

		loss.backward()

		optimizer.step()

#testing
total_count = 0
correct_count = 0
wrong_count = 0

print ("start testing")

for sentence, tags in testing_data:
    
	sentence_in = prepare_sequence(sentence, word_to_ix)
	targets = prepare_sequence(tags, tag_to_ix)
	#print (targets)
	tag_scores, idx = model(sentence_in)
	#print (idx)
	#print (tag_scores)
	
	for t in range(len(targets)):
		total_count += 1
		index = idx[t].data[0]
		
		if targets[t].data[0] == index:
			correct_count += 1
		else:
			wrong_count += 1
            #print ("&&&&&&&&&&&")
            #print targets[t].data[0], index

print('Correct: %d' % correct_count)
print('Wrong: %d' % wrong_count)
print('Total: %d' % total_count)
print('Performance: %f' % (float(correct_count)/float(total_count)))

end = datetime.datetime.now()

print (end - start)
