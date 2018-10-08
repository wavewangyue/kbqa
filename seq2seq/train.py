import random
import time
import torch
import torch.nn as nn	 
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Config(object):
	data_path = "."
	source_train_path = data_path + "/source.train"
	source_test_path = data_path + "/source.test"
	target_train_path = data_path + "/target.train"
	target_test_path = data_path + "/target.test"
	source_vocab_path = data_path + "/source.vocab"
	target_vocab_path = data_path + "/target.vocab"
	checkpoint_path = "checkpoint"
	
	hidden_size = 256
	batch_size = 64
	learning_rate = 0.5
	
	num_epoch = 10000
	print_every = 200
	save_every = 1000

USE_CUDA = True
	
source_w2i = {}
source_i2w = {}
target_w2i = {}
target_i2w = {}

from seq2seq_model import EncoderRNN, DecoderRNN
import seq2seq_model as seq2seq_model	
	
def load_vocabs(vocab_path):
	w2i = {}
	i2w = {}
	vocab = open(vocab_path).read().split("\n")
	for word in vocab:
		if word == "":
			continue
		i = len(w2i)
		w2i[word] = i
		i2w[i] = word
	return w2i, i2w

	
def read_docs_to_seqs(doc_path, w2i):
	docs = open(doc_path).read().split("\n")
	seqs = []
	for doc in docs:
		if doc == "":
			continue
		words = doc.split(" ")
		seq = [w2i[word] for word in words if word in w2i]
		seq.append(w2i["_EOS"])
		seqs.append(seq)
	return seqs

	
def get_batch(pairs, batch_size):
	if batch_size is not None:
		rand_list = [random.randint(0, len(pairs)-1) for i in range(batch_size)]
		pairs_batch = [pairs[rand] for rand in rand_list]
	else:
		pairs_batch = pairs
	#pairs_batch = sorted(pairs_batch, key=lambda p:len(p[0]), reverse=True) # sort based on input len, to use pack function of pytorch
	
	source_batch = [pair[0] for pair in pairs_batch]
	target_batch = [pair[1] for pair in pairs_batch]
	source_lengths = [len(seq) for seq in source_batch]
	target_lengths = [len(seq) for seq in target_batch]
	max_source_length = max(source_lengths)
	max_target_length = max(target_lengths)

	seqs_padded = []
	for seq in source_batch:
		seqs_padded.append(seq + [source_w2i["_PAD"] for pad_num in range(max_source_length - len(seq))])
	source_batch = seqs_padded
	seqs_padded = []
	for seq in target_batch:
		seqs_padded.append(seq + [target_w2i["_PAD"] for pad_num in range(max_target_length - len(seq))])
	target_batch = seqs_padded
	
	source_batch = Variable(torch.LongTensor(source_batch)).transpose(0, 1) # (batch_size x max_len) tensors, transpose into (max_len x batch_size)
	target_batch = Variable(torch.LongTensor(target_batch)).transpose(0, 1)
	if USE_CUDA:
		source_batch = source_batch.cuda()
		target_batch = target_batch.cuda()
	return source_batch, source_lengths, target_batch, target_lengths
	

def run_epoch(source_batch, source_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer=None, decoder_optimizer=None, TRAIN=True):
		
		if TRAIN:
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()
			loss = 0
		else:
			encoder.train(False)
			decoder.train(False)
		
		batch_size = source_batch.size()[1]
		encoder_outputs, encoder_hidden = encoder(source_batch, source_lengths, None)
		decoder_input = Variable(torch.LongTensor([target_w2i["_GO"]] * batch_size))
		decoder_hidden = encoder_hidden
		max_target_length = max(target_lengths)
		all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
		
		if USE_CUDA:
			decoder_input = decoder_input.cuda()
			all_decoder_outputs = all_decoder_outputs.cuda()
		
		for t in range(max_target_length):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
			all_decoder_outputs[t] = decoder_output
			decoder_input = target_batch[t]
		
		# S * B * vocab_size -> B * S * vocab_size
		all_decoder_outputs = all_decoder_outputs.transpose(0, 1).contiguous()
		target_batch = target_batch.transpose(0, 1).contiguous()
		
		if TRAIN: # train
			loss = seq2seq_model.masked_cross_entropy(all_decoder_outputs, target_batch, target_lengths)
			loss.backward()
			encoder_optimizer.step()
			decoder_optimizer.step()
			return loss.data[0]
		else: # test
			preds = []
			hits = 0
			for b in range(batch_size):
				topv, topi = all_decoder_outputs[b].data.topk(1)
				pre = topi.squeeze(1)[:target_lengths[b]]
				sta = target_batch[b][:target_lengths[b]].data
				if torch.equal(pre, sta):
					hits += 1
				preds.append(pre.cpu().numpy().tolist())
			encoder.train(True)
			decoder.train(True)
			return float(hits)*100 / batch_size, preds
		
	
if __name__ == "__main__":
	config = Config()
	
	print "(1) load data to seqs..."
	source_w2i, source_i2w = load_vocabs(config.source_vocab_path);
	target_w2i, target_i2w = load_vocabs(config.target_vocab_path);
	config.source_vocab_size = len(source_w2i)
	config.target_vocab_size = len(target_w2i)
	source_train = read_docs_to_seqs(config.source_train_path, source_w2i);
	source_test = read_docs_to_seqs(config.source_test_path, source_w2i);
	target_train = read_docs_to_seqs(config.target_train_path, target_w2i);
	target_test = read_docs_to_seqs(config.target_test_path, target_w2i);
	
	train_pairs = [(s_seq, t_seq) for s_seq, t_seq in zip(source_train, target_train)]
	test_pairs = [(s_seq, t_seq) for s_seq, t_seq in zip(source_test, target_test)]
	print "train pairs: "+str(len(train_pairs))
	print "example: "
	rand = random.randint(0,len(train_pairs)-1)
	print [source_i2w[index] for index in train_pairs[rand][0]]
	print [target_i2w[index] for index in train_pairs[rand][1]]
	print "test pairs: "+str(len(test_pairs))
	print "example: "
	rand = random.randint(0,len(test_pairs)-1)
	print [source_i2w[index] for index in test_pairs[rand][0]]
	print [target_i2w[index] for index in test_pairs[rand][1]]
	
	'''
	print "(2) split seqs to buckets..."
	train_pairs_buckets = split_seqs_to_buckets(train_pairs, config.buckets)
	for i in range(len(config.buckets)):
		print "buckets "+str(i)+": "+str(config.buckets[i])+" : "+str(len(train_pairs_buckets[i]))
	'''
	
	print "(3) build model..."
	encoder = EncoderRNN(config)
	decoder = DecoderRNN(config)
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.learning_rate)
	criterion = nn.NLLLoss()
	if USE_CUDA:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
	print encoder
	print decoder
	
	print "(4) train model..."
	print_loss_total = 0  # Reset every print_every
	num_epoch = config.num_epoch
	print_every = config.print_every
	save_every = config.save_every
	batch_size = config.batch_size
	
	for iter in range(0, num_epoch):

		source_batch, source_lengths, target_batch, target_lengths = get_batch(train_pairs, batch_size)
	
		loss = run_epoch(source_batch, source_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, TRAIN=True)
		
		print_loss_total += loss
		if iter % print_every == 0:
			print "-----------------------------"
			print "iter " + str(iter) + "/" + str(num_epoch)
			print "time: "+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
			print_loss_avg = (print_loss_total / print_every) if iter > 0 else print_loss_total
			print_loss_total = 0
			print "loss: "+str(print_loss_avg)	
			
			source_batch, source_lengths, target_batch, target_lengths = get_batch(test_pairs, batch_size)
			precision, preds = run_epoch(source_batch, source_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, TRAIN=False)
			print "precision: "+str(precision)
			
		if iter % save_every == 0:
			torch.save(encoder, config.checkpoint_path+"/encoder.model.iter"+str(iter)+".pth")
			torch.save(decoder, config.checkpoint_path+"/decoder.model.iter"+str(iter)+".pth")	