import random
import time
import json
import torch
import torch.nn as nn	 
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import gensim
import sys
reload(sys)
sys.setdefaultencoding('utf8') 


data_path = "../data/simple_question"
source_train_path = data_path + "/simple.v0.source.train"
source_test_path = data_path + "/simple.v0.source.test"
target_train_path = data_path + "/simple.v0.target.train"
target_test_path = data_path + "/simple.v0.target.test"
source_vocab_path = data_path + "/simple.v0.source.vocab"
target_vocab_path = data_path + "/simple.v0.target.vocab"

etype_train_path = data_path + "/simple.v0.source.etype.train"
etype_test_path = data_path + "/simple.v0.source.etype.test"

trs_train_path = data_path + "/simple.v0.source.trs.train"
trs_test_path = data_path + "/simple.v0.source.trs.test"

ps_train_path = data_path + "/simple.v0.source.ps.train"
ps_test_path = data_path + "/simple.v0.source.ps.test"

negrs_train_path = data_path + "/simple.v0.negrs.train"
negrs_test_path = data_path + "/simple.v0.negrs.test"

kse_t2v_path = "transh.fb2m.t2v"
kse_t2i_path = "transh.fb2m.t2i"
kse_r2v_path = "transh.fb2m.r2v"
kse_r2i_path = "transh.fb2m.r2i"
#tr2num_path = "transh.fb2m.tr2num.json"
w2v_path = "wikianswer.vectors.d200.bin"

all_type_list_path = data_path + "/../kb_data/freebase.FB2M.ts.json"

checkpoint_path = "checkpoint4"

source_vocab_size = None
target_vocab_size = None
encoder_dim_wf = 200
encoder_dim_pf = 10
encoder_dim_kf = 100
encoder_dim_mlp = 100
encoder_output_size = 310
decoder_input_size = 200
max_position_distance = 70
classifier_samples = 5
con_loss_lam = 0

learning_rates = [0.6, 0.15]

batch_size = 128
print_everys = [500,100]
save_everys = [1000,100]

part_epochs = [(8000, 2000, 1000), (100, 100, 100)]
#part_epochs = [(10, 10, 10), (10, 10, 10)]
turbo_num = 100

USE_CUDA = True
	
source_w2i = {}
source_i2w = {}
target_w2i = {}
target_i2w = {}
kse_t2v = {}
kse_t2i = {}
kse_i2t = {}
kse_r2v = {}
kse_r2i = {}
kse_i2r = {}
#tr2num = json.load(open(tr2num_path))

import model_apvaturbo
from model_apvaturbo import WFPFKFEncoderRNN, DecoderRNN, MLPClassifier
	
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


def extend_vocabs(word_list_path, w2i, i2w):
	word_list = json.load(open(word_list_path))
	for word in word_list:
		if word == "":
			continue
		if word not in w2i:
			i = len(w2i)
			w2i[word] = i
			i2w[i] = word
	return w2i, i2w
			

def read_docs_to_seqs(docs, w2i):
	seqs = []
	for doc in docs:
		words = doc.split(" ")
		seq = [w2i[word] for word in words if word != ""]
		seq.append(w2i["_EOS"])
		seqs.append(seq)
	return seqs
	

def replace_token_e(seqs, etypes, w2i):
	seqs_p = []
	etypes = [unicode(etype) for etype in etypes]
	for i,seq in enumerate(seqs):
		seq_p = []
		for word in seq:
			if word == w2i["<e>"]:
				seq_p.append(w2i[etypes[i]])
			else:
				seq_p.append(word)
		seqs_p.append(seq_p)
	return seqs_p
	
	
def load_tr_vocabs(vocab_path):
	name2i = {}
	i2name = {}
	lines = open(vocab_path).read().split("\n")
	for line in lines:
		if line == "":
			continue
		items = line.split("\t")
		name2i[items[0]] = int(items[1])
		i2name[int(items[1])] = items[0]
	return name2i, i2name

def load_tr_vectors(vectors_path):
	vectors = []
	lines = open(vectors_path).read().split("\n")
	for line in lines:
		if line == "":
			continue
		vec = [float(number) for number in line.split("\t") if number != ""]
		vectors.append(vec)
	return vectors
	
def read_trs_to_seqs(trs, t2i, r2i):
	t_all = []
	rs_all = []
	for tr in trs:
		items = tr.split("\t")
		t = items[1]
		rs = items[2].split(" ")
		if t in t2i:
			t_all.append(t2i[t])
		else:
			t_all.append(t2i["_None"])
		rs_all.append([r2i[r] for r in rs if r in r2i])
	return t_all, rs_all

def read_position_seqs(docs):
	seqs = []
	for doc in docs:
		seq = [int(p) for p in doc.split(" ")]
		seqs.append(seq)
	return seqs

	
def read_posrs_negrs(docs, r2i):
	posrs = []
	for doc in docs:
		posr = [r2i[r] for r in doc.split("\t") if r in r2i]
		posrs.append(posr)
	return posrs
	
	
def get_batch(pairs, batch_size_local, USE_NEG=False, neg_r2i=None):
	if batch_size_local is not None:
		pairs_batch = []
		while len(pairs_batch) < batch_size_local:
			pair = random.choice(pairs)
			pairs_batch.append(pair)
	else:
		batch_size_local = len(pairs)
		pairs_batch = pairs
	
	source_batch = []
	target_batch = []
	t_batch = []
	rs_batch = []
	ps_batch = []
	for pair in pairs_batch:
		source_batch.append(pair[0])
		target_batch.append(pair[1])
		t_batch.append(pair[2])
		rs_batch.append(pair[3])
		ps_batch.append(pair[4])
	rs_lengths = [len(seq) for seq in rs_batch]
	source_lengths = [len(seq) for seq in source_batch]
	target_lengths = [len(seq) for seq in target_batch]
	max_source_length = max(source_lengths)
	max_target_length = max(target_lengths)
	max_rs_length = max(rs_lengths)

	seqs_padded = []
	for seq in source_batch:
		seqs_padded.append(seq + [source_w2i["_PAD"] for pad_num in range(max_source_length - len(seq))])
	source_batch = seqs_padded
	
	seqs_padded = []
	for seq in target_batch:
		seqs_padded.append(seq + [target_w2i["_PAD"] for pad_num in range(max_target_length - len(seq))])
	target_batch = seqs_padded
	
	seqs_padded = []
	for seq in ps_batch:
		seqs_padded.append(seq + [max_position_distance-1 for pad_num in range(max_source_length - len(seq))])
	ps_batch = seqs_padded
	
	seqs_padded = []
	for seq in rs_batch:
		seqs_padded.append(seq + [-1 for pad_num in range(max_rs_length - len(seq))])
	rs_batch = seqs_padded
	
	t_batch_vec = []
	rs_batch_vec = []
	for i in range(len(t_batch)):
		t = t_batch[i]
		t_vec = kse_t2v[t]
		rs_vec = []
		for r in rs_batch[i]:
			r_vec = kse_r2v[r] if r != -1 else np.zeros(encoder_dim_kf)
			rs_vec.append(r_vec)
		t_batch_vec.append(t_vec)
		rs_batch_vec.append(rs_vec)
		
	source_batch = Variable(torch.LongTensor(source_batch)).transpose(0, 1) # (batch_size x max_len) tensors, transpose into (max_len x batch_size)
	target_batch = Variable(torch.LongTensor(target_batch)).transpose(0, 1)
	ps_batch = Variable(torch.LongTensor(ps_batch)).transpose(0, 1)
	t_batch = Variable(torch.FloatTensor(t_batch_vec))
	rs_batch = Variable(torch.FloatTensor(rs_batch_vec)).transpose(0, 1)
	
	if USE_CUDA:
		source_batch = source_batch.cuda()
		target_batch = target_batch.cuda()
		t_batch = t_batch.cuda()
		rs_batch = rs_batch.cuda()
		ps_batch = ps_batch.cuda()
	
	if not USE_NEG:
		return source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch
	else:
		samples = classifier_samples
		batch_size_p = batch_size_local * samples
		
		source_batch_p = source_batch.unsqueeze(2).expand(-1,-1,samples).contiguous().view(source_batch.size()[0], batch_size_p)
		rs_batch_p = rs_batch.unsqueeze(2).expand(-1,-1,samples,-1).contiguous().view(rs_batch.size()[0], batch_size_p, rs_batch.size()[-1])
		t_batch_p = t_batch.unsqueeze(1).expand(-1,samples,-1).contiguous().view(batch_size_p, t_batch.size()[-1])
		ps_batch_p = ps_batch.unsqueeze(2).expand(-1,-1,samples).contiguous().view(ps_batch.size()[0], batch_size_p)
		
		source_lengths_p = []
		rs_lengths_p = []
		class_r_batch = []
		label_batch = []
		for i in range(source_batch.size()[1]):
			source_lengths_p += [source_lengths[i]]*samples
			rs_lengths_p += [rs_lengths[i]]*samples
			
			samples_pos = 1
			samples_neg = samples - samples_pos
			posr_set = pairs_batch[i][5]
			negr_set = pairs_batch[i][6]
			while len(negr_set) < (samples_neg):
				negr = target_i2w[random.choice(range(len(target_i2w)))]
				if negr not in neg_r2i:
					continue
				else:
					negr = neg_r2i[negr]
				if (negr not in negr_set) and (negr not in posr_set):
					negr_set.append(negr)
			if posr_set != []:
				for loop_i in range(samples_pos):
					class_r_batch.append(random.choice(posr_set))
					label_batch.append(1)
			for loop_i in range(samples_neg):
				class_r_batch.append(random.choice(negr_set))
				label_batch.append(0)
		
		class_r_batch = Variable(torch.LongTensor(class_r_batch)).cuda()
		label_batch = Variable(torch.LongTensor(label_batch)).cuda()
		
		return source_batch_p, source_lengths_p, t_batch_p, rs_batch_p, rs_lengths_p, ps_batch_p, class_r_batch, label_batch
	
	
def run_epoch(source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch, encoder, decoder, encoder_optimizer=None, decoder_optimizer=None, TRAIN=True, relation_fliter=None, BEAM_SEARCH=None, Mr=None):
		
		if not TRAIN:
			encoder.train(False)
			decoder.train(False)

		B = source_batch.size()[1]
		encoder_hidden = encoder(source_batch, source_lengths, ps_batch, rs_batch, rs_lengths, t_batch)
		decoder_input = Variable(torch.LongTensor([target_w2i["_GO"]] * B)).cuda()
		decoder_hidden = encoder_hidden
		
		if TRAIN: # train
			max_target_length = max(target_lengths)
			all_decoder_outputs = Variable(torch.zeros(max_target_length, B, decoder.output_size)).cuda()
			for t in range(max_target_length):
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				all_decoder_outputs[t] = decoder_output
				decoder_input = target_batch[t]
			
			# S * B * vocab_size -> B * S * vocab_size
			all_decoder_outputs = all_decoder_outputs.transpose(0, 1).contiguous()
			target_batch = target_batch.transpose(0, 1).contiguous()
		
			loss = model_apvaturbo.masked_cross_entropy(all_decoder_outputs, target_batch, target_lengths)
			
			return loss
		else: # test
			if BEAM_SEARCH is None:
				# max_target_length = max(target_lengths)
				# all_decoder_outputs = Variable(torch.zeros(max_target_length, B, decoder.output_size)).cuda()
				# for t in range(max_target_length):
					# decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
					# all_decoder_outputs[t] = decoder_output
					# decoder_input = target_batch[t]
				# S * B * vocab_size -> B * S * vocab_size
				# all_decoder_outputs = all_decoder_outputs.transpose(0, 1).contiguous()
				# all_decoder_outputs = F.softmax(all_decoder_outputs, dim=2)
				# encoder.train(True)
				# decoder.train(True)
				# return all_decoder_outputs
				pass
			else:
				if relation_fliter is not None:
					r1_list = [target_w2i[r1] for r1 in relation_fliter["rel_1hop"] if r1 in target_w2i]
					r2_list = [target_w2i[r2] for r2 in relation_fliter["rel_2hop"] if r2 in target_w2i]
					r2_list.append(target_w2i['_EOS'])
				else:
					r1_list = [range(len(target_w2i))]
					r2_list = [range(len(target_w2i))]
				
				K = Mr
				
				# r1
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
				decoder_output = F.softmax(decoder_output, dim=1).data.cpu().numpy()
				predict_seqs = []
				
				for b in range(decoder_output.shape[0]):
					decoder_output_picked = np.array([decoder_output[b,ri] for ri in r1_list])
					decoder_output_picked_sort_index_list = np.argsort(-decoder_output_picked).tolist()
					decoder_output_picked_sort_index_list = decoder_output_picked_sort_index_list[:K]
					decoder_input_topK = [r1_list[pi] for pi in decoder_output_picked_sort_index_list]
					for k in range(len(decoder_input_topK)):
						predict_seqs.append([b, decoder_input_topK[k], None, decoder_output[b,decoder_input_topK[k]]])
				
				encoder.train(True)
				decoder.train(True)
				return predict_seqs, encoder_hidden
		

def calculate_precision(all_decoder_outputs, target_batch, target_lengths):
	target_batch = target_batch.transpose(0, 1).contiguous()
	hits = 0 
	for b in range(all_decoder_outputs.size()[0]):
		topv, topi = all_decoder_outputs[b].data.topk(1)
		pre = topi.squeeze(1)[:target_lengths[b]]
		sta = target_batch[b][:target_lengths[b]].data
		if torch.equal(pre, sta):
			hits += 1
	
	return float(hits)*100 / (all_decoder_outputs.size()[0])

def calculate_precision_class(predict_batch, label_batch):
	_, predicted = torch.max(predict_batch.data, 1)
	total = label_batch.size()[0]
	correct = (predicted == label_batch.data).sum()
	return 100*float(correct)/total
		

def load_pre_embedding(w2v_path, i2w):
	miss = 0
	w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
	matrix = np.zeros((source_vocab_size, encoder_dim_wf))
	for i in range(len(i2w)):
		word = i2w[i]
		if word in w2v_model:
			matrix[i] = w2v_model[word]
		else:
			matrix[i] = np.random.randn(encoder_dim_wf)
			miss += 1
	print "missed words: "+str(miss)+"/"+str(len(i2w))
	return matrix	

	
def get_random_pairs(pairs, pick_size):
	picked_pairs = []
	while len(picked_pairs) < pick_size:
		pair = random.choice(pairs)
		picked_pairs.append(pair)
	return picked_pairs	


def dicts_prepare():
	# read vocab
	global source_w2i, source_i2w, target_w2i, target_i2w, source_vocab_size, target_vocab_size, kse_t2i, kse_i2t, kse_r2i, kse_i2r, kse_t2v, kse_r2v
	source_w2i, source_i2w = load_vocabs(source_vocab_path)
	print "source vocab size: "+str(len(source_w2i))
	source_w2i, source_i2w = extend_vocabs(all_type_list_path, source_w2i, source_i2w)
	print "source vocab size (add types): "+str(len(source_w2i))
	target_w2i, target_i2w = load_vocabs(target_vocab_path);
	print "target vocab size: "+str(len(target_w2i))
	source_vocab_size = len(source_w2i)
	target_vocab_size = len(target_w2i)
	#read kse vecs
	kse_t2i, kse_i2t = load_tr_vocabs(kse_t2i_path)
	kse_r2i, kse_i2r = load_tr_vocabs(kse_r2i_path)
	kse_t2v = load_tr_vectors(kse_t2v_path)
	kse_r2v = load_tr_vectors(kse_r2v_path)
	print "kse t vocab size: "+str(len(kse_t2i))
	print "kse r vocab size: "+str(len(kse_r2i))
	
if __name__ == "__main__":
	
	print "(1) load data to seqs..."
	dicts_prepare()
	
	# read docs
	source_train = read_docs_to_seqs(open(source_train_path).read().strip().split("\n"), source_w2i);
	source_test = read_docs_to_seqs(open(source_test_path).read().strip().split("\n"), source_w2i);
	target_train = read_docs_to_seqs(open(target_train_path).read().strip().split("\n"), target_w2i);
	target_test = read_docs_to_seqs(open(target_test_path).read().strip().split("\n"), target_w2i);
	
	source_train = replace_token_e(source_train, open(etype_train_path).read().strip().split("\n"), source_w2i)
	source_test = replace_token_e(source_test, open(etype_test_path).read().strip().split("\n"), source_w2i)
	
	#read kse trs_vecs
	t_train, rs_train = read_trs_to_seqs(open(trs_train_path).read().split("\n"), kse_t2i, kse_r2i)
	t_test, rs_test = read_trs_to_seqs(open(trs_test_path).read().split("\n"), kse_t2i, kse_r2i)
	
	#read p_vecs
	ps_train = read_position_seqs(open(ps_train_path).read().strip().split("\n"))
	ps_test = read_position_seqs(open(ps_test_path).read().strip().split("\n"))
	
	#read posrs
	posrs_train = read_posrs_negrs(open(target_train_path).read().strip().split("\n"), target_w2i)
	posrs_test = read_posrs_negrs(open(target_test_path).read().strip().split("\n"), target_w2i)
	negrs_train = read_posrs_negrs(open(negrs_train_path).read().strip().split("\n"), target_w2i)
	negrs_test = read_posrs_negrs(open(negrs_test_path).read().strip().split("\n"), target_w2i)
	
	if (len(source_train) != len(target_train)) or (len(source_train) != len(t_train)) or (len(source_train) != len(ps_train)) or (len(source_train) != len(posrs_train)) or (len(source_train) != len(negrs_train)):
		print "train resource num does not match: "
		print len(source_train), len(target_train), len(t_train), len(ps_train), len(posrs_train), len(negrs_train)
		exit(0)
	if (len(source_test) != len(target_test)) or (len(source_test) != len(t_test)) or (len(source_test) != len(ps_test)) or (len(source_test) != len(posrs_test)) or (len(source_test) != len(negrs_test)):
		print "test resource num does not match: "
		print len(source_test), len(target_test), len(t_test), len(ps_test), len(posrs_test), len(negrs_test)
		exit(0)
	
	# combine
	train_pairs = zip(source_train, target_train, t_train, rs_train, ps_train, posrs_train, negrs_train)
	test_pairs = zip(source_test, target_test, t_test, rs_test, ps_test, posrs_test, negrs_test)
	print "train pairs: "+str(len(train_pairs))
	print "example: "
	rand = random.randint(0,len(train_pairs)-1)
	print [source_i2w[index] for index in train_pairs[rand][0]]
	print [target_i2w[index] for index in train_pairs[rand][1]]
	print kse_i2t[train_pairs[rand][2]]
	print [kse_i2r[index] for index in train_pairs[rand][3]]
	print train_pairs[rand][4]
	print [kse_i2r[index] for index in train_pairs[rand][5]]
	print [kse_i2r[index] for index in train_pairs[rand][6]]
	print "test pairs: "+str(len(test_pairs))
	print "example: "
	rand = random.randint(0,len(test_pairs)-1)
	print [source_i2w[index] for index in test_pairs[rand][0]]
	print [target_i2w[index] for index in test_pairs[rand][1]]
	print kse_i2t[test_pairs[rand][2]]
	print [kse_i2r[index] for index in test_pairs[rand][3]]
	print test_pairs[rand][4]
	print [kse_i2r[index] for index in test_pairs[rand][5]]
	print [kse_i2r[index] for index in test_pairs[rand][6]]
	
	print "(3) build model..."
	encoder = WFPFKFEncoderRNN(source_vocab_size, encoder_dim_wf, max_position_distance, encoder_dim_pf, encoder_dim_kf, encoder_dim_mlp, encoder_output_size)
	decoder = DecoderRNN(target_vocab_size, decoder_input_size, encoder_output_size)
	classifier = MLPClassifier(encoder_output_size+decoder_input_size, 2)
	classifier_loss = nn.CrossEntropyLoss()
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rates[0])
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rates[0])
	classifier_optimizer = optim.SGD(classifier.parameters(), lr=learning_rates[0])
	#encoder_optimizer_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=200, gamma=learning_rate_decay)
	#decoder_optimizer_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=200, gamma=learning_rate_decay)

	if USE_CUDA:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		classifier = classifier.cuda()
	print encoder
	print decoder
	print classifier
	
	print "(3.5) load pre_trained word2vec..."
	embedding_matrix = load_pre_embedding(w2v_path, source_i2w)
	encoder.WFPFEncoder.embedding_wf.weight.data.copy_(torch.from_numpy(embedding_matrix))
	#embedding_matrix = load_pre_embedding_decoder(target_i2w)
	#decoder.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
	
	print "(4) train model..."
	
	print_loss_pr = 0
	print_loss_ve = 0
	print_loss_total = 0
	
	for turbo in range(turbo_num):
		
		num_epochs = part_epochs[0] if turbo == 0 else part_epochs[1]
		print_every = print_everys[0] if turbo == 0 else print_everys[1]
		save_every = save_everys[0] if turbo == 0 else save_everys[1]
		lr = learning_rates[0] if turbo == 0 else learning_rates[1]
		
		if turbo == 1:
			for param_group in encoder_optimizer.param_groups:
				param_group['lr'] = lr
			for param_group in decoder_optimizer.param_groups:
				param_group['lr'] = lr
			for param_group in classifier_optimizer.param_groups:
				param_group['lr'] = lr
		
		for part in range(3):
		
			print "::::::::::::::::::::::::"
			print "::: Turbo "+str(turbo)+" : Part "+str(part)+" :::"
			print "::::::::::::::::::::::::"
			
			if part == 0:
				for param in encoder.parameters():
					param.requires_grad = False if turbo > 0 else True
				for param in decoder.parameters():
					param.requires_grad = True
				for param in classifier.parameters():
					param.requires_grad = False
			elif part == 1:
				for param in encoder.parameters():
					param.requires_grad = False
				for param in decoder.parameters():
					param.requires_grad = False
				for param in classifier.parameters():
					param.requires_grad = True
			elif part == 2:
				for param in encoder.parameters():
					param.requires_grad = True
				for param in decoder.parameters():
					param.requires_grad = False
				for param in classifier.parameters():
					param.requires_grad = False
		
			epochs = num_epochs[part]+1
			for iter in range(1,epochs):
				
				encoder_optimizer.zero_grad()
				decoder_optimizer.zero_grad()
				classifier_optimizer.zero_grad()
				
				#get batch
				pairs_batch_pr = get_random_pairs(train_pairs, batch_size)
				pairs_batch_ve = get_random_pairs(train_pairs, batch_size)
				
				# loss pr
				if (part == 0) or (part == 2):
					source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch = get_batch(pairs_batch_pr, None, USE_NEG=False)
					
					loss1 = run_epoch(source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, TRAIN=True)
				else:
					loss1 = None
				
				# loss ve			
				if (part == 1) or (part == 2):
					source_batch, source_lengths, t_batch, rs_batch, rs_lengths, ps_batch, class_r_batch, label_batch = get_batch(pairs_batch_ve, None, USE_NEG=True, neg_r2i=target_w2i)
					
					encoder_hidden = encoder(source_batch, source_lengths, ps_batch, rs_batch, rs_lengths, t_batch)
					class_r_batch = decoder.embedding(class_r_batch)
					predict_batch = classifier(encoder_hidden, class_r_batch)
					loss2 = classifier_loss(predict_batch, label_batch)/classifier_samples
				else:
					loss2 = None
				
				# loss total
				if (loss1 is not None) and (loss2 is not None):
					loss_total = con_loss_lam*loss1 + loss2
				else:
					loss_total = None
				
				# update
				if part == 0:
					loss1.backward()
					if turbo == 0:
						encoder_optimizer.step()
					decoder_optimizer.step()
				elif part == 1:
					loss2.backward()
					classifier_optimizer.step()
				elif part == 2:
					loss_total.backward()
					encoder_optimizer.step()
				
				# print
				print_loss_pr += loss1.data[0] if loss1 is not None else 0
				print_loss_ve += loss2.data[0] if loss2 is not None else 0
				print_loss_total += loss_total.data[0] if loss_total is not None else 0
				
				if iter % print_every == 0:
					print "-----------------------------"
					print "iter " + str(iter) + "/" + str(epochs)
					print "time: "+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
					print_loss_pr_avg = (print_loss_pr / print_every) if iter > 0 else print_loss_pr
					print_loss_ve_avg = (print_loss_ve / print_every) if iter > 0 else print_loss_ve
					print_loss_total_avg = (print_loss_total / print_every) if iter > 0 else print_loss_total
					print_loss_pr = 0
					print_loss_ve = 0
					print_loss_total = 0
					print "loss(pr): "+str(print_loss_pr_avg)
					print "loss(ve): "+str(print_loss_ve_avg)
					print "loss(total): "+str(print_loss_total_avg)
				
				if (iter % save_every == 0) and (part == 0):
					torch.save(encoder, checkpoint_path+"/encoder.model.turbo"+str(turbo)+".iter"+str(iter)+".pth")
					torch.save(decoder, checkpoint_path+"/decoder.model.turbo"+str(turbo)+".iter"+str(iter)+".pth")
					torch.save(classifier, checkpoint_path+"/classifier.model.turbo"+str(turbo)+".iter"+str(iter)+".pth")
	
		
		
		
	
		
	
	
	


