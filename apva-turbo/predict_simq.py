import random
import time
import json
import torch
import torch.nn as nn	 
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F	
import sys
import leveldb
reload(sys)
sys.setdefaultencoding('utf8') 

import train_apvaturbo as train

train.dicts_prepare()

source_w2i, source_i2w, target_w2i, target_i2w = train.source_w2i, train.source_i2w, train.target_w2i, train.target_i2w
kse_t2i, kse_i2t, kse_r2i, kse_i2r, kse_t2v, kse_r2v = train.kse_t2i, train.kse_i2t, train.kse_r2i, train.kse_i2r, train.kse_t2v, train.kse_r2v

lam_test = 0

# def predict_seq2seq():
	# print "testing seq2seq part..."
	# source_test = train.read_docs_to_seqs(open(train.source_test_path).read().split("\n"), source_w2i);
	# target_test = train.read_docs_to_seqs(open(train.target_test_path).read().split("\n"), target_w2i);
	# t_test, rs_test = train.read_trs_to_seqs(open(train.trs_test_path).read().split("\n"), t2i, r2i)
	# test_pairs = zip(source_test, target_test, t_test, rs_test)
	# test_data_num = len(test_pairs)
	# print "test data num: " + str(test_data_num)
	
	# hits = 0
	# batch_size = 5
	# index = 0
	
	# while index < test_data_num:
		# print str(index)+"/"+str(test_data_num)
		# source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch = train.get_batch(test_pairs[index:index+batch_size], None)
		
		# all_decoder_outputs = train.run_epoch(source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch, encoder, decoder, TRAIN=False)
		# topv, predicts = all_decoder_outputs.data.topk(1)
		# predicts = predicts.squeeze(2)
		# targets = target_batch.transpose(0, 1).data
		# for b in range(all_decoder_outputs.size()[0]):
			# for t in range(all_decoder_outputs.size()[1]):
				# if predicts[b][t] == target_w2i["_EOS"]:
					# predicts[b] = predicts[b][:t+1]
				# if targets[b][t] == target_w2i["_EOS"]:
					# targets[b] = targets[b][:t+1]
			# if torch.equal(predicts[b],targets[b]):
				# hits += 1
		
		
		# index += batch_size
		
	# print "seq2seq precision: " +str(float(hits) / test_data_num)
	
	
def run_class(encoder_hidden, can_er_pairs, decoder, classifier):
	
	can_e_list = [can_er_pair[0] for can_er_pair in can_er_pairs]
	can_r_list = [can_er_pair[1] for can_er_pair in can_er_pairs]
	#can_r_list = [target_i2w[r] for r in can_r_list]
	
	encoder_hidden_batch = Variable(torch.zeros(encoder_hidden.size()[0], len(can_er_pairs), encoder_hidden.size()[2])).cuda()
	for i in range(len(can_er_pairs)):
		encoder_hidden_batch[:,i,:] = encoder_hidden[:,can_e_list[i],:]
	
	class_r_batch = can_r_list
	class_r_batch = Variable(torch.LongTensor(class_r_batch)).cuda()	
	class_r_batch = decoder.embedding(class_r_batch)
	
	predict_batch = classifier(encoder_hidden_batch, class_r_batch)
	predict_batch = F.softmax(predict_batch, dim=1)
	
	return predict_batch[:,1].data.cpu().numpy()
	
	
def predict_all_with_nel(ck_path, stage_num, iter_num):
	
	checkpoint_path = ck_path

	# load model
	encoder = torch.load(checkpoint_path+"/encoder.model.turbo"+str(stage_num)+".iter"+str(iter_num)+".pth")
	decoder = torch.load(checkpoint_path+"/decoder.model.turbo"+str(stage_num)+".iter"+str(iter_num)+".pth")
	classifier = torch.load(checkpoint_path+"/classifier.model.turbo"+str(stage_num)+".iter"+str(iter_num)+".pth")
	
	# load types and relations
	type_db = leveldb.LevelDB("../data/kb_data/mid_type.db")
	e2rs = json.load(open("../data/kb_data/e2rs.fb2m.json"))
	
	# load can_entities
	test_data_with_can = open("../data/simple_question/simple.test.nel.fb2m.can100").read().strip().split("\n")
	cans = [json.loads(data_with_can)["can_entity"] for data_with_can in test_data_with_can]
	gold_es = [json.loads(data_with_can)["topic_entity"] for data_with_can in test_data_with_can]
	gold_rs = [json.loads(data_with_can)["rel"] for data_with_can in test_data_with_can]
	questions = [json.loads(data_with_can)["q"] for data_with_can in test_data_with_can]
	
	# load test data
	source_test = train.read_docs_to_seqs(open(train.source_test_path).read().strip().split("\n"), source_w2i);
	target_test = train.read_docs_to_seqs(open(train.target_test_path).read().strip().split("\n"), target_w2i);
	ps_test = train.read_position_seqs(open(train.ps_test_path).read().strip().split("\n"))
	test_data_num = len(source_test)
	
	f_out = open("predict.this.fb2m","w")
	f_out_errors = open("predict.error.fb2m","w")
	hits = 0
	num = 0
	num1 = 0
	
	for i in range(test_data_num):

		if i%2000 == 0:
			print str(i)+"/"+str(test_data_num)
		can_entities = cans[i][:5]
		gold_e = gold_es[i]	
		gold_r = gold_rs[i]
		gold_score = None
		pred_e = None		
		pred_r = None
		
		if gold_e in can_entities:
			# make test batch for i-th test data
			t_test_i = []
			rs_test_i = []
			can_entities_i = []
			can_etypes = []
			for can_entity in can_entities:
				try:
					can_etype = type_db.Get(can_entity)
				except KeyError:
					can_etype = "_None"
				can_e_t = kse_t2i[can_etype]
				can_e_rs = e2rs[can_entity] if can_entity in e2rs else []
				can_e_rs = [kse_r2i[can_e_r] for can_e_r in can_e_rs if can_e_r in kse_r2i]
				if can_e_rs == []:
					continue
				t_test_i.append(can_e_t)
				rs_test_i.append(can_e_rs)
				can_entities_i.append(can_entity)
				can_etypes.append(can_etype)
			i_size = len(can_entities_i)
			source_test_i = [source_test[i]]*i_size
			source_test_i = train.replace_token_e(source_test_i, can_etypes, source_w2i)
			target_test_i = [target_test[i]]*i_size
			ps_test_i = [ps_test[i]]*i_size
			
			test_pairs = zip(source_test_i, target_test_i, t_test_i, rs_test_i, ps_test_i)
			
			# run model
			source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch = train.get_batch(test_pairs, None)
			relation_fliter = {"rel_1hop": e2rs[gold_e], "rel_2hop": []}
			
			can_er_pairs, encoder_hidden = train.run_epoch(source_batch, source_lengths, target_batch, target_lengths, t_batch, rs_batch, rs_lengths, ps_batch, encoder, decoder, TRAIN=False, relation_fliter=relation_fliter, BEAM_SEARCH=1, Mr=10)
			
			class_scores = run_class(encoder_hidden, can_er_pairs, decoder, classifier)
			
			for er_i in range(len(can_er_pairs)):
				can_er_pairs[er_i].append(class_scores[er_i])
				can_er_pairs[er_i].append(lam_test*can_er_pairs[er_i][-2] + can_er_pairs[er_i][-1])
				
			can_er_pairs = sorted(can_er_pairs, key=lambda l:l[-1], reverse=True)
			
			# for i in range(len(can_er_pairs)):
				# for j in range(i+1, len(can_er_pairs)):
					# if can_er_pairs[i][-1] == can_er_pairs[j][-1]:
						# e1_num = can_er_pairs[i][0]
						# e2_num = can_er_pairs[j][0]
						# if torch.equal(encoder_hidden[:,e1_num,:], encoder_hidden[:,e2_num,:]):
							# if (not torch.equal(t_batch[e1_num, :], t_batch[e2_num, :])) or (not torch.equal(rs_batch[:,e1_num,:], rs_batch[:,e2_num,:])) or (not torch.equal(source_batch[:, e1_num], source_batch[:, e2_num])):
								# print "2!!!!!!!!!!!!!!!2"
								# print encoder_hidden[:,e1_num,:],encoder_hidden[:,e2_num,:]
								# print t_batch[e1_num, :], t_batch[e2_num, :]
								# print rs_batch[:,e1_num,:],rs_batch[:,e2_num,:]
						# else:
							# print "1!!!!!!!!!!!!!!!"
							# print encoder_hidden[:,e1_num,:],encoder_hidden[:,e2_num,:]
							# print can_er_pairs[i][-1], can_er_pairs[j][-1]
					
			pred_e = can_entities_i[can_er_pairs[0][0]]
			pred_r = target_i2w[can_er_pairs[0][1]]
			pred_score = can_er_pairs[0][-1]
			
			for can_er_pair in can_er_pairs:
				if gold_e == can_entities_i[can_er_pair[0]]:
					if gold_r == target_i2w[can_er_pair[1]]:
						gold_score = can_er_pair[-1]
		
		result = {}
		result["q"] = questions[i]
		result["gold_e"] = gold_e
		result["gold_r"] = gold_r
		result["gold_score"] = gold_score
		result["pred_e"] = pred_e
		result["pred_r"] = pred_r
		result["pred_score"] = pred_score
		result["can_entity"] = can_entities
		
		
		if (gold_e == pred_e) and (gold_r == pred_r):
			hits += 1 
		else:
			f_out_errors.write(json.dumps(result)+"\n")
	
	acc = float(hits)*100 / test_data_num
	print "Accuracy: " +str(acc)
	#print num
	#print num1
	f_out.close()
	f_out_errors.close()
	return acc	
	
	
if __name__ == "__main__":
	ck_num = "checkpoint4"
	
	# print predict_all_with_nel(ck_num,31,0)
	# exit(0)
	
	# lam_list = [0,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
	# for lam in lam_list:
		# lam_test = lam
		# f1 = predict_all_with_nel(ck_num,49,150)
		# print lam_test, f1
	# exit(0)
	
	lam_test = 10000
	
	f1s = [predict_all_with_nel(ck_num,0,8000)]
	print 0,f1s[0]

	lam_test = 0
	
	for i in range(1,100,5):
		f1 = predict_all_with_nel(ck_num,i,100)
		print i,f1
		f1s.append(f1)
	f1s = [str(f1) for f1 in f1s]
	print " ".join(f1s)
	