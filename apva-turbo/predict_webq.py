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

source_w2i, source_i2w = train.load_vocabs(train.source_vocab_path)
target_w2i, target_i2w = train.load_vocabs(train.target_vocab_path)
t2i, i2t = train.load_tr_vocabs(train.t2i_path)
r2i, i2r = train.load_tr_vocabs(train.r2i_path)
t2v = train.load_tr_vectors(train.t2v_path)
r2v = train.load_tr_vectors(train.r2v_path)
tr2num = json.load(open(train.tr2num_path))

train.source_w2i, train.source_i2w = source_w2i, source_i2w
train.target_w2i, train.target_i2w = target_w2i, target_i2w
train.t2i, train.i2t = t2i, i2t
train.r2i, train.i2r = r2i, i2r
train.t2v, train.r2v = t2v, r2v
train.tr2num = tr2num

lam_test = 0
'''
def predict_seq2seq():
	print "testing seq2seq part..."
	source_test = train.read_docs_to_seqs(open(train.source_test_path).read().split("\n"), source_w2i);
	target_test = train.read_docs_to_seqs(open(train.target_test_path).read().split("\n"), target_w2i);
	t_test, rs_test = train.read_trs_to_seqs(open(train.trs_test_path).read().split("\n"), t2i, r2i)
	test_pairs = zip(source_test, target_test, t_test, rs_test)
	test_data_num = len(test_pairs)
	print "test data num: " + str(test_data_num)
	
	hits = 0
	batch_size = 256
	index = 0
	num = 0
	fout = open("predict.error.seq2seq","w")
	
	while index < test_data_num:
		print str(index)+"/"+str(test_data_num)
		source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch = train.get_batch(test_pairs[index:index+batch_size], None)
		
		all_decoder_outputs, predicts = train.run_epoch(source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch, encoder, decoder, TRAIN=False)
		targets = target_batch.transpose(0, 1).data
		
		for b in range(all_decoder_outputs.shape[0]):
			predicts_this = predicts[b]
			targets_this = targets[b].cpu().numpy().tolist()[:len(predicts_this)]
			
			for t in range(all_decoder_outputs.shape[1]):
				if predicts[b][t] == target_w2i["_EOS"]:
					predicts_this = predicts_this[:t]
					break
			for t in range(all_decoder_outputs.shape[1]):
				if targets[b][t] == target_w2i["_EOS"]:
					targets_this = targets_this[:t]
					break
					
			if predicts_this == targets_this:
				hits += 1
			else:
				predicts_this = [target_i2w[ti] for ti in predicts_this]
				targets_this = [target_i2w[ti] for ti in targets_this]
				fout.write(str(index+b)+" "+str(predicts_this)+" "+str(targets_this)+"\n")

		index += batch_size
	
	print num	
	print "seq2seq precision: " +str(float(hits) / test_data_num)
'''	

# def cos_similarity(v1,v2):
	# A = np.array(v1)
	# B = np.array(v2)
	# num = np.dot(A, B)
	# denom = np.linalg.norm(A) * np.linalg.norm(B)  
	# cos = num / denom 
	# sim = 0.5 + 0.5 * cos 
	# return sim

	
def run_class(source_batch, source_lengths, ps_batch, rs_batch, rs_lengths, t_batch, can_rs, encoder, decoder, classifier):
	encoder_hidden = encoder(source_batch, source_lengths, ps_batch, rs_batch, rs_lengths, t_batch)
	scores = []
	for can_r in can_rs:
		class_r_batch = Variable(torch.LongTensor([can_r])).cuda()
		class_r_batch = decoder.embedding(class_r_batch)
		predict_batch = classifier(encoder_hidden, class_r_batch)
		predict_batch = F.softmax(predict_batch, dim=1).data
		scores.append(predict_batch[0][1])
	return scores
	
	
def predict_all_with_nel(ck_path, stage_num, iter_num):
	checkpoint_path = ck_path

	encoder = torch.load(checkpoint_path+"/encoder.model.stage"+str(stage_num)+".iter"+str(iter_num)+".pth")
	decoder = torch.load(checkpoint_path+"/decoder.model.stage"+str(stage_num)+".iter"+str(iter_num)+".pth")
	classifier = torch.load(checkpoint_path+"/classifier.model.stage"+str(stage_num)+".iter"+str(iter_num)+".pth")
	
	#e2r2t = json.load(open("../data/web_question/wq.e2r2t.json"))
	#mid2name = json.load(open("../data/web_question/wq.test.mid2name.json"))
	#type_db = leveldb.LevelDB("../data/kb_data/mid_type.db")
	e2rs = json.load(open("../data/web_question/en_rel.wq.json"))
	test_data_with_can = open("../data/web_question/wq.test.nel.can20.resort").read().split("\n")
	#test_data_with_can = open("../data/web_question/wq.test.nel.api.can20").read().split("\n")
	cans = [json.loads(data_with_can)["can_entity"] for data_with_can in test_data_with_can if data_with_can != ""]
	gold_es = [json.loads(data_with_can)["topic_entity"] for data_with_can in test_data_with_can if data_with_can != ""]
	questions = [json.loads(data_with_can)["question"] for data_with_can in test_data_with_can if data_with_can != ""]
	#target_test = open(train.target_test_path).read().split("\n")
	#gold_rs = [target for target in target_test]
	#paths2answers_test = open("../data/web_question/wq.test.goldanswers.p2as.all").read().split("\n")
	paths2answers_test = open("../data/web_question/wq.test.goldanswers.p2as.svmed").read().split("\n")
	#paths2answers_test = open("../data/web_question/wq.test.goldanswers.p2as.shuffle").read().split("\n")
	p2ass = [json.loads(data)["paths2answers"] for data in paths2answers_test if data != ""]
	gold_answers = [json.loads(data)["answers"] for data in paths2answers_test if data != ""]
	
	source_test = train.read_docs_to_seqs(open(train.source_test_path).read().split("\n"), source_w2i);
	target_test = train.read_docs_to_seqs(open(train.target_test_path).read().split("\n"), target_w2i);
	t_test, rs_test = train.read_trs_to_seqs(open(train.trs_test_path).read().split("\n"), t2i, r2i)
	ps_test = train.read_position_seqs(open(train.ps_test_path).read().split("\n"))
	test_pairs = zip(source_test, target_test, t_test, rs_test, ps_test)

	f_out = open("predict.all","w")
	f_out_error = open("predict.error.noV","w")
	
	test_data_num = len(source_test)
	precision_sum = 0
	recall_sum = 0
	f1_sum = 0
	hits = 0
	hits_r = 0
	index = 0
	num = 0
	
	for i in range(test_data_num):
		# if i%100 == 0:
			# print str(i)+"/"+str(test_data_num)
		
		question = questions[i]
		can_entities = cans[i][:20]
		p2as = p2ass[i]
		gold_e = gold_es[i]
		gold_r = [path for path in p2as]
		gold_answer = [ans for ans in gold_answers[i] if ans != ""]
		
		pred_entity = None
		pred_relation = None
		
		if (gold_e in can_entities) and (gold_r != []):
		
			gold_r1s = [r.split(" ")[0] for r in gold_r]
			gold_r2s = [r.split(" ")[1] for r in gold_r if len(r.split(" "))>1]
			if any(r1 not in e2rs[gold_e]["rel_1hop"] for r1 in gold_r1s) or any(r2 not in e2rs[gold_e]["rel_2hop"] for r2 in gold_r2s):
				print gold_e
				print gold_r
				print e2rs[gold_e]
				exit(0)
		
			source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch, ps_batch = train.get_batch([test_pairs[i]], None)
			relation_fliter = e2rs[gold_e]
			
			predict_seqs = train.run_epoch(source_batch, source_lengths, target_batch, target_lengths, rs_batch, rs_lengths, t_batch, ps_batch, encoder, decoder, TRAIN=False, relation_fliter=relation_fliter, BEAM_SEARCH=3, Mr=10)	
			
			can_rs = [predict_seq[0] for predict_seq in predict_seqs]
			class_scores = run_class(source_batch, source_lengths, ps_batch, rs_batch, rs_lengths, t_batch, can_rs, encoder, decoder, classifier)
			for si in range(len(predict_seqs)):
				predict_seqs[si].append(class_scores[si])
				predict_seqs[si].append(lam_test*predict_seqs[si][2] + predict_seqs[si][3])
			predict_seqs = sorted(predict_seqs, key=lambda l:l[4], reverse=True)
			
			predict_seq = predict_seqs[0]
			r1_predict = target_i2w[predict_seq[0]]
			r2_predict = target_i2w[predict_seq[1]] if predict_seq[1] != target_w2i["_EOS"] else ""
			
			pred_relation = r1_predict+" "+r2_predict if r2_predict != "" else r1_predict
			
			for can_entity in can_entities:
				rs_1hop = e2rs[can_entity]["rel_1hop"] if can_entity in e2rs else []
				rs_2hop = e2rs[can_entity]["rel_2hop"] if can_entity in e2rs else []
					
				for r1 in rs_1hop:
					if r1 == pred_relation:
						if r1 in p2as:
							pred_entity = can_entity
							# pred_relation = r1
							break
					else:
						for r2 in rs_2hop:
							if (r1+" "+r2) == pred_relation:
								if (r1+" "+r2) in p2as:
									pred_entity = can_entity
									# pred_relation = r1+" "+r2
									break
										
				if pred_entity is not None:
					break
		
		result = {}
		result["q"] = question
		result["can_entity"] = can_entities
		result["gold_e"] = gold_e
		result["gold_r"] = gold_r
		result["gold_answers"] = gold_answer
		result["pred_e"] = pred_entity
		result["pred_r"] = pred_relation
		result["pred_answers"] = p2as[pred_relation] if pred_relation in p2as else []
		# f_out.write(json.dumps(result)+"\n")
		
		if (gold_e == pred_entity) and (pred_relation in p2as):
		#if (gold_e == pred_entity) and (pred_relation in p2as):
			num += 1
			hit_es = p2as[pred_relation]
			# if r2 == "":
				# hit_es_score = {}
				# rs_hit = []
				# for e in hit_es:
					# hit_es_score[e] = 0
				# for r in e2r2t[gold_e]:
					# p_r = max_relation_distribution[target_w2i[r]] if r in target_w2i else 0
					# for t in e2r2t[gold_e][r]:
						# if r not in rs_hit:
							# rs_hit.append(r)
						# if t in hit_es_score:
							# hit_es_score[t] += p_r/len(e2r2t[gold_e][r])
				# if sorted(hit_es, key=lambda e:hit_es_score[e], reverse=True) != hit_es:
					# print ""
					# print ""
					# print question
					# print max_relation
					# for r in rs_hit:
						# print r,e2r2t[gold_e][r]
					# print hit_es
					# print hit_es_score
					# print sorted(hit_es, key=lambda e:hit_es_score[e], reverse=True)
					# print gold_answer
				# hit_es = sorted(hit_es, key=lambda e:hit_es_score[e], reverse=True)
			# hit_es = hit_es[:10]
			# hits_names = []
			# for e in hit_es:
				# if e in mid2name:
					# hits_names += mid2name[e]
				# else:
					# hits_names += [e]
			# gold_names = []
			# for e in gold_answer:
				# if e in mid2name:
					# gold_names += mid2name[e]
				# else:
					# gold_names += [e]
					
			# precision
			hit_num = 0
			for e in hit_es:
				#names = mid2name[e] if e in mid2name else [e]
				#if any(name in gold_names for name in names):
				if e in gold_answer:
					hit_num += 1
				#if e in gold_answer:
				#	hit_num += 1
					
			precision_this = float(hit_num) / len(hit_es)
			
			# recall
			hit_num = 0
			for e in gold_answer:
				#names = mid2name[e] if e in mid2name else [e]
				#if any(name in hits_names for name in names):
				if e in hit_es:
					hit_num += 1
				#if e in hit_es:
				#	hit_num += 1

			recall_this = float(hit_num) / len(gold_answer)
			
			f1_this = 2*precision_this*recall_this / (precision_this + recall_this) if (precision_this + recall_this) != 0 else 0	
			f1_sum += f1_this
			precision_sum += precision_this
			recall_sum += recall_this
			
		else:
			# f_out_error.write(json.dumps(result)+"\n")
			pass
				
	aver_p = float(precision_sum) / test_data_num
	aver_r = float(recall_sum) / test_data_num
	aver_f1 = float(f1_sum) / test_data_num
	# print "aver of p: " +str(aver_p*100)
	# print "aver of r: " +str(aver_r*100)
	# print "f1 of aver: "+str(2*aver_p*aver_r/(aver_p+aver_r)*100)
	# print "aver of f1: " +str(aver_f1*100)
	# print num
	f_out.close()
	return aver_f1*100
	
	
if __name__ == "__main__":
	ck_num = "checkpoint001"
	print predict_all_with_nel(ck_num,0,600)
	exit(0)
	
	lam_list = [0,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
	for lam in lam_list:
		lam_test = lam
		f1 = predict_all_with_nel(ck_num,27,100)
		print lam_test, f1
	exit(0)
	
	lam_test = 10000
	
	f1s = [predict_all_with_nel(ck_num,0,1500)]
	print 0,f1s[0]

	lam_test = 0.1
	
	for i in range(1,50):
		f1 = predict_all_with_nel(ck_num,i,100)
		print i,f1
		f1s.append(f1)
	f1s = [str(f1) for f1 in f1s]
	print " ".join(f1s)
	