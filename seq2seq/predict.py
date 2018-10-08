import torch
import train
from train import Config as config

checkpoint_path = "checkpoint"
iter_num = 8000

train.source_w2i, train.source_i2w = train.load_vocabs(config.source_vocab_path)
train.target_w2i, train.target_i2w = train.load_vocabs(config.target_vocab_path)

config.source_vocab_size = len(train.source_w2i)
config.target_vocab_size = len(train.target_w2i)

source_test = train.read_docs_to_seqs(config.source_test_path, train.source_w2i)
target_test = train.read_docs_to_seqs(config.target_test_path, train.target_w2i)

test_pairs = [(s_seq, t_seq) for s_seq, t_seq in zip(source_test, target_test)]

source_batch, source_lengths, target_batch, target_lengths = train.get_batch(test_pairs, None)

encoder = torch.load(checkpoint_path+"/encoder.model.iter"+str(iter_num)+".pth")
decoder = torch.load(checkpoint_path+"/decoder.model.iter"+str(iter_num)+".pth")

precision, preds = train.run_epoch(source_batch, source_lengths, target_batch, target_lengths, encoder, decoder, TRAIN=False)

print precision

fout = open("predict.result","w")

for i in range(len(preds)):
	fout.write(" ".join([train.source_i2w[j] for j in source_test[i]])+"\n")
	fout.write(" ".join([train.target_i2w[j] for j in target_test[i]])+"\n")
	fout.write(" ".join([train.target_i2w[j] for j in preds[i]])+"\n")
	fout.write(str(preds[i] == target_test[i])+"\n")
	fout.write("\n")

fout.close()