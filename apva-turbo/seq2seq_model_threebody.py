import torch
import numpy as np
from torch.autograd import Variable 
import torch.nn as nn	
import torch.nn.functional as F	

USE_CUDA = True


class WFPFEncoderRNN(nn.Module):
	def __init__(self, input_size_wf, dim_wf, input_size_pf, dim_pf):
		super(WFPFEncoderRNN, self).__init__()
		self.input_size_wf = input_size_wf
		self.dim_wf = dim_wf
		self.input_size_pf = input_size_pf
		self.dim_pf = dim_pf
		self.num_layers = 1
		self.dropout = 0.1
		
		self.embedding_wf = nn.Embedding(self.input_size_wf, self.dim_wf)
		self.embedding_pf = nn.Embedding(self.input_size_pf, self.dim_pf)
		self.gru = nn.GRU(self.dim_wf + self.dim_pf, self.dim_wf + self.dim_pf, self.num_layers, dropout=self.dropout)
		
	def forward(self, input_batch, input_lengths, pf_batch, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# input_batch: S*B
		wf_vecs = self.embedding_wf(input_batch) # S*B*Dw
		pf_vecs = self.embedding_pf(pf_batch) # S*B*Dp
		seq_vecs = torch.cat((wf_vecs, pf_vecs),2)

		#sorted by seq len to pack
		gru_input = seq_vecs
		input_lengths_sorted = sorted(input_lengths, reverse=True)
		sort_index = np.argsort(-np.array(input_lengths)).tolist()
		gru_input_sorted = Variable(torch.zeros(gru_input.size())).cuda()
		batch_size = input_batch.size()[1]
		for b in range(batch_size):
			gru_input_sorted[:,b,:] = gru_input[:,sort_index[b],:]
		packed = torch.nn.utils.rnn.pack_padded_sequence(gru_input_sorted, input_lengths_sorted)
		outputs, hidden = self.gru(packed, hidden)		
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
		#resort output
		outputs_resorted = Variable(torch.zeros(outputs.size())).cuda()
		hidden_resorted = Variable(torch.zeros(hidden.size())).cuda()
		for b in range(batch_size):
			outputs_resorted[:,sort_index[b],:] = outputs[:,b,:]
			hidden_resorted[:,sort_index[b],:] = hidden[:,b,:]
		#outputs: S*B*(Dw+Dp)
		#hidden: NUM_LAYER*B*(Dw+Dp)
		return outputs_resorted, hidden_resorted
		
		
class KFEncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(KFEncoderRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = 1
		self.dropout = 0.1
		
		self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout)
		
	def forward(self, input_batch, input_lengths, hidden):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# input_batch: S*B*D
		
		#sorted by seq len to pack
		gru_input = input_batch
		input_lengths_sorted = sorted(input_lengths, reverse=True)
		sort_index = np.argsort(-np.array(input_lengths)).tolist()
		batch_size = input_batch.size()[1]
		gru_input_sorted = Variable(torch.zeros(gru_input.size())).cuda()
		hidden_sorted = Variable(torch.zeros(hidden.size())).cuda()
		for b in range(batch_size):
			gru_input_sorted[:,b,:] = gru_input[:,sort_index[b],:]
			hidden_sorted[:,b,:] = hidden[:,sort_index[b],:]	
		packed = torch.nn.utils.rnn.pack_padded_sequence(gru_input_sorted, input_lengths_sorted)
		outputs, hidden = self.gru(packed, hidden_sorted)		
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
		#resort output
		outputs_resorted = Variable(torch.zeros(outputs.size())).cuda()
		hidden_resorted = Variable(torch.zeros(hidden.size())).cuda()
		for b in range(batch_size):
			outputs_resorted[:,sort_index[b],:] = outputs[:,b,:]
			hidden_resorted[:,sort_index[b],:] = hidden[:,b,:]
		#outputs: S*B*Dk
		#hidden: NUM_LAYER*B*Dk
		return outputs_resorted, hidden_resorted
		
		
class WFPFKFEncoderRNN(nn.Module):
	def __init__(self, source_vocab_size, dim_wf, max_p_distance, dim_pf, dim_kf, dim_mlp, output_size):
		super(WFPFKFEncoderRNN, self).__init__()
		self.input_size_wf = source_vocab_size
		self.dim_wf = dim_wf
		self.input_size_pf = max_p_distance
		self.dim_pf = dim_pf
		self.dim_kf = dim_kf
		self.dim_mlp = dim_mlp
		self.output_size = output_size
		
		self.WFPFEncoder = WFPFEncoderRNN(self.input_size_wf, self.dim_wf, self.input_size_pf, self.dim_pf)
		self.KFEncoder = KFEncoderRNN(self.dim_kf, self.dim_kf)
		
		#self.matrix1 = nn.Linear(self.dim_kf, self.dim_mlp)
		#self.matrix2 = nn.Linear(self.dim_mlp, self.dim_mlp)
		
		self.matrix3 = nn.Linear(self.dim_wf + self.dim_pf + self.dim_mlp, self.output_size)
		
	def forward(self, input_seqs, input_seq_lengths, p_seqs, rvec_seqs, rvec_lengths, tvecs):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# input_seqs: S*B
		# p_seqs: S*B
		# rvec_seqs: S*B*(Dk+1)
		# tvecs: B*Dk
		wfpf_encode_output, wfpf_encode_hidden = self.WFPFEncoder(input_seqs, input_seq_lengths, p_seqs, hidden=None)
		tvecs = tvecs.unsqueeze(0)
		kf_encode_output, kf_encode_hidden = self.KFEncoder(rvec_seqs, rvec_lengths, tvecs)
		#encode_output: S*B*D
		#encode_hidden: NUM_LAYER*B*D
		wfpf_encode_hidden = wfpf_encode_hidden[0]
		kf_encode_hidden = kf_encode_hidden[0]
		#encode_hidden: B*D
		kf_encode_hidden_plus = kf_encode_hidden
		#kf_encode_hidden_plus = F.tanh(self.matrix1(kf_encode_hidden)) 
		#kf_encode_hidden_plus = F.tanh(self.matrix2(kf_encode_hidden_plus)) 
		#encode_hidden_plus: B*Dm
		con_hidden = torch.cat((wfpf_encode_hidden, kf_encode_hidden_plus),1).unsqueeze(0)
		#con_hidden: 1 * B * (Dw + Dp + Dm)
		#output = F.tanh(self.matrix3(con_hidden))
		#output: 1 * B * encoder_output_size
		return con_hidden
			
			
class DecoderRNN(nn.Module):
	def __init__(self, target_vocab_size, input_size, hidden_size):
		super(DecoderRNN, self).__init__()
		# Define parameters
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = target_vocab_size
		self.num_layers = 1
		self.dropout = 0.1
		# Define layers
		self.embedding = nn.Embedding(self.output_size, self.input_size)
		self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout)
		self.out = nn.Linear(self.hidden_size, self.output_size)
	
	def forward(self, word_input, prev_hidden):
		# Note: we run this one step at a time
		# TODO: FIX BATCHING
		# Get the embedding of the current input word (last output word)
		# word input: B
		# prev_hidden: 1*B*D
		batch_size = word_input.size(0)
		embedded = self.embedding(word_input) # B*D
		embedded = embedded.view(1, batch_size, self.input_size) # 1*B*D
		
		# Get current hidden state from input word and last hidden state
		rnn_output, hidden = self.gru(embedded, prev_hidden)
		# rnn_output : 1*B*D
		# hidden : 1*B*D
		rnn_output = rnn_output.squeeze(0)
		output = self.out(rnn_output)
		
		return output, hidden


class MLPClassifier(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLPClassifier, self).__init__()
		# Define parameters
		self.input_size = input_size
		self.output_size = output_size
		self.hidden1 = 200
		self.hidden2 = 100
		# Define layers
		self.matrix1 = nn.Linear(self.input_size, self.hidden1)
		self.matrix2 = nn.Linear(self.hidden1, self.hidden2)
		self.out = nn.Linear(self.hidden2, self.output_size)
	
	def forward(self, input_batch, r_batch):
		# Note: we run this one step at a time
		# TODO: FIX BATCHING
		din = torch.cat((input_batch.squeeze(0), r_batch), 1)
		dout = F.relu(self.matrix1(din))
		dout = F.relu(self.matrix2(dout))
		dout = self.out(dout)
		return dout
		
		
def masked_cross_entropy(logits, target, lengths):
	"""
	Args:
		logits: A Variable containing a FloatTensor of size
			(batch, max_len, num_classes) which contains the
			unnormalized probability for each class.
		target: A Variable containing a LongTensor of size
			(batch, max_len) which contains the index of the true
			class for each corresponding step.
		length: A Variable containing a LongTensor of size (batch,)
			which contains the length of each data in a batch.

	Returns:
		loss: An average loss value masked by the length.
	"""
	lengths = Variable(torch.LongTensor(lengths))
	if USE_CUDA:
		lengths = lengths.cuda()
	logits_flat = logits.view(-1, logits.size(-1)) # BS * target_vocab_size
	log_probs_flat = F.log_softmax(logits_flat, dim=1) # BS * target_vocab_size
	target_flat = target.view(-1, 1) # BS * 1
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # BS * 1
	losses = losses_flat.view(*target.size()) # B * S
	mask = sequence_mask(sequence_lengths=lengths, max_len=target.size(1)) # B * S
	losses = losses * mask.float() # B * S
	loss = losses.sum() / lengths.float().sum()
	return loss

	
def sequence_mask(sequence_lengths, max_len=None):
	if max_len is None:
		max_len = sequence_lengths.data.max()
	batch_size = sequence_lengths.size(0)
	seq_range = torch.arange(0, max_len).long() # S
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) # B*S
	seq_range_expand = Variable(seq_range_expand)
	if sequence_lengths.is_cuda:
		seq_range_expand = seq_range_expand.cuda()
	seq_length_expand = (sequence_lengths.unsqueeze(1)
						 .expand_as(seq_range_expand))
	return seq_range_expand < seq_length_expand
		