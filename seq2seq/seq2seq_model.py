import torch
from torch.autograd import Variable 
import torch.nn as nn	
import torch.nn.functional as F	
import numpy as np

USE_CUDA = True

class EncoderRNN(nn.Module):
	def __init__(self, config):
		super(EncoderRNN, self).__init__()
		self.input_size = config.source_vocab_size
		self.hidden_size = config.hidden_size
		self.num_layers = 1
		self.dropout = 0.1
		
		self.embedding = nn.Embedding(self.input_size, self.hidden_size)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout, bidirectional=False)
		
	def forward(self, input_seqs, input_lengths, hidden=None):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		# input: S*B
		embedded = self.embedding(input_seqs) # S*B*D
		
		#sorted by seq len to pack
		gru_input = embedded
		input_lengths_sorted = sorted(input_lengths, reverse=True)
		sort_index = np.argsort(-np.array(input_lengths)).tolist()
		gru_input_sorted = Variable(torch.zeros(gru_input.size())).cuda()
		batch_size = gru_input.size()[1]
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
		outputs = outputs_resorted
		hidden = hidden_resorted
		
		'''
		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)	
		'''
		#outputs: S*B*2D
		#hidden: 2*B*D
		#outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		#hidden = hidden[:1, :, :] + hidden[-1:, :, :]
		#outputs: S*B*D
		#hidden: 1*B*D
		return outputs, hidden
			
			
class DecoderRNN(nn.Module):
	def __init__(self, config):
		super(DecoderRNN, self).__init__()
		# Define parameters
		self.hidden_size = config.hidden_size
		self.output_size = config.target_vocab_size
		self.num_layers = 1
		self.dropout_p = 0.1
		# Define layers
		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p)
		self.out = nn.Linear(self.hidden_size, self.output_size)
	
	def forward(self, word_input, prev_hidden):
		# Get the embedding of the current input word (last output word)
		# word input: B
		# prev_hidden: 1*B*D
		batch_size = word_input.size(0)
		embedded = self.embedding(word_input) # B*D
		embedded = self.dropout(embedded)
		embedded = embedded.unsqueeze(0) # 1*B*D
		
		rnn_output, hidden = self.gru(embedded, prev_hidden)
		# rnn_output : 1*B*D
		# hidden : 1*B*D
		rnn_output = rnn_output.squeeze(0) # B*D
		output = self.out(rnn_output) # B*target_vocab_size
		return output, hidden

		
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