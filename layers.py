import math
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
from torch import nn as nn 
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch





class ZERON_GCN(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(ZERON_GCN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight1 = Parameter(torch.Tensor(in_features, out_features))


		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 6. / math.sqrt(self.weight1.size(1) + self.weight1.size(0))
		self.weight1.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-0, 0)

	def forward(self, input, adj, activation):
		support = torch.mm(input, self.weight1)
		# print torch.mm(adj['adj'], support[:, :support.shape[1]//2]).shape, support[:, support.shape[1]//2:].shape
		output = torch.cat((torch.mm(adj['adj'], support[:, :support.shape[1]//10]), support[:, support.shape[1]//10:]), dim = 1)
		# output = (torch.mm(adj['adj'], support))
		# print output.shape, exit()
		if self.bias is not None:
			output = output + self.bias
		return activation(output)

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GCNMax(Module):
	#### https://arxiv.org/pdf/1509.09292.pdf ####
	def __init__(self, in_features, print_length):
		super(GCNMax, self).__init__()
		self.in_features = in_features
		self.print_length = print_length
		self.weight_Ws = nn.ParameterList(Parameter(torch.Tensor(in_features, print_length)) for i in range(1))
		self.weight_Bs = nn.ParameterList(Parameter(torch.Tensor(print_length)) for i in range(1))
		self.reset_parameters()

	def reset_parameters(self):
		for i in range(1):
			stdv = 6. / math.sqrt(self.weight_Bs[i].size(0))

			self.weight_Bs[i].data.uniform_(-stdv, stdv)
			stdv = 6. / math.sqrt(self.weight_Ws[i].size(0) + self.weight_Ws[i].size(1))
			self.weight_Ws[i].data.uniform_(-stdv, stdv)
			

	def forward(self, r_s, adj, activation):

		
		bias = self.weight_Bs[0]
		weight_W = self.weight_Ws[0]
		


		v_s = torch.mm(r_s, weight_W)  ## 10
		v_s = torch.cat((torch.mm(adj['adj'], v_s[:, :v_s.shape[1]//10]), v_s[:, v_s.shape[1]//10:]), dim = 1)


		v_s = v_s + bias
		

		i_s = activation(v_s)       ## 10
	
		f   = torch.max(v_s, dim = 0)[0]       ## 11
		# f   = torch.sum(v_s, dim = 0)

		
		return f                            ## 12

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.print_length) + ' -> ' \
			   + str(self.print_length) + ')'




class Image_ZERON_GCNGCN(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(Image_ZERON_GCNGCN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight1 = Parameter(torch.Tensor(in_features, out_features))


		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 6. / math.sqrt((self.weight1.size(1) + self.weight1.size(0)))
		stdv*= .6


		# stdv = math.sqrt(6. / (self.weight1.size(1) + self.weight1.size(0)))
		# stdv*= .2
		self.weight1.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-.1, .1)

	def forward(self, input, adj, activation):
		
		support = torch.mm(input, self.weight1)
		
		output = torch.cat((torch.mm(adj['adj'], support[:, :support.shape[1]//3]), support[:, support.shape[1]//3:]), dim = 1)
		
		if self.bias is not None:
			output = output + self.bias
		return activation(output)

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'
