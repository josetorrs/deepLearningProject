import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class MiniBatchDiscrimination(nn.Module):
	def __init__(self, feat_num, out_size, row_size, batch_size):
		super(MiniBatchDiscrimination, self).__init__()
		self.feat_num = feat_num
		self.out_size = out_size
		self.row_size = row_size
		self.N = batch_size
		self.T = Parameter(torch.Tensor(feat_num,
                                        out_size,
                                        row_size))
		self.reset_parameters()

	def forward(self, x):
		# Output matrices after matrix multiplication
		M = x.mm(self.T.view(self.feat_num,
                             self.out_size*self.row_size)).view(-1, self.out_size, self.row_size)
		out = Variable(torch.zeros(self.N, self.out_size))

        # Not happy about this 'for' loop, but this is the best we could do using PyTorch IMO
		for k in range(self.N):
			c = torch.exp(-torch.sum(torch.abs(M[k,:]-M),2)) # exp(-L1 Norm of Rows difference)
			if k != 0 and k != self.N -1:
				out[k,:] = torch.sum(c[0:k,:],0) + torch.sum(c[k:-1,:],0)
			else:
				if k == 0:
					out[k,:] = torch.sum(c[1:,:],0)
				else:
					out[k,:] = torch.sum(c[0:self.N-1],0)
		return out

	def reset_parameters(self):
		stddev = 1/self.feat_num
		self.T.data.uniform_(stddev)
