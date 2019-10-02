import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil
from parameters import *

def getPaddingForSAMEConv(L_in, kernel_size, stride, dilation):
	#https://pytorch.org/docs/stable/nn.html
	if L_in % stride==0:
		doublepadding = max(dilation*(kernel_size - 1) + 1 - stride, 0)
	else:
		doublepadding = max(dilation*(kernel_size - 1) + 1 - (L_in % stride), 0)
	padding=int(doublepadding/2)
	if doublepadding % 2==0:
		return(padding)
	else:
		print('Proper SAME padding is not possible with these L_in, kernelsize, stride, dilation parameters.')
		exit(-1)

def getLoutForSAMEConv(L_in, stride):
	#https://pytorch.org/docs/stable/nn.html
	L_out=ceil(float(L_in) / float(stride))
	return(L_out)

def getLoutForVALIDConv(L_in, padding, kernel_size, stride, dilation):
	#https://pytorch.org/docs/stable/nn.html
	L_out=floor((float(L_in) + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride)+1)
	return(L_out)

# CNN attributes
# image
image_width = FORAGER_VISRESOL # the width of the input layer (visual information) 
image_channels = 2 # number of features (one value for each site)
# convolutional layers
C_outs = [32, 64, 64] # number of extracted features in subsequent convolutional layers
C_ins = [image_channels]+C_outs[0:-1] # number of input channels (features)
dilation=1
conv_kernel_sizes = [3, 3, 3] # convolutional kernel widths in subsequent convolutional layers
conv_strides = [1, 1, 1] # convolution strides in subsequent convolutional layers
L_ins=[]
L_outs=[]
conv_paddings=[]
for i in range(len(conv_kernel_sizes)):
	if i==0:
		L_ins.append(image_width)
	else:
		L_ins.append(L_outs[i-1])
	conv_paddings.append(getPaddingForSAMEConv(L_ins[i], conv_kernel_sizes[i], conv_strides[i], dilation))
	L_outs.append(getLoutForSAMEConv(L_ins[i], conv_strides[i]))
# fully connected layers
n_hidden_ins = L_outs[-1]*L_outs[-1]*C_outs[-1] # number of inputs in the first fully connected layer
n_hidden = 512 # number of cells in the first fully connected layer
n_outputs=DIR_RESOL # number of cells in the second fully connected layer (outputs)

class q_network(nn.Module):
	def __init__(self):
		super(q_network, self).__init__()
		self.c_layers = []
		self.c_batchnorms = []
		self.conv1_layer = nn.Conv2d(in_channels = C_ins[0], out_channels = C_outs[0], kernel_size=conv_kernel_sizes[0], stride=conv_strides[0], padding=conv_paddings[0])
		#self.bn1 = nn.BatchNorm2d(C_outs[0])
		self.conv2_layer = nn.Conv2d(in_channels = C_ins[1], out_channels = C_outs[1], kernel_size=conv_kernel_sizes[1], stride=conv_strides[1], padding=conv_paddings[1])
		#self.bn2 = nn.BatchNorm2d(C_outs[1])
		self.conv3_layer = nn.Conv2d(in_channels = C_ins[2], out_channels = C_outs[2], kernel_size=conv_kernel_sizes[2], stride=conv_strides[2], padding=conv_paddings[2])
		#self.bn3 = nn.BatchNorm2d(C_outs[2])
		self.hidden = nn.Linear(n_hidden_ins, n_hidden)
		self.Q_outputs = nn.Linear(n_hidden, n_outputs)
		
	def forward(self, x):
		x = F.relu(self.conv1_layer(x))
		x = F.relu(self.conv2_layer(x))
		x = F.relu(self.conv3_layer(x))
		#x = x.view(-1, self.num_flat_features(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.hidden(x))
		x = F.relu(self.Q_outputs(x))
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
