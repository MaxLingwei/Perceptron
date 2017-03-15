import numpy as np

import sys
import os
import math
import random

def sigmod(x):
	return 1 / (1 + math.exp(-x))

def read_data(filename):
	data_set = []
	file = open(filename)
	for line in file:
		sample = [float(x) for x in line.split()]
		data_set.append(sample)
	return data_set

def load_config(filename, u1, v1, bias1, u2, v2):
	config = open(filename, 'r')
	line = config.readline()
	temp_u1 = [float(x) for x in line.split()]
	u1 += temp_u1

	line = config.readline()
	temp_v1 = [float(x) for x in line.split()]
	v1 += temp_v1

	line = config.readline()
	temp_bias1 = [float(x) for x in line.split()]
	bias1 += temp_bias1

	line = config.readline()
	temp_u2 = [float(x) for x in line.split()]
	u2 += temp_u2

	line = config.readline()
	temp_v2 = [float(x) for x in line.split()]
	v2 += temp_v2

	line = config.readline()
	temp_bias2 = [float(x) for x in line.split()]

	config.close()

	return temp_bias2[0]


def forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x):
	for i in range(0, 10):
		temp_in = u1[i] * x[0] * x[0] + v1[i] * x[0] + u1[i + 10] * x[1] * x[1] + v1[i + 10] * x[1] + bias1[i]
		hidden_in.append(temp_in)
		hidden_out.append(sigmod(temp_in))

	out = 0
	for i in range(0, 10):
		out += u2[i] * hidden_out[i] * hidden_out[i] + v2[i] * hidden_out[i]

	out += bias2

	y = sigmod(out)
	return y

if __name__ == '__main__':
	train_set = read_data('two_spiral_train.txt')
	test_set = read_data('two_spiral_test.txt') 
	u1 = []
	v1 = []
	bias1 = []
	u2 = []
	v2 = []
	bias2 = 0
	
	out = []
	final_set = []
	img = [[0 for i in range(1200)] for i in range(1200)]
	out_set = []
	threshold = 0.5

	filename = 'final_para.txt'

	bias2 = load_config(filename, u1, v1, bias1, u2, v2)

	for i in range(0, 400):
		for j in range(0, 400):
			y = 0
			x1 = float(i - 200) / 40.0
			x2 = float(j - 200) / 40.0
			#print 'x1 = ' + str(x1)
			#print 'x2 = ' + str(x2)
			hidden_in = []
			hidden_out = []
			result = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [x1, x2])
			
			if result > threshold:
			    y = 1
			out_set.append([x1, x2, y])
			print [x1, x2, y]

	output = open('img.txt', 'w')

	for i in range(0, len(out_set)):
		output.write(str(out_set[i][0]) + ' ' + str(out_set[i][1]) + ' ' + str(out_set[i][2]) + '\n')
	output.close()
