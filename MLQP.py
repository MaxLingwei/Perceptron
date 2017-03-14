import numpy as np

import sys
import os
import math
import random

def read_data(filename):
	data_set = []
	file = open(filename)
	for line in file:
		sample = [float(x) for x in line.split()]
		data_set.append(sample)
	return data_set

def sigmod(x):
	return 1 / (1 + math.exp(-x))

def forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x, label):
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

def backpropagation(u1, v1, bias1, u2, v2, bias2, learining_rate, x, label):
	deri_error_u2 = []
	deri_error_v2 = []
	deri_error_bias2 = []
	deri_error_u1 = []
	deri_error_v1 = []
	deri_error_bias = 0

	hidden_in = []
	hidden_out = []
	bias = bias2

	y = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x, label)

	for i in range(0, len(u2)):
		temp_e_u = (y - label) * y * (1 - y) * hidden_out[i] * hidden_out[i]
		deri_error_u2.append(temp_e_u)
		u2[i] -= learining_rate * temp_e_u

		temp_e_v = (y - label) * y * (1 - y) * hidden_out[i]
		deri_error_v2.append(temp_e_v)
		v2[i] -= learining_rate * temp_e_v

	temp_e_b = (y - label) * y * (1 - y)
	bias -= learining_rate * temp_e_b

	for i in range(0, len(u2)):
		temp_e_u1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0] * x[0]
		temp_e_v1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0]
		temp_e_b1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i])

		temp_e_u1_10 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1] * x[1]
		temp_e_v1_10 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1]


		u1[i] -= learining_rate * temp_e_u1
		v1[i] -= learining_rate * temp_e_v1

		u1[i + 10] -= learining_rate * temp_e_u1_10
		v1[i + 10] -= learining_rate * temp_e_v1_10
		bias1[i] -= learining_rate * temp_e_b

	return bias


def get_random(num):
	result = []
	for i in range(0, num):
		random.seed()
		random_num = random.random() * 2 - 1
		result.append(random_num)

	return result

def load_config(filename, u1, v1, bias1, u2, v2):
	config = open('config.txt', 'r')
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

def write_config(filename, u1, v1, bias1, u2, v2, bias2):
	config = open(filename, 'w')

	for i in range(0, len(u1)):
		config.write(str(u1[i]) + ' ')
	config.write('\n')

	for i in range(0, len(v1)):
		config.write(str(v1[i]) + ' ')
	config.write('\n')

	for i in range(0, len(bias1)):
		config.write(str(bias1[i]) + ' ')
	config.write('\n')

	for i in range(0, len(u2)):
		config.write(str(u2[i]) + ' ')
	config.write('\n')

	for i in range(0, len(v2)):
		config.write(str(v2[i]) + ' ')
	config.write('\n')
	config.write(str(bias2))
	config.close()



def write_file(filename, data):
	output = open(filename, 'w')
	for i in range(0, len(data)):
		output.write(str(data[i]) + '\n')
	output.close()

if __name__ == '__main__':
	train_set = read_data('two_spiral_train.txt')
	test_set = read_data('two_spiral_test.txt') 

	u1 = []
	v1 = []
	bias1 = []
	u2 = []
	v2 = []
	bias2 = 0

	learining_rate = 0.1
	filename = 'config.txt'
	train_error_set = []
	train_error_step = []
	test_error_set = []
	test_error_step = []
	test_result = []
	error = 0

	bias2 = load_config(filename, u1, v1, bias1, u2, v2)

	for i in range(0, 5000):
		print "iteration" + str(i)
		for j in range(0, len(train_set)):
			bias2 = backpropagation(u1, v1, bias1, u2, v2, bias2, learining_rate, [train_set[j][0], train_set[j][1]], train_set[j][2])

		error = 0
		for k in range(0, len(train_set)):
			hidden_in = []
			hidden_out = []
			y = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [train_set[k][0], train_set[k][1]], train_set[k][2])
			label = train_set[k][2]
			temp_error = 0.5 * (y - label) * (y - label)
			train_error_step.append(temp_error)
			error += temp_error
		error = error / len(train_set)
		train_error_set.append(error)

		error = 0
		for k in range(0, len(test_set)):
			hidden_in = []
			hidden_out = []
			y = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [test_set[k][0], test_set[k][1]], test_set[k][2])
			label = test_set[k][2]
			temp_error = 0.5 * (y - label) * (y - label)
			test_error_step.append(temp_error)
			error += temp_error
		error = error / len(test_set)
		test_error_set.append(error)


	write_file('train_error_set.txt', train_error_set)
	write_file('test_error_set.txt', test_error_set)
	write_file('train_error_step.txt', train_error_step)
	write_file('test_error_step.txt', test_error_step)
	
	print u1
	print v1
	print bias1
	print u2
	print v2
	print bias2
	
	write_config('final_para.txt', u1, v1, bias1, u2, v2, bias2)