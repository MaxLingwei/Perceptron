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

def backpropagation(u1, v1, bias1, u2, v2, bias2, learining_rate, x, label, error_set):
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
	error = 0.5 * (y - label) * (y - label)
	error_set.append(error)
	for i in range(0, len(u2)):
		temp_e_u = y * y * (1 - y) * hidden_out[i] * hidden_out[i]
		deri_error_u2.append(temp_e_u)
		u2[i] -= learining_rate * temp_e_u

		temp_e_v = y * y * (1 - y) * hidden_out[i]
		deri_error_v2.append(temp_e_v)
		v2[i] -= learining_rate * temp_e_v

	temp_e_b = y * y * (1 - y)
	bias -= learining_rate * temp_e_b

	for i in range(0, len(u2)):
		temp_e_u1 = y * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0] * x[0]
		temp_e_v1 = y * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0]
		temp_e_b1 = y * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i])

		temp_e_u1_10 = y * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1] * x[1]
		temp_e_v1_10 = y * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1]


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


if __name__ == '__main__':
	train_set = read_data('two_spiral_train.txt')
	test_set = read_data('two_spiral_test.txt') 

	u1 = get_random(20)
	v1 = get_random(20)
	bias1 = get_random(10)
	u2 = get_random(10)
	v2 = get_random(10)
	bias2 = get_random(1)
	bias2 = bias2[0]
	learining_rate = 0.0001
	error_set = []
	error = 0

	for i in range(0, 1000):
		for j in range(0, len(train_set)):
			print "iteration" + str(i)
			print j
			bias2 = backpropagation(u1, v1, bias1, u2, v2, bias2, learining_rate, [train_set[j][0], train_set[j][1]], train_set[j][2], error_set)


	output = open('error_set.txt', 'w')
	for i in range(0, len(error_set)):
		output.write(str(error_set[i]) + '\n')
	output.close()


	print u1
	print v1
	print bias1
	print u2
	print v2
	print bias2
	hidden_in = []
	hidden_out = []

	for i in range(0, len(test_set)):
		y = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [train_set[i][0], train_set[i][1]], train_set[i][2])
		label = train_set[i][2]
		error += 0.5 * (y - label) * (y - label)

	print error