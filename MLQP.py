import numpy as np

import sys
import os
import math

def read_data(filename):
	data_set = []
	file = open(filename)
	for line in file:
		sample = [double(x) for x in line]
		data_set.append(sample)
	return data_set

def forward(u, v, sample, label)

def backpropagation(u, v, bias, learining_rate, train_set):


if __name__ == '__main__':
	train_set = read_data('two_spiral_train.txt')
	test_set = read_data('two_spiral_test.txt')