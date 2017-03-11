#include <iostream>
#include <vector>


using namespace std;

typedef struct training_input_tag
{
	vector<double> input_vector;
	int label;
}training_input;



int hardlim(double input)
{
	return input > 0 ? 1 : 0;
}

double inner_product(vector<double> vec1, vector<double> vec2)
{
	double sum = 0.0;
	for (int i = 0; i < vec1.size(); i++)
	{
		sum += vec1[i] * vec2[i];
	}
	return sum;
}

int train_perceptron(vector<double> &weight, double learning_rate, vector<training_input> train_set)
{
	int error_num = 1;
	int iter_num = 0;
	double output;
	double result;
	double error;
	while (error_num > 0)
	{
		for (vector<training_input>::iterator iter = train_set.begin(); iter != train_set.end(); iter++)
		{
			vector<double> tmp_vector(iter->input_vector);
			tmp_vector.push_back(1);
			output = inner_product(weight, tmp_vector);
			result = hardlim(output);
			error = double(iter->label) - result;

			if (error == 0)
				continue;
			else
			{
				for (int j = 0; j < weight.size(); j++)
				{
					weight[j] += learning_rate * error * tmp_vector[j];
				}
				error_num++;
				iter_num++;
			}
		}
	}
	
	return 0;
}

int main(void)
{
	vector<training_input> train_set;
	vector<training_input> origin_set;


}