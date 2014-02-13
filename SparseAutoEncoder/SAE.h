#pragma once
#include <Eigen/Dense>
#include <ctime>
#include <fstream>

#define GD 1
#define MINI_BATCH_SGD 0

using namespace std;
using namespace Eigen;

class SAE
{
public:
	//weights
	MatrixXd theta1;
	MatrixXd theta2;
	//bias
	MatrixXd b1;
	MatrixXd b2;
private:
	int inputSize;
	int hiddenSize;
public:
	SAE(int inputSize,int hiddenSize);
	void train(MatrixXd &trainData,
		double lambda,double alpha,
		double beta,double sparsityParam,int maxIter,
		int trainMode,void *option);
	bool saveModel(char *szFileName);
	bool loadModel(char *szFileName);
private:
	MatrixXd KLDivergence(MatrixXd &rho,double sparsityParam);
	MatrixXd KLGradient(MatrixXd &rho,double sparsityParam);
	MatrixXd randomInitialize(int lIn,int lOut);
	MatrixXd sigmoid(MatrixXd &z);
	MatrixXd sigmoidGradient(MatrixXd &z);
	MatrixXd logMat(MatrixXd &z);
	double computeCost(MatrixXd &x,double lambda,
				   MatrixXd &theta1Grad,MatrixXd &theta2Grad,
				   MatrixXd &b1Grad,MatrixXd &b2Grad,
				   double beta,double sparsityParam);

	void updateParameters(MatrixXd &theta1Grad,
		MatrixXd &theta2Grad,MatrixXd &b1Grad,
		MatrixXd &b2Grad,double alpha);

	void gradientDescent(MatrixXd &trainData,
		   double lambda,double alpha,
		   int maxIter,double beta,double sparsityParam);

	void miniBatchSGD(MatrixXd &trainData,
		   double lambda,double alpha,
		   int maxIter,int batchSize,double beta,double sparsityParam);
};

//constructor
SAE::SAE(int inputSize,int hiddenSize)
{
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;
	// initialize the weights
	theta1 = randomInitialize(inputSize,hiddenSize);
	theta2 = randomInitialize(hiddenSize,inputSize);
	b1 = MatrixXd::Zero(1,hiddenSize);
	b2 = MatrixXd::Zero(1,inputSize);
}

//component wise KL divergence value
MatrixXd SAE::KLDivergence(MatrixXd &rho,double sparsityParam)
{
	double sp = sparsityParam;
	MatrixXd result = rho;
	for(int i = 0;i < rho.rows();i++)
	{
		for(int j = 0; j < rho.cols(); j++)
		{
			result(i,j) = sp * log(sp/rho(i,j))
				+ (1 - sp) * log((1 - sp) / (1 - rho(i,j)));
		}
	}
	return result;
}

//component wise KL divergence gradient value
MatrixXd SAE::KLGradient(MatrixXd &rho,double sparsityParam)
{
	double sp = sparsityParam;
	MatrixXd result = rho;
	for(int i = 0;i < rho.rows();i++)
	{
		for(int j = 0; j < rho.cols(); j++)
		{
			result(i,j) = (1 - sp) / (1 - rho(i,j)) - sp / rho(i,j);
		}
	}
	return result;
}

//random initialize the weights
MatrixXd SAE::randomInitialize(int lIn,int lOut)
{
	//random initialize the weight
	int i,j;
	double epsilon = sqrt(6.0) / sqrt((this->inputSize + this->hiddenSize + 1));
	MatrixXd result(lOut,lIn);
	srand((unsigned int)time(NULL));
	for(i = 0;i < lOut;i++)
	{
		for(j = 0;j < lIn;j++)
		{
			result(i,j) = ((double)rand() / (double)RAND_MAX) * 2 * epsilon - epsilon;
		}
	}
	return result;
}

//component wise sigmoid function
MatrixXd SAE::sigmoid(MatrixXd &z)
{
	//return  1.0 ./ (1.0 + exp(-z));
	MatrixXd result(z.rows(),z.cols());
	for(int i = 0;i < z.rows();i++)
	{
		for(int j = 0;j < z.cols();j++)
		{
			result(i,j) = 1.0 / (1 + exp(-z(i,j)));
		}
	}
	return result;
}

//compute the gradient of sigmoid function
MatrixXd SAE::sigmoidGradient(MatrixXd &z)
{
	//return sigmoid(z) .* (1 - sigmoid(z))
	MatrixXd result;
	MatrixXd sigm = sigmoid(z);
	MatrixXd item = MatrixXd::Ones(z.rows(),z.cols()) - sigm;
	result = sigm.cwiseProduct(item);
	return result;
}

//component wise log function
MatrixXd SAE::logMat(MatrixXd &z)
{
	MatrixXd result(z.rows(),z.cols());
	for(int i = 0;i < z.rows();i++)
	{
		for(int j = 0;j < z.cols();j++)
		{
			result(i,j) = log(z(i,j));
		}
	}
	return result;
}

// simple gradient descent update rule
void SAE::updateParameters(MatrixXd &theta1Grad,
			MatrixXd &theta2Grad,MatrixXd &b1Grad,
			MatrixXd &b2Grad,double alpha)
{
	theta1 -= theta1Grad*alpha;
	theta2 -= theta2Grad*alpha;
	b1Grad -= b1Grad * alpha;
	b2Grad -= b2Grad * alpha;
}

//cost function
double SAE::computeCost(MatrixXd &x,double lambda,
				   MatrixXd &theta1Grad,MatrixXd &theta2Grad,
				   MatrixXd &b1Grad,MatrixXd &b2Grad,
				   double beta,double sparsityParam)
{
	double sp = sparsityParam;
	double J = 0;
	//forward
	int xRows = x.rows();
	int xCols = x.cols();
	int numOfExamples = xRows;
	MatrixXd a1 = x;
	MatrixXd z2 = a1*theta1.transpose() + b1.replicate(numOfExamples,1);
	MatrixXd a2 = sigmoid(z2);
	MatrixXd z3 = a2*theta2.transpose() + b2.replicate(numOfExamples,1);
	MatrixXd a3 = sigmoid(z3);

	//average activation of hidden neurons
	MatrixXd rho = a2.colwise().sum()/a2.rows();

	//regularziation term of the cost function
	double JReg = lambda/(2*numOfExamples)*(theta1.array().square().sum()
		+ theta2.array().square().sum());
	
	//cost function 
	J = (1.0/numOfExamples * 0.5) * (a3 - x).array().square().sum() + JReg 
		+ beta * KLDivergence(rho,sp).array().sum(); 
	
	//compute delta
	MatrixXd delta3 = (a3 - x).cwiseProduct(sigmoidGradient(z3));
	MatrixXd klg = KLGradient(rho,sp) * beta;
	MatrixXd delta2 = (delta3*theta2 + klg.replicate(delta3.rows(),1))
		.cwiseProduct(sigmoidGradient(z2));
	
	//compute gradients
	theta1Grad = delta2.transpose() * a1 * (1.0/numOfExamples);
	theta2Grad = delta3.transpose() * a2 * (1.0/numOfExamples);
	b1Grad = delta2.colwise().sum() * (1.0 / numOfExamples);
	b2Grad = delta3.colwise().sum() * (1.0 / numOfExamples);
	
	//Add regularization
	theta1Grad += theta1 * lambda;
	theta2Grad += theta2 * lambda;
	return J;
}


//gradient descent method
void SAE::gradientDescent(MatrixXd &trainData,
		   double lambda,double alpha,
		   int maxIter,double beta,double sparsityParam)
{
	double J = 0;
	double sp = sparsityParam;
	//get the binary code of labels
	MatrixXd theta1Grad(theta1.rows(),theta1.cols());
	MatrixXd theta2Grad(theta2.rows(),theta2.cols());
	MatrixXd b1Grad(b1.rows(),b1.cols());
	MatrixXd b2Grad(b2.rows(),b2.cols());
	int iter = 1;
	//gradient decent
 	for(int i = 0; i < maxIter;i++)
	{
		// compute the cost
		J = computeCost(trainData,lambda,
			theta1Grad,theta2Grad,b1Grad,b2Grad,beta,sp);
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
		updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
	}
}

//mini batch stochastic gradient descent
void SAE::miniBatchSGD(MatrixXd &trainData,
		   double lambda,double alpha,int maxIter,
		   int batchSize,double beta,double sparsityParam)
{
	double sp = sparsityParam;
	MatrixXd theta1Grad(theta1.rows(),theta1.cols());
	MatrixXd theta2Grad(theta2.rows(),theta2.cols());
	MatrixXd b1Grad(b1.rows(),b1.cols());
	MatrixXd b2Grad(b2.rows(),b2.cols());
	MatrixXd miniTrainData(batchSize,trainData.cols());
	int numBatches = trainData.rows() / batchSize;
	int iter = 1;
	//mini batch stochastic gradient decent
 	for(int i = 0; i < maxIter;i++)
	{
		double J = 0;
		// compute the cost
		for(int j = 0;j < numBatches; j++)
		{
			miniTrainData = trainData.middleRows(j * batchSize,batchSize);
			J += computeCost(miniTrainData,lambda,
				theta1Grad,theta2Grad,b1Grad,b2Grad,beta,sp);
#ifdef _IOSTREAM_
			if(miniTrainData.rows() < 1)
			{
				cout << "Too few training examples!"  << endl; 
			}
#endif
			updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
		}
		J = J / numBatches;
#ifdef _IOSTREAM_
		cout << "iter: " << iter++ << "  cost: " << J << endl;
#endif
 	}
}

//train the model
void SAE::train(
		   MatrixXd &trainData,double lambda,
		   double alpha,double beta,double sparsityParam,int maxIter,
		   int trainMode,void *option)
{
	double sp = sparsityParam;
	if(trainData.cols() != this->inputSize)
	{
#ifdef _IOSTREAM_
		cout << "dimension mismatch!" << endl;
#endif
		return;
	}
	if(trainMode == GD)
	{
		gradientDescent(trainData,lambda,alpha,maxIter,beta,sp);
	}
	else if(trainMode == MINI_BATCH_SGD)
	{
		miniBatchSGD(trainData,lambda,alpha,maxIter,
			*((int*)option),beta,sp);
	}
}


//save model to file
bool SAE::saveModel(char *szFileName)
{
	ofstream ofs(szFileName);
	if(!ofs)
	{
		return false;
	}
	int i,j;
	//save the size
	ofs << inputSize << " " << hiddenSize << endl;
	//save theta1
	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			ofs << theta1(i,j) << " ";
		}
	}
	ofs << endl;
	//save theta2
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			ofs << theta2(i,j) << " ";
		}
	}
	ofs << endl;
	//save b1
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			ofs << b1(i,j) << " ";
		}
	}
	ofs << endl;
	//save b2
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			ofs << b2(i,j) << " ";
		}
	}
	ofs.close();
	return true;
}

//load model from file
bool SAE::loadModel(char *szFileName)
{
	ifstream ifs(szFileName);
	if(!ifs || ifs.eof())
	{
		return false;
	}
	ifs >> this -> inputSize >> this -> hiddenSize;
	int i,j;
	//resize the parameters
	theta1.resize(this->hiddenSize,this->inputSize);
	theta2.resize(this->inputSize,this->hiddenSize);
	b1.resize(1,hiddenSize);
	b2.resize(1,inputSize);

	//load theta1
	for(i = 0; i < theta1.rows(); i++)
	{
		for(j = 0;j < theta1.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta1(i,j);
		}
	}
	//load theta2
	for(i = 0; i < theta2.rows(); i++)
	{
		for(j = 0;j < theta2.cols(); j++)
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> theta2(i,j);
		}
	}
	//load b1
	for(i = 0; i < b1.rows(); i++)
	{
		for(j = 0; j < b1.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b1(i,j);
		}
	}
	//load b2
	for(i = 0; i < b2.rows(); i++)
	{
		for(j = 0; j < b2.cols(); j++) 
		{
			if(ifs.eof())
			{
				ifs.close();
				return false;
			}
			ifs >> b2(i,j);
		}
	}
	ifs.close();
	return true;
}