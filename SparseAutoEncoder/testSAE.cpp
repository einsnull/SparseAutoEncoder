#include <iostream>
#include "SAE.h"
#include <opencv2/opencv.hpp>
#include "cmath"
#include "getConfig.h"

using namespace std;

#pragma comment(lib,"opencv_core243.lib")
#pragma comment(lib,"opencv_highgui243.lib")
#pragma comment(lib,"opencv_imgproc243.lib")

bool loadDataSet(MatrixXd &data);

void toImage(MatrixXd &theta,int imgWidth,
			 char *szFileName,bool showFlag = false,double ratio = 1);

int main()
{
	//regularization coefficient
	double lambda = 0.0001;
	//learning rate
	double alpha = 1;
	double beta = 3;
	//sparsity parameter
	double sp = 0.01;
	//maximum iteration
	int maxIter = 5000;
	//mini batch size of mini batch stochastic gradient descent
	int miniBatchSize = 1000;
	int n = 2;
	int inputLayerSize = 64;
	int hiddenLayerSize = 25;
	int imgWidth = 8; 
	char *fileBuf  = new char[4096];
	//load the config file
	bool ret = loadFileToBuf("ParamConfig.ini",fileBuf,4096);
	if(ret)
	{
		getConfigDoubleValue(fileBuf,"lambda:",lambda);
		getConfigDoubleValue(fileBuf,"alpha:",alpha);
		getConfigDoubleValue(fileBuf,"beta:",beta);
		getConfigDoubleValue(fileBuf,"sparsityParam:",sp);
		getConfigIntValue(fileBuf,"maxIter:",maxIter);
		getConfigIntValue(fileBuf,"miniBatchSize:",miniBatchSize);
		getConfigIntValue(fileBuf,"hiddenSize:",hiddenLayerSize);
		getConfigIntValue(fileBuf,"inputSize:",inputLayerSize);
		getConfigIntValue(fileBuf,"imgWidth:",imgWidth);
		cout << "lambda:" << lambda << endl;
		cout << "alpha:" << alpha << endl;
		cout << "beta:" << beta << endl;
		cout << "sparseParam:" << sp << endl;
		cout << "maxIter:" << maxIter << endl;
		cout << "miniBatchSize:" << miniBatchSize << endl;
		cout << "hiddenLayerSize:" << hiddenLayerSize << endl;
		cout << "inputLayerSize:" << inputLayerSize << endl;
		cout << "imgWidth:" << imgWidth << endl;
	}
	delete []fileBuf;

	MatrixXd trainingData(n,inputLayerSize);
	//n is a temp variable to initialize the matrix object
	//the number of training set examples will be reset to
	//its real size in loadDataSet
	SAE sae(inputLayerSize,hiddenLayerSize);
	if(!loadDataSet(trainingData))
	{
		cout << "Couldn't load training data set." << endl;
		return -1;
	}
	//save the top 100 examples to image
	MatrixXd showImage = trainingData.topRows(100);
	toImage(showImage,imgWidth,"data.bmp");
	clock_t start = clock();
	//train the sparse auto encoder
	sae.train(trainingData,lambda,alpha,beta,sp,
		maxIter,MINI_BATCH_SGD,&miniBatchSize);
	//save the trained weights to image
	toImage(sae.theta1,imgWidth,"weights.bmp");
	clock_t end = clock();
	cout << "The code ran for " << (end - start)/(double)(CLOCKS_PER_SEC*60)
		<< " minutes." << endl;
	sae.saveModel("SAE_Model.txt");
	cout << "lambda:" << lambda << endl;
	cout << "alpha:" << alpha << endl;
	cout << "beta:" << beta << endl;
	cout << "sparsityParam:" << sp << endl;
	cout << "miniBatchSize:" << miniBatchSize << endl;
	system("pause");
	return 0;
}

bool loadDataSet(MatrixXd &data)
{
	ifstream ifs("mydata.txt");
	if(!ifs)
	{
		return false;
	}
	cout << "Loading data..." << endl;
	int inputLayerSize;
	ifs >> inputLayerSize;
	int dataSetSize;
	ifs >> dataSetSize;
	data.resize(dataSetSize,inputLayerSize);
	for(int i = 0; i < dataSetSize; i++)
	{
		for(int j = 0; j < inputLayerSize; j++)
		{
			ifs >> data(i,j);
		}
	}
	//data = data - MatrixXd::Ones(data.rows(),data.cols())*0.5;
	ifs.close();
	return true;
}

void toImage(MatrixXd &theta,int imgWidth,char* szFileName,bool showFlag,double ratio)
{
	int margin = 1;
	int rows = theta.rows();
	int cols = theta.cols();
	if(rows <= 0 || cols <= 0)
	{
		return ;
	}
	/*cout << rows << endl;
	cout << cols << endl;*/
	double pr = sqrt((double)rows);
	int perRow = (int)pr + (pr - (int)pr > 0);
	double tc = (double)rows / (double)perRow;
	int tCols = (int)tc + (tc - (int)tc > 0);
	/*cout << "perRow: " << perRow << endl;
	cout << "tCols: " << tCols << endl;*/
	MatrixXd max = theta.rowwise().maxCoeff();
	MatrixXd min = theta.rowwise().minCoeff();

	int imgHeight = cols/imgWidth;
	IplImage* iplImage = cvCreateImage(
		cvSize(imgWidth * perRow + margin * (perRow+1),imgHeight * tCols + margin * (tCols + 1)),
		IPL_DEPTH_8U,1);
	
	int step = iplImage->widthStep;
	uchar *data = (uchar *)iplImage->imageData;
	int h = iplImage->height;
	int w = iplImage->width;
	for(int x = 0; x < w; x++)
	{
		for(int y = 0; y < h; y++)
		{
			data[y * step + x] = 0;
		}
	}

	for(int i = 0;i < rows;i++)
	{
		
		int n = 0;
		int hIdx = i / perRow;
		int wIdx = i % perRow;

		for(int j = 0;j < imgHeight;j++)
		{
			for(int k = 0;k < imgWidth; k++)
			{
				double per = (theta(i,n) - min(i,0) ) / (max(i,0) - min(i,0));
				//data[j * step + k] = 255;
				int val = (hIdx * imgHeight+j + margin * (hIdx+1)) * step
					+ (wIdx * imgWidth + k + margin * (wIdx+1));
				if(val > step * (imgHeight * tCols + margin * (tCols + 1)))
				{
					cout << "error" << endl;
				}
				data[val] = (uchar)(int)(per * 230.0);
				n ++;
			}
		}
		
	}
	
	cvSaveImage(szFileName,iplImage);

	if(showFlag)
	{
		cvNamedWindow("Image",CV_WINDOW_AUTOSIZE);
		IplImage* iplImageShow = cvCreateImage(
			cvSize((int)(iplImage->width * ratio),(int)(iplImage->height * ratio)),IPL_DEPTH_8U,1);
		cvResize(iplImage,iplImageShow,CV_INTER_CUBIC);
		cvShowImage("Image",iplImageShow);
		cvWaitKey(100000);
		cvDestroyWindow("Image");
		cvReleaseImage(&iplImageShow);
	}
	cvReleaseImage(&iplImage);
}

