#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小
	int mapNum;       //特征模板的数目
	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int DataSize;
	float* mapData;     //存放特征模块的数据
	float* basicData;   //偏置
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
	float* d; // 网络的局部梯度,δ值  
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	float* basicData;   //偏置
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	float* y; // 采样函数后神经元的输出,无激活函数
	float* d; // 网络的局部梯度,δ值
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct nn_layer{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	float* wData; // 权重数据，为一个inputNum*outputNum大小
	float* basicData;   //偏置，大小为outputNum大小

	// 下面三者的大小同输出的维度相同
	float* v; // 进入激活函数的输入值
	float* y; // 激活函数后神经元的输出
	float* d; // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // 训练误差
}CNN;

typedef struct ImgSize{
	int w;
	int h;
}nSize;

typedef struct train_opts{
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
*/
void cnntrain(CNN* cnn,nSize inputSize,int outputSize,
	float* inputData,float* outputData,int dataNum,CNNOpts opts);

// 初始化卷积层
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int mapNum,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// 初始化采样层
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// 初始化输出层
OutLayer* initOutLayer(int inputNum,int outputNum);

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float* activation_Sigma(float* input,int inputNum,float bas); // sigma激活函数

void cnnff(CNN* cnn,float* inputData); // 网络的前向传播
void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
void cnnapplygrads(CNN* cnn,CNNOpts opts);

float* cov(float* map,nSize mapSize,float* inputData,nSize inSize); // 卷积操作

/*
	Pooling Function
	input 输入数据
	inputNum 输入数据数目
	mapSize 求平均的模块区域
*/
float* avgPooling(float* input,nSize inputSize,int mapSize); // 求平均值

/* 
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
float* nnff(float* input,float* wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

#endif
