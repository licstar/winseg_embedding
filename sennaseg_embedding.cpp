#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <omp.h>
#ifdef LINUX
#include <sys/time.h>
#else
#include <time.h>
#endif

using namespace std;

const int H = 100; //隐藏层
const int MAX_C = 50; //最大分类数
const int MAX_F = 1000; //输入层最大的大小
const char *model_name = "model_300_nosuff_noinit";

const char *train_file = "large_unsupervised.txt";
const char *dict_file = "dict.txt";

int input_size; //特征数，输入层大小 input_size = window_size*vector_size
int window_size; //窗口大小
int vector_size; //一个词单元的向量大小 = 词向量大小（约50） + 所有特征的大小（约10）

//===================== 所有要优化的参数 =====================
struct embedding_t{
	int size; //里面包含多少个变量（value 里面的变量个数） size = element_size * element_num
	int element_size; //一个向量的长度
	int element_num; //向量的个数
	double *value; //所有的参数

	void init(int element_size, int element_num){
		this->element_size = element_size;
		this->element_num = element_num;
		size = element_size * element_num;
		value = new double[size];
	}
};

embedding_t words; //词向量

double *A; //特征矩阵：[分类数][隐藏层] 第二层的权重
double *B; //特征矩阵：[隐藏层][特征数] 第一层的权重
double *gA, *gB;

//===================== 已知数据 =====================
struct data_t{
	int word; //词的编号
};
//训练集
data_t *data; //训练数据：[样本数][特征数]
int N; //训练集大小

//验证集
data_t *vdata; //测试数据：[样本数][特征数]
int vN; //测试集大小

#include "fileutil.hpp"


double time_start;
double lambda = 0;//0.01; //正则项参数权重
double alpha = 0.01; //学习速率
int iter = 0;

const int thread_num = 4;
const int patch_size = thread_num;

double getTime(){
#ifdef LINUX
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return 0;
#endif
}

double nextDouble(){
	return rand() / (RAND_MAX + 1.0);
}

void softmax(double hoSums[], double result[], int n){
	double max = hoSums[0];
	for (int i = 0; i < n; ++i)
		if (hoSums[i] > max) max = hoSums[i];
	double scale = 0.0;
	for (int i = 0; i < n; ++i)
		scale += exp(hoSums[i] - max);
	for (int i = 0; i < n; ++i)
		result[i] = exp(hoSums[i] - max) / scale;
}

double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

double hardtanh(double x){
	if(x > 1)
		return 1;
	if(x < -1)
		return -1;
	return x;
}

//b = Ax
void fastmult(double *A, double *x, double *b, int xlen, int blen){
	double val1, val2, val3, val4;
	double val5, val6, val7, val8;
	int i;
	for (i=0; i<blen/8*8; i+=8) {
		val1=0;
		val2=0;
		val3=0;
		val4=0;

		val5=0;
		val6=0;
		val7=0;
		val8=0;

		for (int j=0; j<xlen; j++) {
			val1 += x[j] * A[j+(i+0)*xlen];
			val2 += x[j] * A[j+(i+1)*xlen];
			val3 += x[j] * A[j+(i+2)*xlen];
			val4 += x[j] * A[j+(i+3)*xlen];

			val5 += x[j] * A[j+(i+4)*xlen];
			val6 += x[j] * A[j+(i+5)*xlen];
			val7 += x[j] * A[j+(i+6)*xlen];
			val8 += x[j] * A[j+(i+7)*xlen];
		}
		b[i+0] += val1;
		b[i+1] += val2;
		b[i+2] += val3;
		b[i+3] += val4;

		b[i+4] += val5;
		b[i+5] += val6;
		b[i+6] += val7;
		b[i+7] += val8;
	}

	for (; i<blen; i++) {
		for (int j=0; j<xlen; j++) {
			b[i] += x[j] * A[j+i*xlen];
		}
	}
}

double calcNet(data_t *id, double *x, double *h){
	for(int i = 0, j = 0; i < window_size; i++){
		int offset = id[i].word * words.element_size;
		for(int k = 0; k < words.element_size; k++,j++){
			x[j] = words.value[offset + k];
		}
	}

	fastmult(B, x, h, input_size, H);
	for(int i = 0; i < H; i++){
		h[i] = tanh(h[i]);
	}

	double ret = 0;
	for(int j = 0; j < H; j++){
		ret += h[j] * A[j];
	}
	return tanh(ret);
}

void bpNet(data_t *id, double *x, double *h, double dy){
	double dh[H] = {0};
	for(int j = 0; j < H; j++){
		dh[j] = dy * A[j];
		dh[j] *= 1-h[j]*h[j];
	}

	for(int j = 0; j < H; j++){
		A[j] += alpha/sqrt(H) * (dy * h[j] - lambda * A[j]);
	}

	double dx[MAX_F] = {0};

	for(int i = 0; i < H; i++){
		for(int j = 0; j < input_size; j++){
			dx[j] += dh[i] * B[i*input_size+j];
		}
	}

	for(int i = 0; i < H; i++){
		for(int j = 0; j < input_size; j++){
			int t = i*input_size+j;
			B[t] += alpha/sqrt(input_size) * (x[j] * dh[i] - lambda * B[t]);
		}
	}

	for(int i = 0, j = 0; i < window_size; i++){
		int offset = id[i].word * words.element_size;
		for(int k = 0; k < words.element_size; k++,j++){
			int t = offset + k;
			words.value[t] += alpha * (dx[j] - lambda * words.value[t]);
		}
	}
}

double checkCase(data_t *positive, data_t *negative, bool gd = false){
	
	double hPositive[H] = {0};
	double hNegative[H] = {0};
	double xPositive[MAX_F];
	double xNegative[MAX_F];
	double vp = calcNet(positive, xPositive, hPositive);
	double vn = calcNet(negative, xNegative, hNegative);

	double ret = max(0.0, 1-vp+vn);

	if(gd && 1-vp+vn>0){ //修改参数
		bpNet(positive, xPositive, hPositive, 1*(1-vp*vp));
		bpNet(positive, xNegative, hPositive, -1*(1-vn*vn));
	}
	return ret;
}


void writeFile(const char *name, double *A, int size){
	FILE *fout = fopen(name, "wb");
	fwrite(A, sizeof(double), size, fout);
	fclose(fout);
}

int readFile(const char *name, double *A, int size){
	FILE *fin = fopen(name, "rb");
	if(!fin)
		return 0;
	int len = (int)fread(A, sizeof(double), size, fin);
	fclose(fin);
	return len;
}

const int chk_valid_size = 1000000;
data_t *chk_valid_pointer[chk_valid_size];
int chk_valid_center[chk_valid_size];

double checkSet(data_t *data, int N){
	int hw = (window_size-1)/2;

	for(int i = 0; i < chk_valid_size; i++){
		int s = rand() % N;
		data_t *positive = data + s * window_size;
		chk_valid_pointer[i] = positive;

		int center = 0;
		while(center == 0 || center == 2 || center == positive[hw].word){
			center = rand() % words.element_num;
		}
		chk_valid_center[i] = center;
	}

	double ret = 0;
	#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
	for(int i = 0; i < chk_valid_size; i++){
		data_t negative[10]; //要比window_size大
		data_t *positive = chk_valid_pointer[i];

		//生成负样本
		for(int j = 0; j < window_size; j++){
			negative[j] = positive[j];
		}
		negative[hw].word = chk_valid_center[i];

		double tv = checkCase(positive, negative);

		#pragma omp critical
		{
			ret += tv;
		}
	}
	return ret / chk_valid_size;
}

bool readParaFiles(){
	char fname[100];
	int size = 0;
	sprintf(fname, "%s_A", model_name);
	size += readFile(fname, A, H);
	sprintf(fname, "%s_B", model_name);
	size += readFile(fname, B, H*input_size);
	sprintf(fname, "%s_w", model_name);
	size += readFile(fname, words.value, words.size);
	if(size > 0)
		return true;
	else
		return false;
}

void writeParaFiles(const char *model_name){
	char fname[100];
	sprintf(fname, "%s_A", model_name);
	writeFile(fname, A, H);
	sprintf(fname, "%s_B", model_name);
	writeFile(fname, B, H*input_size);
	sprintf(fname, "%s_w", model_name);
	writeFile(fname, words.value, words.size);
}

//检查正确率和似然
//返回值是似然
double check(double training_result){
	double ret = 0;
	char fname[100];

	ret = checkSet(vdata, vN);

	sprintf(fname, "%s_%d_output", model_name, iter);

	double ps = 0;
	int pnum = 0;
	for(int i = 0; i < H; i++,pnum++){
		ps += A[i]*A[i];
	}
	for(int i = 0; i < H*input_size; i++,pnum++){
		ps += B[i]*B[i];
	}
	for(int i = 0; i < words.size; i++,pnum++){
		ps += words.value[i]*words.value[i];
	}

	writeParaFiles(model_name);

	double fret = ret + ps/pnum*lambda/2;
	printf("para:%lf, train: %.16lf, valid: %.16lf, time:%.1lf\n",
		ps/pnum/2, training_result/1000000, ret,
		getTime()-time_start);
	fflush(stdout);
	return fret;
}

unsigned long get_file_size(const char *filename){
	unsigned long size;
	FILE* fp = fopen( filename, "rb" );
	if(fp == NULL){
		//printf("ERROR: Open file %s failed.\n", filename);
		return 0;
	}
	fseek( fp, SEEK_SET, SEEK_END );
	size=ftell(fp);
	fclose(fp);
	return size;
}

int readDataFile(const char *name){
	int size = (int)get_file_size(name);
	if(size == 0)
		return 0;
	data = new data_t[size/sizeof(data_t)];
	FILE *fin = fopen(name, "rb");
	size = (int)fread(data, sizeof(data_t), size/sizeof(data_t), fin);
	N = size / window_size;
	fclose(fin);
	return 1;
}

void writeDataFile(const char *name){
	FILE *fout = fopen(name, "wb");
	fwrite(data, sizeof(data_t), N * window_size, fout);
	fclose(fout);
}

int main(int argc, char **argv){
	if(argc < 2){
		printf("Useage: ./embedding unsupervised.txt\n");
		return 0;
	}
	model_name = argv[0];
	train_file = argv[1];

	window_size = 5;
	vector_size = 50;
	input_size = window_size * vector_size;

	//初始化字典
	init(dict_file);
	words.init(vector_size, chk.size());

	char train_file_dat[100];
	sprintf(train_file_dat, "%s.dat", train_file);

	//查看文件是否存在
	if(!readDataFile(train_file_dat)){
		printf("read data\n");
		readAllData(train_file, window_size, data, N);
		writeDataFile(train_file_dat);
	}


	printf("init. input(features):%d, hidden:%d, alpha:%lf, lambda:%.16lf\n", input_size, H, alpha, lambda);
	printf("window_size:%d, vector_size:%d, vocab_size:%d, data_size:%d\n", window_size, vector_size, words.element_num, N);
	
	vN = N / 20;
	N -= vN;
	vdata = data + N * window_size;

	printf("training:%d, validation:%d\n", N, vN);


	A = new double[H];
	gA = new double[H];
	B = new double[H*input_size];
	gB = new double[H*input_size];

	for(int i = 0; i < H; i++){
		A[i] = (nextDouble()-0.5) / sqrt(H);
	}
	for(int i = 0; i < H * input_size; i++){
		B[i] = (nextDouble()-0.5) /sqrt(input_size);
	}
	for(int i = 0; i < words.size; i++){
		words.value[i] = (nextDouble()-0.5);
	}
	//先初始化，然后尝试读一下参数，有就覆盖
	readParaFiles();

	time_start = getTime();

	double lastLH = 1e100;
	double training_result = 0;
	double bestH = 1e100;
	while(1){
		//计算正确率
		printf("iter: %3d, alpha:%.10lf, ", iter, alpha);
		double LH = check(training_result);

		//保存最佳参数
		if(LH < bestH){
			char fname[100];
			sprintf(fname, "%s_%d", model_name, iter);
			writeParaFiles(fname);
			bestH = LH;
		}

		/*if(LH > lastLH){
			alpha *= 0.5;
		}
		lastLH = LH;*/

		iter++;

		double lastTime = getTime();
		data_t negative[10]; //要比window_size大
		int hw = (window_size-1)/2;
		training_result = 0;
		for(int i = 0; i < 10000000; i++){
			int s = rand()%N; //选取一个正样本

			data_t *positive = data + s * window_size;

			//生成负样本
			for(int j = 0; j < window_size; j++){
				negative[j] = positive[j];
			}
			int center = 0;
			while(center == 0 || center == 2 || center == positive[hw].word){
				center = rand() % words.element_num;
			}
			negative[hw].word = center;

			training_result += checkCase(positive, negative, true);

			if ((i%1000)==0){
			//	printf("%citer: %3d, alpha:%.10lf,                train: %.16f, Progress: %.2f%%, Pairs/sec: %.1f ", 13, iter, alpha, training_result/i, 100.*i/10000000, i/(getTime()-lastTime));
			}
		}
		//printf("%c", 13);
	}
	return 0;
}