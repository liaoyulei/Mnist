#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cmath>
typedef std::vector<float> FLTARY;

class InnerProduct{
public:
    int n, m;
    std::vector<FLTARY> w;
    FLTARY b, top;
    InnerProduct(int input, int output) {
        n = output;
        m = input;
        std::random_device rd; // random seed generator
        std::mt19937 rg(rd()); // random generator
        std::normal_distribution<> normDist(0, 0.1); // normal distribution
        w.resize(output);
        b.resize(output);
        top.resize(output);
        for(int i = 0; i < output; ++i) {
            w[i].resize(input);
            for(int j = 0; j < input; ++j) {
                w[i][j] = normDist(rg);
            }
            b[i] = normDist(rg);
        }
    }
    void forward_pass(FLTARY bottom) {
        for(int i = 0;i < n; ++i) {
            top[i] = b[i];
            for(int j = 0; j < m; ++j) {
                top[i] += w[i][j] * bottom[j];
            }
        }
    }
    void backward_pass() {
    }
};

class ReLU{
public:
    int n;
    FLTARY top;
    ReLU(int output) {
        n = output;
        top.resize(output);
    }
    void forward_pass(FLTARY bottom) {
        for(int i = 0; i < n; ++i) {
            top[i] = bottom[i] > 0 ? bottom[i] : 0;
        }
    }
};

class SoftmaxWithLoss{
public:
    int n;
    FLTARY top;
    SoftmaxWithLoss(int output) {
        n = output;
        top.resize(output);
    }
    void forward_pass(FLTARY bottom) {
        double ymax = *std::max_element(bottom.begin(), bottom.end()), sum = 0;
        for(int i = 0; i < n; ++i) {
            top[i] = exp(bottom[i] - ymax);
            sum += top[i];
        }
        for(int i = 0; i < n; ++i) {
            top[i] /= sum;
        }
    }
    void back_pass() {

    }
};

template<typename _IS>
void LoadData(_IS &inStream, int *pImgRows, int *pImgCols,
	std::vector<FLTARY> &trainImages, FLTARY &trainLabels,
	std::vector<FLTARY> &testImages) {
	int nTrainCnt, nTestCnt;
	inStream >> nTrainCnt >> nTestCnt >> *pImgRows >> *pImgCols;
	int nImgArea = *pImgRows * *pImgCols, n = 41;
	trainImages.resize(nTrainCnt);
	trainLabels.resize(nTrainCnt);
	testImages.resize(nTestCnt);
	for (int i = 0; i < nTrainCnt + nTestCnt; ++i) {
		std::string strLine;
		inStream >> strLine;
		std::vector<float> fltBuf(nImgArea);
		for (int j = 0; j < nImgArea / 2; ++j) {
			const char *p = strLine.c_str() + j * 3;
			int rawCode = (int)(p[0] - '0') * n * n;
			rawCode += (int)(p[1] - '0') * n;
			rawCode += (int)(p[2] - '0');
			fltBuf[j * 2 + 0] = ((rawCode & 0xFF) - 128.0f) / 255.0f;
			fltBuf[j * 2 + 1] = ((rawCode >> 8) - 128.0f) / 255.0f;
		}
		if (i < nTrainCnt) {
			fltBuf.swap(trainImages[i]);
			inStream >> trainLabels[i];
		}
		else fltBuf.swap(testImages[i - nTrainCnt]);
	}
}

int main() {
    int nImgRows, nImgCols, batch = 1000;
    double base_lr = 0.1;
	std::vector<FLTARY> trainImgs, testImgs;
	FLTARY trainLabels;
	std::ifstream ifs("train_2000a.txt");
	LoadData(ifs, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
//	LoadData(std::cin, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
	ifs.close();
//    InnerProduct fc1(nImgRows * nImgCols, 500), fc2(500, 10);
    InnerProduct fc(nImgRows * nImgCols, 10);
    ReLU relu(500);
    SoftmaxWithLoss loss(10);
    for(int i = 0; i < 100; ++i) {//max_iter
        if (!(i % 300)) {//sttesize
            base_lr *= 0.1;//gammar
        }
        for(int j = 0; j < trainImgs.size(); j += batch) {
            for(int k = 0; k < batch; ++k) {
            /*    fc1.forward_pass(trainImgs[j + k]);
                relu.forward_pass(fc1.top);
                fc2.forward_pass(relu.top);
                loss.forward_pass(fc2.top);*/
                fc.forward_pass(trainImgs[j + k]);
                loss.forward_pass(fc.top);
            }
        }
    }
	system("pause");
	return 0;
}
