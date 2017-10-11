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
    FLTARY b, top, grads;
    InnerProduct(int input, int output) {
        n = output;
        m = input;
        std::random_device rd; // random seed generator
        std::mt19937 rg(rd()); // random generator
        std::normal_distribution<> normDist(0, 0.1); // normal distribution
        w.resize(output);
        b.resize(output);
        top.resize(output);
        grads.resize(input);
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
    void backward_pass(FLTARY top_grads, FLTARY bottom, double base_lr) {
        grads.assign(m, 0);
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < m; ++j) {
                grads[j] += top_grads[i] * w[i][j];
                w[i][j] -= base_lr * top_grads[i] * bottom[j];
            }
            b[i] -= base_lr * top_grads[i];
        }

    }
};

class ReLU{
public:
    int n;
    FLTARY top, grads;
    ReLU(int output) {
        n = output;
        top.resize(output);
        grads.resize(output);
    }
    void forward_pass(FLTARY bottom) {
        for(int i = 0; i < n; ++i) {
            top[i] = bottom[i] > 0 ? bottom[i] : 0;
        }
    }
    void backward_pass(FLTARY top_grads) {
        for(int i = 0; i < n; ++i) {
            grads[i] = top[i] > 0 ? top_grads[i] : 0;
        }
    }
};

class SoftmaxWithLoss{
public:
    int n;
    FLTARY top, grads;
    SoftmaxWithLoss(int output) {
        n = output;
        top.resize(output);
        grads.resize(output);
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
    void backward_pass(float label) {
        for(int i = 0; i < n; ++i) {
            grads[i] = top[i] - (label == i);
        }
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
    int nImgRows, nImgCols, tp, fp;
    double base_lr = 0.1;
	std::vector<FLTARY> trainImgs, testImgs;
	FLTARY trainLabels;
	FLTARY testLabels;
	testLabels.resize(500);
	std::ifstream ifs("train_2000a.txt");
	LoadData(ifs, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
//	LoadData(std::cin, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
	ifs.close();
	ifs.open("label_2000a.txt");
	for(int i = 0; i < 500; ++i) {
        ifs >> testLabels[i];
	}
	ifs.close();
    InnerProduct fc1(nImgRows * nImgCols, 100), fc2(100, 10);
//    InnerProduct fc(nImgRows * nImgCols, 10);
    ReLU relu(100);
    SoftmaxWithLoss loss(10);
    for(int i = 0; i < 10000 / trainImgs.size(); ++i) {//max_iter
        for(int j = 0; j < trainImgs.size(); ++j) {//max_iter
            if (!(j % 2000)) {//stepsize
                base_lr *= 0.1;//gammar
            }
            fc1.forward_pass(trainImgs[j]);
            relu.forward_pass(fc1.top);
            fc2.forward_pass(relu.top);
            loss.forward_pass(fc2.top);
            loss.backward_pass(trainLabels[j]);
            fc2.backward_pass(loss.grads, relu.top, base_lr);
            relu.backward_pass(fc2.grads);
            fc1.backward_pass(relu.grads, trainImgs[j], base_lr);
        }
        std::cout << i;
        tp = fp = 0;
        for(int k = 0; k < trainImgs.size(); ++k) {
            fc1.forward_pass(trainImgs[k]);
            relu.forward_pass(fc1.top);
            fc2.forward_pass(relu.top);
            loss.forward_pass(fc2.top);
            std::max_element(loss.top.begin(), loss.top.end()) - loss.top.begin() == trainLabels[k] ? ++tp : ++fp;
        }
        std::cout << "\ttrain:\t" << 1.0 * tp / (tp + fp);
        tp = fp = 0;
        for(int k = 0; k < testImgs.size(); ++k) {
            fc1.forward_pass(testImgs[k]);
            relu.forward_pass(fc1.top);
            fc2.forward_pass(relu.top);
            loss.forward_pass(fc2.top);
            std::max_element(loss.top.begin(), loss.top.end()) - loss.top.begin() == testLabels[k] ? ++tp : ++fp;
        }
        std::cout << "\ttest:\t" << 1.0 * tp / (tp + fp) <<std::endl;
    }
	system("pause");
	return 0;
}
