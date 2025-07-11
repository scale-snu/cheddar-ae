#include <stdlib.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class MNIST {
 private:

 public:
  explicit MNIST(){};

  /****************************************************************************/
  double innerproduct(double* vec1, double* vec2, long size) {
    double ip = 0.;
    for (long i = 0; i < size; i++) {
      ip += vec1[i] * vec2[i];
    }
    return ip;
  }
  /****************************************************************************/
  double** zDataFromFile(std::string path, long& factorDim, long& sampleDim,
                         bool isfirst) {
    std::vector<std::vector<double> > zline;
    factorDim = 1;  // dimension of x
    sampleDim = 0;  // number of samples
    std::ifstream openFile(path.data());
    if (openFile.is_open()) {
      std::string line, temp;
      std::getline(openFile, line);
      long i;
      size_t start = 0;
      size_t end = 0;
      for (i = 0; i < line.length(); ++i) {
        if (line[i] == ',') factorDim++;
      }
      while (std::getline(openFile, line)) {
        std::vector<double> vecline;
        do {
          end = line.find_first_of(',', start);
          temp = line.substr(start, end);
          vecline.push_back(atof(temp.c_str()));
          start = end + 1;
        } while (start);
        zline.push_back(vecline);
        sampleDim++;
      }
    } else {
      std::cout << "Error: cannot read file" << std::endl;
    }
    openFile.close();
    double** zData = new double*[sampleDim];
    if (isfirst) {
      for (long j = 0; j < sampleDim; ++j) {
        double* zj = new double[factorDim];
        zj[0] = 2 * zline[j][0] - 1;
        for (long i = 1; i < factorDim; ++i) {
          zj[i] = zj[0] * zline[j][i];
        }
        zData[j] = zj;
      }
    } else {
      for (long j = 0; j < sampleDim; ++j) {
        double* zj = new double[factorDim];
        zj[0] = 2 * zline[j][factorDim - 1] - 1;
        for (long i = 1; i < factorDim; ++i) {
          zj[i] = zj[0] * zline[j][i - 1];
        }
        zData[j] = zj;
      }
    }
    return zData;
  }
  /****************************************************************************/
  void normalizeZData(double** zData, long factorDim, long sampleDim) {
    for (long i = 0; i < factorDim; ++i) {
      double m = 0.0;
      for (long j = 0; j < sampleDim; ++j) {
        if (m < abs(zData[j][i])) m = abs(zData[j][i]);
      }
      if (m > 1e-10) {
        for (long j = 0; j < sampleDim; ++j) {
          zData[j][i] = zData[j][i] / m;
        }
      }
    }
  }
  /****************************************************************************/
  void shuffleZData(double** zData, long factorDim, long sampleDim) {
    srand(time(NULL));
    double* tmp = new double[factorDim];
    for (long i = 0; i < sampleDim; ++i) {
      long idx = rand() % sampleDim;
      while (idx != i) {
        idx = rand() % sampleDim;
      }
      std::copy(zData[i], zData[i] + factorDim, tmp);
      std::copy(zData[idx], zData[idx] + factorDim, zData[i]);
      std::copy(tmp, tmp + factorDim, zData[idx]);
    }
    delete[] tmp;
  }
  /****************************************************************************/
  /****************************************************************************/
  void testAUROC(double& auc, double& accuracy, double** zData, long factorDim,
                 long sampleDim, double* wData, bool isfirst) {
    // print first 10 element of wData
    std::cout << "\t - wData = [";
    for (long i = 0; i < 10; i++) {
      std::cout << wData[i] << ',';
      if (i == 9) std::cout << wData[i] << "]" << std::endl;
    }

    // compute AUROC and accuracy of this model
    long TN = 0, FP = 0;
    std::vector<double> thetaTN(0);
    std::vector<double> thetaFP(0);

    for (long i = 0; i < sampleDim; ++i) {
      if (zData[i][0] > 0) {
        if (innerproduct(zData[i], wData, factorDim) < 0) {
          TN++;
        }
        thetaTN.push_back(zData[i][0] *
                          innerproduct(zData[i] + 1, wData + 1, factorDim - 1));
      } else {
        if (innerproduct(zData[i], wData, factorDim) < 0) {
          FP++;
        }
        thetaFP.push_back(zData[i][0] *
                          innerproduct(zData[i] + 1, wData + 1, factorDim - 1));
      }
    }
    accuracy = (double)(sampleDim - TN - FP) / sampleDim;
    std::cout << "\t - Accuracy: " << accuracy << std::endl;
    auc = 0.0;
    if (thetaFP.size() == 0 || thetaTN.size() == 0) {
      std::cout << "\t - n_test_yi = 0 : cannot compute AUC" << std::endl;
    } else {
      for (long i = 0; i < thetaTN.size(); ++i) {
        for (long j = 0; j < thetaFP.size(); ++j) {
          if (thetaFP[j] <= thetaTN[i]) auc++;
        }
      }
      auc /= thetaTN.size() * thetaFP.size();
      std::cout << "\t - AUC: " << auc << std::endl;
    }
    std::vector<double>().swap(thetaTN);  ///< to solve memory leakage problem
    std::vector<double>().swap(thetaFP);  ///< to solve memory leakage problem
  }
  /****************************************************************************/
};