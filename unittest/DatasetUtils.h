#pragma once

#include "cnpy.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>

typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> Matrix_t;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<long double, 1, Eigen::Dynamic> RowVector;

static std::default_random_engine generator;

class CIFAR {
private:
  std::string data_dir;

public:
  Matrix_t train_data;
  Matrix_t train_labels;
  Matrix_t test_data;
  Matrix_t test_labels;

  void read_cifar_data(std::string filename, Matrix_t &data, Matrix_t &labels){
    std::cout << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("File does not exist");
    }
    if (file.is_open()) {
        const int number_of_images = 10000;
        const int n_channels = 3;
        const int n_rows = 32;
        const int n_cols = 32;
        data.resize(n_cols * n_rows * n_channels, number_of_images);
        labels.resize(number_of_images, 1);
        for (int i = 0; i < number_of_images; i++) {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels(i) = (long double)label;
        for (int ch = 0; ch < n_channels; ch++) {
            for (int r = 0; r < n_rows; r++) {
            for (int c = 0; c < n_cols; c++) {
                unsigned char image = 0;
                file.read((char *)&image, sizeof(image));
                data(ch * n_rows * n_cols + r * n_cols + c, i) = (long double)image;
            }
            }
        }
        }
    }
  }
void transform(std::vector<double> mean, std::vector<double> stdev) {
    const int number_of_images = 10000;
    const int n_channels = 3;
    const int n_rows = 32;
    const int n_cols = 32;
    Matrix_t mean_mat;
    Matrix_t stdev_mat;


    test_data.block<n_cols * n_rows, number_of_images>(0, 0) -=
            Matrix_t::Constant(n_cols * n_rows, number_of_images, mean[0]);
    test_data.block<n_cols * n_rows, number_of_images>(0, 0) /= stdev[0];

    test_data.block<n_cols * n_rows, number_of_images>(n_cols * n_rows, 0) -=
            Matrix_t::Constant(n_cols * n_rows, number_of_images, mean[1]);
    test_data.block<n_cols * n_rows, number_of_images>(n_cols * n_rows, 0) /=
            stdev[1];

    test_data.block<n_cols * n_rows, number_of_images>(2 * n_cols * n_rows, 0) -=
            Matrix_t::Constant(n_cols * n_rows, number_of_images, mean[2]);
    test_data.block<n_cols * n_rows, number_of_images>(2 * n_cols * n_rows, 0) /=
            stdev[2];
}

explicit CIFAR(std::string data_dir) : data_dir(data_dir) {}

void read() {
    std::cout << data_dir + "/cifar-10-batches-bin/data_batch_1.bin" << std::endl;
    read_cifar_data(data_dir + "/cifar-10-batches-bin/data_batch_1.bin",
                    train_data, train_labels);
    read_cifar_data(data_dir + "/cifar-10-batches-bin/test_batch.bin",
                    test_data, test_labels);
}
};

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(long double* arr, int n, long double mu,
                              long double sigma) {
  std::normal_distribution<long double> distribution(mu, sigma);
  for (int i = 0; i < n; i++) {
    arr[i] = distribution(generator);
  }
}

// shuffle cols of matrix
inline void shuffle_data(Matrix_t& data, Matrix_t& labels) {
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(data.cols());
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(),
                      perm.indices().data() + perm.indices().size());
  data = data * perm;  // permute columns
  labels = labels * perm;
}

// encode discrete values to one-hot values
inline Matrix_t one_hot_encode(const Matrix_t& y, int n_value) {
  int n = y.cols();
  Matrix_t y_onehot = Matrix_t::Zero(n_value, n);
  for (int i = 0; i < n; i++) {
    y_onehot(int(y(i)), i) = 1;
  }
  return y_onehot;
}

// classification accuracy
inline long double compute_accuracy(const Matrix_t& preditions,
                                    const Matrix_t& labels) {
  int n = preditions.cols();
  long double acc = 0;
  for (int i = 0; i < n; i++) {
    Matrix_t::Index max_index;
    long double max_value = preditions.col(i).maxCoeff(&max_index);
    acc += int(max_index) == labels(i);
  }
  return acc / n;
}

