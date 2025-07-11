#include <chrono>
#include <vector>

#include "PrepareInput.h"
#include "Testbed.h"
#include "core/MultiLevelCiphertext.h"
#include "extension/EvalPoly.h"
#include "extension/ExampleOps.h"
static int max_level = 15;

static constexpr int half_degree = 1 << 15;

static constexpr int warm_up = 0;

static constexpr int f = 256;  // equal to num slots;

static constexpr int num_iter = 32;
static int num_iter_per_boot = max_level / 5;

static constexpr double learning_rate = 1.0;

using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;
using Evk = EvaluationKey<word>;
using namespace cheddar::example_ops;

inline void AssertTrue(bool expr, const std::string &msg) {
  if (!expr) {
    std::cerr << "ERROR: " << msg << std::endl << std::flush;
    std::exit(EXIT_FAILURE);
  }
}
// performance options
static bool merged_bootstrap = true;

unsigned int Log2Ceil(unsigned int x) {
  if (x == 0) return 0;
  unsigned int res = 0;
  x--;
  while (x > 0) {
    x >>= 1;
    res++;
  }
  return res;
}

TEST_P(Testbed, HELR1024) {
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  max_level = boot_context->boot_param_.GetEndLevel();
  num_iter_per_boot = max_level / 5;

  int batch = 1024;
  int num_ctxt = (f * batch) / half_degree;  // 8;
  AssertTrue(num_ctxt > 0, "Too small batch used");

  double gamma = learning_rate / batch;

  PrepareBoot(f, false,
              merged_bootstrap ? BootVariant::kMergeTwoReal
                               : BootVariant::kImaginaryRemoving);
  AssertTrue(max_level % 5 == 0, "Max level should be a multiple of 5");

  // generating masks and rotation keys
  for (int i = 0; i <= Log2Ceil(half_degree) - 1; i++) {
    int rot_distance = 1 << i;
    interface_->PrepareRotationKey(rot_distance, max_level);
    interface_->PrepareRotationKey(half_degree - rot_distance, max_level);
  }

  // Constant For one
  std::vector<Constant<word>> constant_one(max_level + 1);
  std::vector<Constant<word>> constant_half(max_level / 3);
  for (int i = 0; i <= max_level; i++) {
    context_->encoder_.EncodeConstant(constant_one[i], i,
                                      context_->param_.GetScale(i), 1.0);
  }
  for (int i = 0; i < max_level / 3; i++) {
    context_->encoder_.EncodeConstant(constant_half[i], i * 3,
                                      context_->param_.GetScale(i * 3), 0.5);
  }

  // TODO: replace this with real data
  std::string mnist_path =
      std::string(PROJECT_ROOT) + "/MNIST_data/MNIST_train.txt";
  std::string test_data_path =
      std::string(PROJECT_ROOT) + "/MNIST_data/MNIST_test.txt";
  MNIST mnist;
  long factorDim_test, sampleDim_test;
  double **zDataT =
      mnist.zDataFromFile(test_data_path, factorDim_test, sampleDim_test, true);
  mnist.shuffleZData(zDataT, factorDim_test, sampleDim_test);
  mnist.normalizeZData(zDataT, factorDim_test, sampleDim_test);
  double auc, accuracy;
  long factorDim, sampleDim;
  double **zData = mnist.zDataFromFile(mnist_path, factorDim, sampleDim, true);
  mnist.shuffleZData(zData, factorDim, sampleDim);
  mnist.normalizeZData(zData, factorDim, sampleDim);
  std::vector<std::vector<Ct>> z_data(num_iter);
  std::vector<Constant<word>> eta_const(num_iter);
  std::vector<Constant<word>> one_minus_eta_const(num_iter);

  double alpha0 = 0.01;
  double alpha1 = (1.0 + std::sqrt(1.0 + 4.0 * alpha0 * alpha0)) / 2.0;

  int numBatch = sampleDim / batch;
  for (int it = 0; it < num_iter; it++) {
    z_data[it].resize(num_ctxt);
    int level = max_level - (it % num_iter_per_boot) * 5;
    double z_scale = context_->param_.GetScale(level);
    for (int i = 0; i < num_ctxt; i++) {
      std::vector<Complex> z(half_degree, 0);
      int samplePerCtxt = half_degree / f;
      int batch_offset = (it % numBatch) * batch + i * samplePerCtxt;
      for (int j = 0; j < samplePerCtxt; j++) {
        for (int k = 0; k < f; k++) {
          if (k >= factorDim) {
            z[j * f + k] = Complex(0.0, 0.0);
          } else {
            z[j * f + k] = Complex(zData[batch_offset + j][k], 0.0);
          }
        }
      }
      EncodeAndEncrypt(z_data[it][i], z, level);
    }

    double eta = (1 - alpha0) / alpha1;
    level = max_level - (it % num_iter_per_boot) * 5 - 4;
    double scale = context_->param_.GetScale(level);
    context_->encoder_.EncodeConstant(one_minus_eta_const[it], level, scale,
                                      1.0 - eta);
    scale = (scale * scale) / z_scale;
    context_->encoder_.EncodeConstant(eta_const[it], level, scale, eta);

    alpha0 = alpha1;
    alpha1 = (1.0 + std::sqrt(1.0 + 4.0 * alpha0 * alpha0)) / 2.0;
  }

  // Prepare y_mask and constants.
  std::vector<Pt> y_mask(num_iter_per_boot);
  std::vector<Complex> plain_y_mask(half_degree, Complex(0.0, 0.0));
  for (int i = 0; i < half_degree; i += f) {
    plain_y_mask[i] = Complex(1.0, 0.0);
  }
  for (int i = 0; i < num_iter_per_boot; i++) {
    int level = i * 5 + 4;
    Encode(y_mask[i], plain_y_mask, level);
  }

  // sigmoid(x) = 0.5 - 0.0843 * x + 0.0002 * x^3
  Sigmoid<word> sigmoid(boot_context, f, num_iter_per_boot, gamma);

  // Prepare initial data
  std::vector<Complex> zero(f, Complex(0.0, 0.0));
  Ciphertext<word> w_data, v_data;

  __ProfileStart("HELR1024", warm_up, {
    EncodeAndEncrypt(w_data, zero, max_level);
    EncodeAndEncrypt(v_data, zero, max_level);
  });

  for (int i = 0; i < num_iter; i++) {
    int target_level = max_level - (i % num_iter_per_boot) * 5 - 5;

    std::vector<Ct> inner_prod;
    InnerProduct(boot_context, inner_prod, z_data[i], v_data,
                 y_mask[num_iter_per_boot - 1 - (i % num_iter_per_boot)],
                 interface_->GetEvkMap(), f);
    Ct grad;
    sigmoid.Evaluate(grad, z_data[i], inner_prod, interface_->GetEvkMap());

    Ct tmp2;
    context_->LevelDown(tmp2, v_data, context_->param_.NPToLevel(grad.GetNP()));
    context_->Add(grad, grad, tmp2);
    // update
    Ct tmp;
    context_->Mult(tmp, grad, one_minus_eta_const[i]);
    int eta_const_level = context_->param_.NPToLevel(eta_const[i].GetNP());
    int w_level = context_->param_.NPToLevel(w_data.GetNP());
    auto mlct = MultiLevelCiphertext<word>(std::move(w_data));
    while (!context_->IsMultUnsafeCompatible(eta_const_level, w_level)) {
      w_level -= 1;
    }
    // mlct.AllocateLevel(w_level);
    for (int j = mlct.GetMinLevel() - 1; j >= w_level; j--) {
      NPInfo np = context_->param_.LevelToNP(j);
      mlct.AllocateLevel(j);
      Ct tmp;
      context_->Mult(tmp, mlct.AtLevel(j + 1), mlct.GetLevelDownConst(j + 1));
      context_->Rescale(mlct.AtLevel(j), tmp);
    }
    context_->MadUnsafe(tmp, mlct.AtLevel(w_level), eta_const[i]);
    context_->Rescale(v_data, tmp);
    context_->LevelDown(w_data, grad, target_level);

    if (i != num_iter - 1 && target_level == 0) {
      if (!merged_bootstrap) {
        std::vector<Complex> tmp;
        boot_context->Boot(w_data, w_data, interface_->GetEvkMap());
        boot_context->Boot(v_data, v_data, interface_->GetEvkMap());
        continue;
      }

      // Merged bootstrapping for real numbers
      context_->MultImaginaryUnit(w_data, w_data);
      context_->Add(v_data, v_data, w_data);
      // context_->Boot(w_data, w_data, interface_->GetEvkMap());
      boot_context->Boot(v_data, v_data, interface_->GetEvkMap());

      context_->HConj(tmp, v_data, interface_->GetConjugationKey());
      context_->Sub(w_data, tmp, v_data);
      context_->MultImaginaryUnit(w_data, w_data);
      context_->Add(v_data, v_data, tmp);
    }
  }

  __ProfileEnd("HELR1024");
  std::vector<Complex> res_w_data;
  DecryptAndDecode(res_w_data, w_data);
  double *w_data_host = new double[f];
  for (int i = 0; i < f; i++) {
    if (i < factorDim) {
      w_data_host[i] = res_w_data[i].real();
    } else {
      w_data_host[i] = 0.0;
    }
  }
  mnist.testAUROC(auc, accuracy, zDataT, factorDim_test, sampleDim_test,
                  w_data_host, true);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("helrparam_30.json", "helrparam_35.json",
                    "helrparam_40.json", "helrparam_48.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });