#include <unistd.h>

#include "DatasetUtils.h"
#include "Testbed.h"
#include "cnpy.h"
#include "extension/EvalPoly.h"
#include "extension/ExampleOps.h"
#include "extension/StripedMatrix.h"

using namespace cheddar::example_ops;

struct WeightPath {
  std::string weight_path;
  std::string bias_path;
};

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

inline void AssertTrue(bool expr, const std::string &msg) {
  if (!expr) {
    std::cerr << "ERROR: " << msg << std::endl << std::flush;
    std::exit(EXIT_FAILURE);
  }
}
template <typename Int>
bool IsPowOfTwo(Int a) {
  static_assert(std::is_integral<Int>::value,
                "IsPowOfTwo only accepts integers");
  AssertTrue(a > 0, "IsPowOfTwo only accepts positive values");
  return (a & (a - 1)) == 0;
}
static std::vector<std::vector<WeightPath>> path_list = {
    // path_list[0]
    {
        {std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.weight",
         std::string(PROJECT_ROOT) + "/resnet20_fused/conv1_reparam.bias"},
    },
    // path_list[1]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.0.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.0.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.0.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.0.conv2_reparam.bias"},
    },
    // path_list[2]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.1.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.1.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.1.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.1.conv2_reparam.bias"},
    },
    // path_list[3]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.2.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.2.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.2.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer1.2.conv2_reparam.bias"},
    },
    // path_list[4]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.conv2_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.shortcut_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.0.shortcut_reparam.bias"},
    },
    // path_list[5]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.1.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.1.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.1.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.1.conv2_reparam.bias"},
    },
    // path_list[6]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.2.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.2.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.2.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer2.2.conv2_reparam.bias"},
    },
    // path_list[7]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.conv2_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.shortcut_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.0.shortcut_reparam.bias"},
    },
    // path_list[8]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.1.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.1.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.1.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.1.conv2_reparam.bias"},
    },
    // path_list[9]
    {
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.2.conv1_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.2.conv1_reparam.bias"},
        {std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.2.conv2_reparam.weight",
         std::string(PROJECT_ROOT) +
             "/resnet20_fused/layer3.2.conv2_reparam.bias"},
    },
    // path_list[10]
    {{std::string(PROJECT_ROOT) + "/resnet20_fused/linear.weight",
      std::string(PROJECT_ROOT) + "/resnet20_fused/linear.bias"}}};

static int max_level = 8;
static int relu1_level = 8;
static int relu2_level = 4;
static int relu3_level = 8;
static constexpr double relu_range = 10;

static constexpr int conv_level = 2;
static constexpr int pool_level = 2;
static constexpr int fc_level = 1;

static constexpr int half_degree = 1 << 15;

static constexpr int resnet_input_width = 32;
static constexpr int resnet_input_pack = 1;
static constexpr int resnet_input_channel = 3;

static constexpr int warm_up = 0;

// "pack * width == x_width" should satisfy
static constexpr int x_width = 32;
static constexpr int z_width = half_degree / (x_width * x_width);

using Ct = Ciphertext<word>;
using Pt = Plaintext<word>;
using Evk = EvaluationKey<word>;
using Complex = std::complex<double>;

Ct TMP;
bool directoryExists(const std::string &directory) {
  struct stat buffer;
  return (stat(directory.c_str(), &buffer) == 0);
}

void downloadCifar10Data() {
  std::string script = R"(
        #!/bin/bash
        wget -P cifar10_data https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
        tar -xvzf cifar10_data/cifar-10-binary.tar.gz -C cifar10_data
    )";

  std::ofstream file("cifar10_data/download.sh");
  file << script;
  file.close();

  std::string command =
      "chmod +x cifar10_data/download.sh && ./cifar10_data/download.sh";
  system(command.c_str());
}
void AddRequiredRotationsForTrace(EvkRequest &rotations, int start_rot_amount,
                                  int num_acuum, int level) {
  for (int i = 1; i < num_acuum; i *= 2) {
    int rot_amount = (start_rot_amount * i) % half_degree;
    if (rot_amount < 0) rot_amount += half_degree;
    rotations.AddRequest(rot_amount, level);
  }
}

void AdjustLevelWithBoot(std::shared_ptr<BootContext<word>> context, Ct &input,
                         int level, const EvkMap<word> &evk_map) {
  // static int boot_count = 0;
  if (context->param_.NPToLevel(input.GetNP()) < level) {
    // simple bootstrapping
    context->Boot(input, input, evk_map, false);

    // boot_count += 1;
  }
  AssertTrue(context->param_.NPToLevel(input.GetNP()) >= level,
             "Bootstrapping does not support level " + std::to_string(level));
  if (context->param_.NPToLevel(input.GetNP()) > level) {
    context->LevelDown(input, input, level);
  }
  // std::cout << "Boot Count: " << boot_count << std::endl;
}

class ResNetBlock {
 private:
  std::shared_ptr<BootContext<word>> context_;

  ConvBN<word> conv1_;
  ConvBN<word> conv2_;
  std::shared_ptr<EvalReLU<word>> relu_;
  std::unique_ptr<DownSample<word>> downsample_ = nullptr;

 public:
  ResNetBlock(std::shared_ptr<BootContext<word>> context, int input_width,
              int input_channel, int input_pack, bool narrowing,
              std::shared_ptr<EvalReLU<word>> relu,
              std::vector<WeightPath> &param_paths)
      : context_{context},
        conv1_{context,
               input_width,
               input_channel,
               input_pack,
               narrowing ? input_channel * 2 : input_channel,
               relu_range,
               narrowing,
               cnpy::npy_load(param_paths[0].weight_path).data<float>(),
               cnpy::npy_load(param_paths[0].bias_path).data<float>()},
        conv2_{context,
               conv1_.output_width_,
               conv1_.output_channel_,
               conv1_.output_pack_,
               conv1_.output_channel_,
               relu_range,
               false,
               cnpy::npy_load(param_paths[1].weight_path).data<float>(),
               cnpy::npy_load(param_paths[1].bias_path).data<float>()},
        relu_{relu} {
    if (narrowing) {
      downsample_ = std::make_unique<DownSample<word>>(
          context, input_width, input_channel, input_pack, input_channel * 2,
          relu_range, cnpy::npy_load(param_paths[2].weight_path).data<float>(),
          cnpy::npy_load(param_paths[2].bias_path).data<float>());
    }
  }

  void Evaluate(Ct &res, Ct &ct, const EvkMap<word> &evk_map) {
    Ct tmp;
    conv1_.Evaluate(tmp, ct, evk_map);
    relu_->Evaluate(tmp, tmp, evk_map);
    conv2_.Evaluate(tmp, tmp, evk_map);
    if (downsample_ != nullptr) {
      downsample_->Evaluate(res, ct, evk_map);
      context_->Copy(TMP, res);
      context_->Add(res, res, tmp);
    } else {
      context_->LevelDown(ct, ct, conv_level - 2);
      context_->Add(res, tmp, ct);
    }
    relu_->Evaluate(res, res, evk_map);
  }

  void AddRequiredRotations(EvkRequest &rotations) {
    conv1_.AddRequiredRotations(rotations);
    conv2_.AddRequiredRotations(rotations);
    if (downsample_ != nullptr) {
      downsample_->AddRequiredRotations(rotations);
    }
  }
};
StripedMatrix ConvertToStripedMatrix(
    const std::vector<std::vector<Complex>> &a) {
  int height = a.size();
  int width = a[0].size();
  StripedMatrix c(height, width);

  for (int i = 0; i < width; i++) {
    std::vector<Complex> diag(height, Complex(0));
    bool all_zero = true;
    for (int j = 0; j < height; j++) {
      diag[j] = a[j][(i + j) % width];
      all_zero = all_zero && (diag[j] == Complex(0));
    }
    if (!all_zero) {
      c.try_emplace(i, diag);
    }
  }

  return c;
}
void SetupLevel(std::shared_ptr<BootContext<word>> context) {
  max_level = context->boot_param_.GetEndLevel();
  relu1_level = max_level;
  relu2_level = max_level - 4;
  if (relu2_level <= 0) relu2_level += max_level;
  relu3_level = relu2_level - 4;
  if (relu3_level <= 0) relu3_level += max_level;
}

TEST_P(Testbed, ResNet20) {
  MultiLevelCiphertext<word>::StaticInit(context_->param_, context_->encoder_);
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  SetupLevel(boot_context);
  PrepareBoot(1 << 14, false, BootVariant::kImaginaryRemoving);
  // PrepareBoot(1 << 13, false, BootVariant::kImaginaryRemoving);
  // PrepareBoot(1 << 12, false, BootVariant::kImaginaryRemoving);

  auto relu = std::make_shared<EvalReLU<word>>(
      boot_context, max_level, relu1_level, relu2_level, relu3_level);

  EvkRequest rotations;

  // ConvBN conv0(
  // context_, 32, 3, 1, 16, false,
  // {"./resnet20_fused/test_weight.npy", "./resnet20_fused/test_bias.npy"});
  ConvBN conv0(boot_context, 32, 3, 1, 16, relu_range, false,
               cnpy::npy_load(path_list[0][0].weight_path).data<float>(),
               cnpy::npy_load(path_list[0][0].bias_path).data<float>(), true);
  std::cout << "Conv0 instantiated" << std::endl;
  ConvBN conv1_0(boot_context, 32, 16, 1, 16, relu_range, false,
                 cnpy::npy_load(path_list[1][0].weight_path).data<float>(),
                 cnpy::npy_load(path_list[1][0].bias_path).data<float>());
  ResNetBlock block1_1(boot_context, 32, 16, 1, false, relu, path_list[1]);
  std::cout << "Block1_1 instantiated" << std::endl;
  ResNetBlock block1_2(boot_context, 32, 16, 1, false, relu, path_list[2]);
  std::cout << "Block1_2 instantiated" << std::endl;
  ResNetBlock block1_3(boot_context, 32, 16, 1, false, relu, path_list[3]);
  std::cout << "Block1_3 instantiated" << std::endl;
  ResNetBlock block2_1(boot_context, 32, 16, 1, true, relu, path_list[4]);
  std::cout << "Block2_1 instantiated" << std::endl;
  ResNetBlock block2_2(boot_context, 16, 32, 2, false, relu, path_list[5]);
  std::cout << "Block2_2 instantiated" << std::endl;
  ResNetBlock block2_3(boot_context, 16, 32, 2, false, relu, path_list[6]);
  std::cout << "Block2_3 instantiated" << std::endl;
  ResNetBlock block3_1(boot_context, 16, 32, 2, true, relu, path_list[7]);
  std::cout << "Block3_1 instantiated" << std::endl;
  ResNetBlock block3_2(boot_context, 8, 64, 4, false, relu, path_list[8]);
  std::cout << "Block3_2 instantiated" << std::endl;
  ResNetBlock block3_3(boot_context, 8, 64, 4, false, relu, path_list[9]);
  std::cout << "Block3_3 instantiated" << std::endl;

  conv0.AddRequiredRotations(rotations);
  conv1_0.AddRequiredRotations(rotations);
  block1_1.AddRequiredRotations(rotations);
  block1_2.AddRequiredRotations(rotations);
  block1_3.AddRequiredRotations(rotations);
  block2_1.AddRequiredRotations(rotations);
  block2_2.AddRequiredRotations(rotations);
  block2_3.AddRequiredRotations(rotations);
  block3_1.AddRequiredRotations(rotations);
  block3_2.AddRequiredRotations(rotations);
  block3_3.AddRequiredRotations(rotations);

  constexpr int pool_input_width = 8;
  constexpr int pool_pack = 4;
  constexpr int pool_channel = 64;

  int pool_input_size = pool_input_width * pool_input_width * pool_channel;
  int pool_z_channel = pool_channel / (pool_pack * pool_pack);
  int pool_input_repeat = half_degree / pool_input_size;

  PlainHoistMap pool_mask;
  // using gs = 1;
  pool_mask.try_emplace(0, std::map<int, Message>());

  for (int i = 0; i < pool_z_channel; i++) {
    for (int j = 0; j < pool_pack; j++) {
      int rot_src = (i * x_width * x_width) + j * x_width;
      int rot_dst = pool_pack * (i * pool_pack + j);
      int rot_idx = rot_src - rot_dst;
      if (rot_idx < 0) rot_idx += half_degree;
      pool_mask[0].try_emplace(rot_idx, half_degree);
      auto &message = pool_mask[0][rot_idx];
      for (int l = 0; l < half_degree; l++) {
        message[l] = Complex(0, 0);
      }
      for (int l = 0; l < pool_pack; l++) {
        message[rot_dst + l] = Complex(
            1.0 / double(pool_input_width * pool_input_width) * relu_range, 0);
      }
    }
  }

  HoistHandler<word> avg_pool(boot_context, pool_mask, pool_level,
                              boot_context->param_.GetScale(pool_level), true);
  avg_pool.AddRequiredRotations(rotations);
  AddRequiredRotationsForTrace(rotations, pool_pack, pool_input_width,
                               pool_level);
  AddRequiredRotationsForTrace(rotations, x_width * pool_pack, pool_input_width,
                               pool_level);
  AddRequiredRotationsForTrace(rotations, pool_channel,
                               half_degree / pool_input_width, pool_level - 1);

  constexpr int fc_input_width = 64;
  constexpr int fc_output_width = 10;

  // std::map<int, std::map<int, Message>> hoist_map;
  PlainHoistMap fc_hoist_map;

  std::vector<std::vector<Complex>> fc_weight(
      fc_input_width, std::vector<Complex>(fc_input_width, 0.0));
  cnpy::NpyArray fc_weight_npy = cnpy::npy_load(path_list[10][0].weight_path);
  cnpy::NpyArray fc_bias_npy = cnpy::npy_load(path_list[10][0].bias_path);
  for (int i = 0; i < fc_input_width; i++) {
    if (i < fc_output_width) {
      for (int j = 0; j < fc_input_width; j++) {
        fc_weight[i][j] = fc_weight_npy.data<float>()[i * fc_input_width + j];
      }
    } else {
      for (int j = 0; j < fc_input_width; j++) {
        fc_weight[i][j] = 0.0;
      }
    }
  }
  constexpr int fc_bs = 8;
  constexpr int fc_gs = 8;
  LinearTransform<word> fc(boot_context, ConvertToStripedMatrix(fc_weight),
                           fc_level, boot_context->param_.GetScale(fc_level),
                           fc_bs, fc_gs, 0, 0);
  fc.AddRequiredRotations(rotations);
  Pt fc_bias;
  std::vector<Complex> plain_fc_bias(fc_input_width);
  for (int i = 0; i < fc_output_width; i++) {
    plain_fc_bias[i] = fc_bias_npy.data<float>()[i];
  }
  for (int j = fc_output_width; j < fc_input_width; j++) {
    plain_fc_bias[j] = 0.0;
  }
  boot_context->encoder_.Encode(fc_bias, fc_level - 1,
                                boot_context->param_.GetScale(fc_level - 1),
                                plain_fc_bias);

  interface_->PrepareRotationKey(rotations);

  // Get CIFAR-10 data
  std::string datasetDirectory = "cifar10_data";
  if (!directoryExists(datasetDirectory)) {
    int result = mkdir(datasetDirectory.c_str(), 0777);
    if (result != 0) {
      std::cerr << "Failed to create directory: " << datasetDirectory
                << std::endl;
      return;
    }
    downloadCifar10Data();
  }
  int num_test_images = 1;
  CIFAR *cifar;
  Matrix_t output;
  Matrix_t test_data;
  Matrix_t test_labels;
  output.resize(10, num_test_images);

  cifar = new CIFAR("./" + datasetDirectory);
  cifar->read();
  cifar->transform({0, 0, 0}, {255, 255, 255});
  cifar->transform({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010});
  test_data = cifar->test_data;
  test_labels = cifar->test_labels;
  std::vector<Complex> msg;
  std::cout << "size of cifar->test_data: " << test_data.rows() << "x"
            << test_data.cols() << std::endl;
  std::cout << "size of cifar->test_labels: " << test_labels.rows() << "x"
            << test_labels.cols() << std::endl;

  test_data.conservativeResize(32 * 32 * 4, num_test_images);
  for (int i = 32 * 32 * 3; i < 32 * 32 * 4; i++)
    for (int j = 0; j < num_test_images; j++) test_data(i, j) = 0;
  std::cout << "loaded cifar10 data!\n";

  int msg_size = 1 << Log2Ceil(resnet_input_width * resnet_input_width *
                               resnet_input_channel);

  Ct main_ct;
  std::vector<Complex> input_vecs(4 * 32 * 32, 0);
  std::vector<Complex> output_vec;
  for (int i = 0; i < num_test_images; i++) {
    for (int j = 0; j < 4 * 32 * 32; j++) {
      if (j < 3 * 32 * 32) {
        input_vecs[j] = Complex(test_data(j, i), 0.0);
      } else {
        input_vecs[j] = Complex(0.0, 0.0);
      }
    }
    __ProfileStart("ResNet20", warm_up,
                   EncodeAndEncrypt(main_ct, input_vecs, conv_level));
    // Actual evaluation
    std::cout << "-- Conv 0 --" << std::endl;
    conv0.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    relu->Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Block 1 --" << std::endl;
    block1_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block1_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block1_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Block 2 --" << std::endl;
    block2_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block2_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block2_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    std::cout << "-- Block 3 --" << std::endl;
    block3_1.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block3_2.Evaluate(main_ct, main_ct, interface_->GetEvkMap());
    block3_3.Evaluate(main_ct, main_ct, interface_->GetEvkMap());

    std::cout << "-- AvgPool --" << std::endl;
    // AvgPool
    AdjustLevelWithBoot(boot_context, main_ct, pool_level,
                        interface_->GetEvkMap());
    main_ct.SetNumSlots(half_degree);
    boot_context->Trace(main_ct, pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    boot_context->Trace(main_ct, x_width * pool_pack, pool_input_width, main_ct,
                        interface_->GetEvkMap());
    avg_pool.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Trace(main_ct, pool_channel, half_degree / pool_channel,
                        main_ct, interface_->GetEvkMap());
    main_ct.SetNumSlots(fc_input_width);

    std::cout << "-- FC --" << std::endl;
    // FC
    AdjustLevelWithBoot(boot_context, main_ct, fc_level,
                        interface_->GetEvkMap());
    fc.Evaluate(context_, main_ct, main_ct, interface_->GetEvkMap());
    boot_context->Add(main_ct, main_ct, fc_bias);

    main_ct.SetNumSlots(half_degree);
    __ProfileEnd("ResNet20");
    DecryptAndDecode(output_vec, main_ct);
    for (int j = 0; j < 10; j++) {
      output(j, i) = output_vec[j].real();
    }
  }
  long double acc = compute_accuracy(output, test_labels);
  std::cout << "Accuracy: " << acc << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("resnetparam_30.json", "resnetparam_35.json",
                    "resnetparam_40.json", "resnetparam_48.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });