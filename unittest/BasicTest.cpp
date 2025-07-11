#undef ENABLE_EXTENSION

#include <chrono>

#include "Testbed.h"

static constexpr int warm_up = 5;

std::vector<int> test_levels = {19, 31};

TEST_P(Testbed, CtAddCt) {
  for (int level : test_levels) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + msg2[i]);
    }
    Ciphertext<word> ct1, ct2;

    std::string name = "CtAddCt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct1, ct1, ct2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct1);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, HMult) {
  for (int level : test_levels) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res, ct_tmp;

    std::string name =
        "HMult(tensor + relinearize) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };
    __ProfileStart(name, warm_up, prepare_cts(););
    context_->HMult(ct_tmp, ct1, ct2, interface_->GetMultiplicationKey(),
                    false);
    __ProfileEnd(name);

    context_->Rescale(ct_res, ct_tmp);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, HRot) {
  int num_slots = (1 << log_degree_) / 2;
  word test_rot_dist = 1234;
  interface_->PrepareRotationKey(test_rot_dist, param_->max_level_);

  for (int level : test_levels) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[(i + test_rot_dist) % num_slots]);
    }
    Ciphertext<word> ct1, ct_res;
    std::string name = "HRot at level" + std::to_string(level);
    __ProfileStart(name, warm_up, EncodeAndEncrypt(ct1, msg1, level););
    context_->HRot(ct_res, ct1, interface_->GetRotationKey(test_rot_dist),
                   test_rot_dist);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed, Rescale) {
  for (int level : test_levels) {
    if (level == 0) continue;
    std::cout << "Level: " << level << std::endl;
    std::vector<Complex> msg1;
    GenerateRandomMessage(msg1);
    Plaintext<word> pt1;
    Ciphertext<word> ct1;
    double scale = DetermineScale(level);
    context_->encoder_.Encode(pt1, level, scale * scale, msg1, 0);

    Ciphertext<word> ct_res;
    std::string name = "Rescale at level" + std::to_string(level);
    __ProfileStart(name, warm_up, interface_->Encrypt(ct1, pt1););
    context_->Rescale(ct_res, ct1);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(msg1, res);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed, testing::Values("bootparam_35.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
