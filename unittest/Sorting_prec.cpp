#include <algorithm>

#include "Testbed.h"
#include "extension/ExampleOps.h"
using namespace cheddar::example_ops;

static constexpr int msg_length = 1 << 14;
static constexpr int g_iter = 8;
static constexpr int f_iter = 4;

static constexpr int warm_up = 0;

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

TEST_P(Testbed, Sorting) {
  PrepareBoot(msg_length, false, BootVariant::kImaginaryRemoving);

  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  const int max_level = boot_context->boot_param_.GetEndLevel();
  AssertTrue(((g_iter + f_iter) * 3) % (max_level - 1) == 0,
             "Parameters are not adequate for sorting");
  const int num_boot_per_iter = ((g_iter + f_iter) * 3) / (max_level - 1);

  // Preparation
  Compare compare(boot_context, max_level, num_boot_per_iter, g_iter, f_iter);
  int num_stages = Log2Ceil(msg_length);

  // generating masks and rotation keys
  std::vector<std::vector<Plaintext<word>>> flip_mask(num_stages);
  std::vector<std::vector<Plaintext<word>>> comp_mask(num_stages);
  for (int i = 0; i < num_stages; i++) {
    flip_mask[i].resize(max_level + 1);
    comp_mask[i].resize(max_level + 1);

    int rot_distance = 1 << i;
    std::vector<Complex> flip_msg(msg_length);
    std::vector<Complex> comp_msg(msg_length);
    for (int j = 0; j < msg_length; j++) {
      int k = j >> (i + 1);
      flip_msg[j] = ((k & 1) == 0 ? -1.0 : 1.0);
    }
    for (int j = 0; j < msg_length; j++) {
      int k = j >> i;
      comp_msg[j] = ((k & 1) == 0 ? 1.0 : 0.0);
    }
    for (int j = 0; j <= max_level; j++) {
      Encode(flip_mask[i][j], flip_msg, j);
      Encode(comp_mask[i][j], comp_msg, j);
    }

    interface_->PrepareRotationKey(rot_distance, max_level);
    interface_->PrepareRotationKey(msg_length - rot_distance, max_level);
  }

  // Constant For one
  std::vector<Constant<word>> constant_one(max_level + 1);
  std::vector<Constant<word>> constant_half(max_level + 1);
  for (int i = 0; i <= max_level; i++) {
    context_->encoder_.EncodeConstant(constant_one[i], i,
                                      context_->param_.GetScale(i), 1.0);
    context_->encoder_.EncodeConstant(constant_half[i], i,
                                      context_->param_.GetScale(i), 0.5);
  }

  auto LevelDown = [&](Ciphertext<word> &ct, int target_level) -> void {
    int ct_level = context_->param_.NPToLevel(ct.GetNP());
    AssertTrue(ct_level >= target_level, "LevelDown: Invalid target level");
    Ciphertext<word> tmp;
    while (ct_level > target_level) {
      context_->Mult(tmp, ct, constant_one[ct_level]);
      context_->Rescale(ct, tmp);
      ct_level -= 1;
    }
  };

  int num_iter = 20;
  double max_real_abs_diffs[num_iter];
  std::vector<Complex> msg;
  std::vector<Complex> res;
  for (int iter = 0; iter < num_iter; iter++) {
    GenerateRandomMessage(msg, msg_length, -0.5, 0.5, false);

    Ciphertext<word> input;

    EncodeAndEncrypt(input, msg, max_level);
    int input_level = context_->param_.NPToLevel(input.GetNP());
    for (int i = 0; i < num_stages; i++) {
      for (int j = (1 << i); j >= 1; j = (j >> 1)) {
        Ciphertext<word> tmp;

        if (i != num_stages - 1) {
          // flip
          if (input_level == 0) {
            boot_context->Boot(input, input, interface_->GetEvkMap());
            input_level = max_level;
          }
          context_->Mult(tmp, input, flip_mask[i][input_level]);
          context_->Rescale(input, tmp);
          input_level -= 1;
        }

        // Guarantee that input_level >= 2 && ct_comp_level >= 1
        if (input_level <= 1) {
          LevelDown(input, 0);
          boot_context->Boot(input, input, interface_->GetEvkMap());
          input_level = max_level;
        }

        // Perform comparison
        Ciphertext<word> ct_comp;
        int comp_dist = j;
        int log_comp_dist = Log2Ceil(j);

        context_->HRot(ct_comp, input, interface_->GetRotationKey(comp_dist),
                       comp_dist);
        context_->Mult(ct_comp, ct_comp, comp_mask[log_comp_dist][input_level]);

        compare.Evaluate(ct_comp, input, interface_->GetEvkMap(), comp_dist,
                         input_level, constant_one, constant_half);

        // We do not use ct_comp_level from here

        // ct1: left part masked
        // ct2: right part masked
        Ciphertext<word> ct1, ct2;
        context_->Mult(tmp, input, comp_mask[log_comp_dist][input_level]);
        context_->Rescale(ct1, tmp);
        // rotate input by comp_dist (j)
        context_->HRot(input, input, interface_->GetRotationKey(comp_dist),
                       comp_dist);
        context_->Mult(tmp, input, comp_mask[log_comp_dist][input_level]);
        context_->Rescale(ct2, tmp);

        Ciphertext<word> tmp1, tmp2;

        // tmp2 = (ct1 - ct2) * ct_comp + ct2;
        context_->Sub(tmp2, ct1, ct2);
        context_->Mult(tmp2, tmp2, ct_comp);
        context_->Relinearize(tmp2, tmp2, interface_->GetMultiplicationKey());
        context_->MadUnsafe(tmp2, ct2, constant_one[input_level - 1]);

        // tmp1 = (ct2 - ct1) * ct_comp + ct1 = -tmp2 + (ct1 + ct2);
        context_->Neg(tmp1, tmp2);
        context_->Add(ct1, ct1, ct2);
        context_->MadUnsafe(tmp1, ct1, constant_one[input_level - 1]);

        // i.e., ct_comp == 1 --> tmp1 = ct1, tmp2 = ct2
        //       ct_comp == 0 --> tmp1 = ct2, tmp2 = ct1
        context_->HRotAdd(tmp1, tmp2, tmp1,
                          interface_->GetRotationKey(msg_length - comp_dist),
                          msg_length - comp_dist);
        context_->Rescale(input, tmp1);

        input_level -= 2;
        if (i != num_stages - 1) {
          if (input_level == 0) {
            boot_context->Boot(input, input, interface_->GetEvkMap());
            input_level = max_level;
          }
          // int input_level = context_->param_.NPToLevel(input.GetNP());
          context_->Mult(tmp, input, flip_mask[i][input_level]);
          context_->Rescale(input, tmp);
          input_level -= 1;
        }
      }
    }

    DecryptAndDecode(res, input);

    std::sort(msg.begin(), msg.end(), [](const Complex &a, const Complex &b) {
      return a.real() < b.real();
    });
    double max_real_abs_diff = 0;
    for (int i = 0; i < msg_length; i++) {
      max_real_abs_diff =
          std::max(max_real_abs_diff, std::abs(msg[i].real() - res[i].real()));
    }
    max_real_abs_diffs[iter] = log2(1 / max_real_abs_diff);
  }
  CompareMessages(msg, res, 1e-2);
  // Average of max_real_abs_diffs
  double sum = 0;
  for (int i = 0; i < num_iter; i++) {
    sum += max_real_abs_diffs[i];
  }
  sum /= num_iter;
  std::cout << "Average precision: " << sum << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed,
    testing::Values("sortingparam_30.json", "sortingparam_35.json",
                    "sortingparam_40.json", "sortingparam_48.json"),
    [](const testing::TestParamInfo<Testbed::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });