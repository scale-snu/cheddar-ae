#pragma once
#include "Export.h"
#include "core/Context.h"
#include "extension/BootContext.h"
#include "extension/EvalPoly.h"

namespace cheddar {
namespace example_ops {

/**
 * @brief AppReLU operation for ResNet20.
 * https://proceedings.mlr.press/v162/lee22e.html
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT EvalReLU {
 public:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;
  std::shared_ptr<BootContext<word>> context_;

  std::vector<EvalPoly<word>> relu_list_;
  Constant<word> final_half_;

  int input_level_;
  int max_level_;
  int relu1_level_;
  int relu2_level_;
  int relu3_level_;
  double input_scale_;

  explicit EvalReLU(std::shared_ptr<BootContext<word>> context, int max_level,
                    int relu1_level, int relu2_level, int relu3_level);

  void Evaluate(Ct &res, Ct &ct, const EvkMap<word> &evk_map);
};

/**
 * @brief ConvBN operation for ResNet20.
 * https://proceedings.mlr.press/v162/lee22e.html
 *
 * @tparam word
 */
template <typename word>
class API_EXPORT ConvBN {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  std::shared_ptr<BootContext<word>> context_;

  int input_width_;
  int input_channel_;
  int input_pack_;
  int padded_input_size_;
  int input_repeat_;

  bool narrowing_;

  int output_width_;
  int output_channel_;
  int output_pack_;
  int padded_output_size_;
  int output_repeat_;


  std::vector<HoistHandler<word>> conv_weight_;
  std::vector<HoistHandler<word>> conv_mask_;
  Pt bias_;

  ConvBN(std::shared_ptr<BootContext<word>> context, int input_width,
         int input_channel, int input_pack, int output_channel,
         double relu_range, bool narrowing, float *weight_data, float *bias_data,
         bool first = false);

  void Evaluate(Ct &res, Ct &ct, const EvkMap<word> &evk_map);

  void AddRequiredRotations(EvkRequest &rotations);
};

/**
 * @brief DownSample operation for ResNet20.
 * https://proceedings.mlr.press/v162/lee22e.html
 *
 * @tparam word
 */
template <typename word>
class API_EXPORT DownSample {
 public:
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  std::shared_ptr<BootContext<word>> context_;

  int input_width_;
  int input_channel_;
  int input_pack_;
  int padded_input_size_;
  int input_repeat_;

  int output_width_;
  int output_channel_;
  int output_pack_;
  int padded_output_size_;
  int output_repeat_;


  std::vector<Pt> conv_weight_;
  std::vector<HoistHandler<word>> conv_mask_;
  Pt bias_;

  DownSample(std::shared_ptr<BootContext<word>> context, int input_width,
             int input_channel, int input_pack, int output_channel,
             double relu_range, float *weight_data,
             float *bias_data);

  void Evaluate(Ct &res, Ct &ct, const EvkMap<word> &evk_map);

  void AddRequiredRotations(EvkRequest &rotations);
};

/**
 * @brief Sigmoid function for HELR.
 * https://doi.org/10.1609/aaai.v33i01.33019466
 * 
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Sigmoid {
 public:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;
  std::shared_ptr<BootContext<word>> context_;

  std::vector<Constant<word>> c1_div_c3_;
  std::vector<Constant<word>> gamma_times_c3_;
  std::vector<Constant<word>> gamma_times_c0_;
  int f_;

  static constexpr double c0 = 0.5;
  static constexpr double c1 = -0.0843;
  static constexpr double c3 = 0.0002;

  explicit Sigmoid(std::shared_ptr<BootContext<word>> context, int f,
                   int num_iter_per_boot, double gamma);

  void Evaluate(Ct &grad, const std::vector<Ct> &z_data,
                const std::vector<Ct> &inner_prod, const EvkMap<word> &evk_map);
};

/**
 * @brief Tanh function for RNN.
 * https://ceur-ws.org/Vol-2573/PrivateNLP_Paper3.pdf
 * 
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Tanh {
 public:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;
  using Const = Constant<word>;
  std::shared_ptr<BootContext<word>> context_;

  // Preparing tanh poly
  static constexpr double c3 = -0.00163574303018748;
  static constexpr double c1_div_c3 = 0.24947636562803 / c3;

  std::vector<Const> encoded_c1_;
  std::vector<Const> encoded_c3_;

  explicit Tanh(std::shared_ptr<BootContext<word>> context,
                int num_iter_per_boot);

  void Evaluate(Ct &res, int level_index, const Evk &mult_key);
};

/**
 * @brief Compare function for Sorting Network.
 * https://doi.org/10.1109/TIFS.2021.3106167
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class API_EXPORT Compare {
 public:
  using Ct = Ciphertext<word>;
  using Evk = EvaluationKey<word>;
  std::shared_ptr<BootContext<word>> context_;

  std::map<int, EvalPoly<word>> f_list_;
  std::map<int, EvalPoly<word>> g_list_;
  std::map<int, EvalPoly<word>> f_list_last_;

  int max_level_;
  int num_boot_per_iter_;

  int g_iter_;
  int f_iter_;

  explicit Compare(std::shared_ptr<BootContext<word>> context, int max_level,
                   int num_boot_per_iter, int g_iter, int f_iter);
  void Evaluate(Ct &ct_comp, Ct &input, const EvkMap<word> &evk_map,
                int comp_dist, int &input_level,
                std::vector<Constant<word>> &constant_one,
                std::vector<Constant<word>> &constant_half);
};


// Inner product operation for HELR.
template <typename word>
API_EXPORT void InnerProduct(std::shared_ptr<BootContext<word>> context,
                             std::vector<Ciphertext<word>> &res,
                             const std::vector<Ciphertext<word>> &z_data,
                             const Ciphertext<word> &v_data,
                             const Plaintext<word> &y_mask,
                             const EvkMap<word> &evk_map, int f);

}  // namespace example_ops
}  // namespace cheddar