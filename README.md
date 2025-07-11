# Artifact Evaluation
This repository contains the artifact for reproducing the experiments in our ASPLOS 2026 paper.
The artifact provides a Docker environment with scripts to reproduce the main experimental results.

## Evaluation Components
The scripts generate results for four key evaluations:
1. **FHE Mechanism Timing**: Evaluation of core FHE operations including HMult, HAdd, HRot, and Rescale
2. **Workload Execution Time**: Performance measurement of Bts, HELR, ResNet, and Sorting workloads
3. **Kernel Fusion Sensitivity Study**: Analysis of kernel fusion effects on the core workloads
4. **Accuracy/Precision Evaluation**: Correctness assessment as Δ varies, measuring accuracy for ResNet (evaluated on 1,000 images from the CIFAR-10 test set) and HELR workloads, and bit precision for Sorting.


## Expected Runtime
The following execution times are based on an NVIDIA A100 80GB PCIe.
- FHE mechanism evaluation: ≤ 3 minutes
- Workload execution time: ≤ 1 hour
- Kernel fusion sensitivity study: ≤ 1 hour
- Correctness evaluation (varying Δ): ≤ 3 hour (on 1,000 images; approx. 15 hours for 10,000 images)

## Hardware Requirements
- NVIDIA GPU (server or consumer grade) with Pascal architecture or later
- At least 16 GB of DRAM
- For exact paper reproduction: NVIDIA RTX 4090, A100 80GB PCIe, or H100 80GB PCIe

## Software Requirements
- CUDA Toolkit (≥ 11.8)
- Docker and nvidia-container-toolkit
- C++ compiler supporting C++17
- CMake (version ≥ 3.24)

## Quick Start
### Installation
**Note:** The following commands require an NVIDIA GPU and the NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) to be installed on your system.
```
# Clone the repository
git clone https://github.com/scale-snu/cheddar-ae.git
cd cheddar-ae

# Build the Docker image
sudo docker build -t cheddar-test .

# Run the Docker container
sudo docker run --rm -it --gpus=all cheddar-test
```

### Running Experiments
Once inside the container, you can run each experiment using the following commands:
```
# Run experiment for FHE mechanism evaluation
python3 Experiment1.py

# Run experiment for workload execution time
python3 Experiment2.py

# Run experiment for sensitivity study
python3 Experiment3.py

# Run experiment for correctness evaluation
python3 Experiment4.py
```

Optionally, to evaluate ResNet on the entire 10,000 images of the CIFAR-10 test set, run the command below:
```
# Evaluate ResNet on the 10,000 CIFAR -10 test images
python3 Experiment4-1.py
```

## Expected Output
After running each experiment script:
- `.csv` files containing the numerical results will be automatically generated
- The sensitivity study will produce `.png` files with performance visualizations
- All results correspond to the graphs and tables presented in the paper

## Contact
* Jongmin Kim (firstname.lastname@snu.ac.kr)
* Wonseok Choi (firstname.lastname@snu.ac.kr)

## License and Citing
See the [License](./LICENSE).
Cheddar (all the files in this repository) is licensed under the MIT License.

Cheddar dynamically links the following third-party libraries:
* NVIDIA CUDA Runtime library (cudart), which is provided under the NVIDIA CUDA Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/index.html
* RMM (licensed under the Apache 2.0 License): https://github.com/rapidsai/rmm
* libtommath (public domain software): https://github.com/libtom/libtommath
* GoogleTest (licensed under the BSD 3-Clause License): https://github.com/google/googletest
* JsonCpp (licensed under the MIT License / public domain software): https://github.com/open-source-parsers/jsoncpp

In addition, the following library is used to implement the workloads:
* cnpy (licensed under the MIT License): https://github.com/rogersce/cnpy
