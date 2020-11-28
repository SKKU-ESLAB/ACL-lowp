/* Copyright (c) 2019-2020 SKKU ESLAB, and contributors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include "arm_compute/graph.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

using namespace arm_compute;
using namespace std;
using namespace utils;
using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock; 

int main(int argc, char** argv) {
  arm_compute::Tensor a1 { }, b1 { }, c1 { }, a2 { };

  int M = 1024;
  int K = 1024;
  int N = 1024;
  
  int times = 100;

  // Import memory
  float* weight = new float[N*K];
  uint8_t* dataA1 = new uint8_t[M*K];
  uint8_t* dataB1 = new uint8_t[N*K];
  uint32_t* dataC1 = new uint32_t[N*M];

  // Fill matrix value.
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
	  dataA1[i*K+j] = 16;
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
	  dataB1[i*N+j] = 32;
    }
  }

  a1.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::QASYMM8));
  b1.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::QASYMM8));
  c1.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::S32));

  a1.allocator()->import_memory(dataA1, M*K);
  b1.allocator()->import_memory(dataB1, N*K);
  c1.allocator()->import_memory(dataC1, N*M);
  
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm_core;

  arm_compute::GEMMInfo info(false, false, true);
  gemm_core.configure(&a1, &b1, nullptr, &c1, info);

  std::cout << "Warm up" << std::endl;
  for (int i = 0; i < times/10; ++i) {
    gemm_core.run();
  }

  std::cout << "Call gemm" << std::endl;

  auto tbegin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < times; ++i) {
    gemm_core.run();
  }
  auto tend = std::chrono::high_resolution_clock::now();
  double cost = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tbegin).count();
  std::cout <<"Elapsed time : " << cost/times <<" ns"<<std::endl;

  return 0;
}
