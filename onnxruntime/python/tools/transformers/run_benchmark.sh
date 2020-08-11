# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This measures the performance of OnnxRuntime, PyTorch and TorchScript on transformer models.
# Please install PyTorch (see https://pytorch.org/) before running this benchmark. Like the following:
# GPU:   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# CPU:   conda install pytorch torchvision cpuonly -c pytorch

# When run_cli=true, this script is self-contained and you need not copy other files to run benchmarks
#                    it will use onnxruntime-tools package.
# If run_cli=false, it depends on other python script (*.py) files in this directory.
run_cli=false

# only need once
run_install=false

# Engines to test.
run_ort=false
run_torch=true
run_torchscript=false

# Devices to test (You can run either CPU or GPU, but not both: gpu need onnxruntime-gpu, and CPU need onnxruntime).
run_gpu_fp32=true
run_gpu_fp16=true
run_cpu_fp32=false
run_cpu_int8=false

average_over=10
# CPU takes longer time to run, only run 100 inferences to get average latency.
if [ "$run_cpu" = true ] ; then
  average_over=1
fi

# Enable optimizer (use script instead of OnnxRuntime for graph optimization)
use_optimizer=true

# Batch Sizes and Sequence Lengths
batch_sizes="1"
sequence_lengths="4"

# Number of inputs (input_ids, token_type_ids, attention_mask) for ONNX model.
# Not that different input count might lead to different performance
# Here we only test one input (input_ids) for fair comparison with PyTorch.
input_counts=1

# Pretrained transformers models can be a subset of: bert-base-cased roberta-base gpt2 distilgpt2 distilbert-base-uncased
models_to_test="ctrl camembert-base t5-base xlm-roberta-base flaubert/flaubert_base_uncased facebook/bart-base DialoGPT-medium reformer-enwik8 allenai/longformer-base-4096"

# If you have mutliple GPUs, you can choose one GPU for test. Here is an example to use the second GPU:
# export CUDA_VISIBLE_DEVICES=1

# This script will generate a logs file with a list of commands used in tests.
echo echo "ort=$run_ort torch=$run_torch torchscript=$run_torchscript gpu_fp32=$run_gpu_fp32 gpu_fp16=$run_gpu_fp16 cpu=$run_cpu optimizer=$use_optimizer batch=$batch_sizes sequence=$sequence_length models=$models_to_test" >> benchmark.log

# Set it to false to skip testing. You can use it to dry run this script with the log file.
run_tests=true

# Directory for downloading pretrained models.
cache_dir="./cache_models"

# Directory for ONNX models
onnx_dir="./onnx_models"

# Use raw attention mask in Attention operator or not.
use_raw_attention_mask=false

# -------------------------------------------
if [ "$run_cpu_fp32" = true ] || [ "$run_cpu_int8" = true ]; then
  if [ "$run_gpu_fp32" = true ] ; then
    echo "cannot test cpu and gpu at same time"
    exit 1
  fi
  if [ "$run_gpu_fp16" = true ] ; then
    echo "cannot test cpu and gpu at same time"
    exit 1
  fi
fi


if [ "$run_install" = true ] ; then
  pip uninstall --yes ort_nightly
  pip uninstall --yes onnxruntime
  pip uninstall --yes onnxruntime-gpu
  if [ "$run_cpu_fp32" = true ] || [ "$run_cpu_int8" = true ]; then
    pip install onnxruntime
  else
    pip install onnxruntime-gpu
  fi
  pip install --upgrade onnxruntime-tools
  pip install --upgrade transformers
fi

if [ "$run_cli" = true ] ; then
  echo "Use onnxruntime_tools.transformers.benchmark"
  benchmark_script="-m onnxruntime_tools.transformers.benchmark"
else
  benchmark_script="benchmark.py"
fi

onnx_export_options="-i $input_counts -v -b 0 --overwrite -f fusion.csv -c $cache_dir --onnx_dir $onnx_dir"
benchmark_options="-b $batch_sizes -s $sequence_lengths -t $average_over -f fusion.csv -r result.csv -d detail.csv -c $cache_dir --onnx_dir $onnx_dir"

if [ "$use_optimizer" = true ] ; then
  onnx_export_options="$onnx_export_options -o"
  benchmark_options="$benc