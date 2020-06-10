#!/bin/bash

cd /workspace/onnxruntime

bash build.sh --enable_training --use_cuda --config=Release --build_wheel --update --build --parallel --skip_tests
echo y | pip uninstall onnxruntime-gpu
pip install build/Linux/Release/dist/onnxruntime_gpu-1.3.0-cp37-cp37m-linux_x86_64.whl

