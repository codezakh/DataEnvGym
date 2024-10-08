#!/bin/bash

cd src/external/LLaMA-Factory
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train ../../../tests/build_tests/opt_125m_lora_sft.yaml 