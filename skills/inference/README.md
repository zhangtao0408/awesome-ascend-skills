# Inference

面向推理、模型转换、量化与评测的开发入口目录。

当前 `skills/inference/` 已承载真实 skill 目录；请按下面链接进入对应 skill 开发：

- [`atc-model-converter/`](atc-model-converter/)：ATC 模型转换与 OM 推理
- [`vllm-ascend/`](vllm-ascend/)：vLLM-Ascend 推理服务
- [`msmodelslim/`](msmodelslim/)：量化、压缩与部署适配
- [`ais-bench/`](ais-bench/)：精度与性能评测
- [`diffusers-ascend/`](diffusers-ascend/)：Diffusers 环境、权重与推理
- [`wan-ascend-adaptation/`](wan-ascend-adaptation/)：Wan 系列模型昇腾适配

推荐场景：

- 模型转换与部署
- 量化压缩
- 推理性能评估

对应 bundle：`ascend-inference`
