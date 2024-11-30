# PyTorch Integration Tests

Welcome to the `pytorch-integration-tests` repository! This repository is
designed to facilitate the integration testing of different accelerators with
PyTorch. Our primary focus is to ensure seamless integration and compatibility
across various devices by running comprehensive GitHub workflows.

## Accelerator Integration Test Results

<!-- Start -->

|                                 | [torch_npu][1] |
|---------------------------------|----------------|
| simple_gpt                      | ❌              |
| detectron2_fasterrcnn_r_50_dc5  | ❌              |
| LearningToPaint                 | ✅              |
| hf_GPT2_large                   | ✅              |
| dcgan                           | ✅              |
| nanogpt                         | ✅              |
| fastNLP_Bert                    | ✅              |
| moondream                       | ❌              |
| mobilenet_v2_quantized_qat      | ❌              |
| functorch_dp_cifar10            | ✅              |
| simple_gpt_tp_manual            | ❌              |
| speech_transformer              | ✅              |
| yolov3                          | ✅              |
| resnet50_quantized_qat          | ❌              |
| sam_fast                        | ❌              |
| alexnet                         | ✅              |
| timm_efficientnet               | ✅              |
| pyhpc_isoneutral_mixing         | ✅              |
| basic_gnn_edgecnn               | ✅              |
| nvidia_deeprecommender          | ❌              |
| opacus_cifar10                  | ✅              |
| dlrm                            | ✅              |
| hf_Bert                         | ✅              |
| hf_T5_generate                  | ✅              |
| resnet50                        | ✅              |
| hf_BigBird                      | ✅              |
| resnext50_32x4d                 | ✅              |
| pyhpc_turbulent_kinetic_energy  | ✅              |
| llama                           | ✅              |
| detectron2_maskrcnn_r_50_c4     | ❌              |
| Super_SloMo                     | ✅              |
| moco                            | ❌              |
| stable_diffusion_unet           | ❌              |
| microbench_unbacked_tolist_sum  | ✅              |
| detectron2_maskrcnn_r_101_c4    | ❌              |
| hf_distil_whisper               | ✅              |
| mnasnet1_0                      | ✅              |
| detectron2_fasterrcnn_r_50_fpn  | ❌              |
| timm_resnest                    | ✅              |
| hf_GPT2                         | ✅              |
| squeezenet1_1                   | ✅              |
| basic_gnn_gin                   | ✅              |
| hf_clip                         | ✅              |
| mobilenet_v2                    | ✅              |
| drq                             | ✅              |
| hf_Roberta_base                 | ✅              |
| detectron2_maskrcnn_r_50_fpn    | ❌              |
| timm_nfnet                      | ✅              |
| timm_vovnet                     | ✅              |
| doctr_det_predictor             | ✅              |
| sam                             | ✅              |
| hf_T5_large                     | ✅              |
| mobilenet_v3_large              | ✅              |
| detectron2_fcos_r_50_fpn        | ❌              |
| soft_actor_critic               | ✅              |
| llava                           | ❌              |
| timm_regnet                     | ✅              |
| functorch_maml_omniglot         | ✅              |
| detectron2_fasterrcnn_r_101_c4  | ❌              |
| hf_DistilBert                   | ✅              |
| tts_angular                     | ✅              |
| detectron2_maskrcnn             | ❌              |
| basic_gnn_sage                  | ✅              |
| tacotron2                       | ❌              |
| detectron2_maskrcnn_r_101_fpn   | ❌              |
| lennard_jones                   | ✅              |
| pytorch_unet                    | ✅              |
| vgg16                           | ✅              |
| BERT_pytorch                    | ✅              |
| timm_efficientdet               | ❌              |
| pyhpc_equation_of_state         | ✅              |
| maml                            | ✅              |
| detectron2_fasterrcnn_r_50_c4   | ❌              |
| resnet152                       | ✅              |
| phlippe_densenet                | ✅              |
| maml_omniglot                   | ✅              |
| phlippe_resnet                  | ✅              |
| pytorch_CycleGAN_and_pix2pix    | ✅              |
| hf_Whisper                      | ✅              |
| hf_T5                           | ✅              |
| densenet121                     | ✅              |
| cm3leon_generate                | ✅              |
| detectron2_fasterrcnn_r_101_fpn | ❌              |
| hf_Bert_large                   | ✅              |
| stable_diffusion_text_encoder   | ❌              |
| hf_Reformer                     | ❌              |
| detectron2_fasterrcnn_r_101_dc5 | ❌              |
| demucs                          | ✅              |
| pytorch_stargan                 | ✅              |
| hf_T5_base                      | ✅              |
| torch_multimodal_clip           | ✅              |
| vision_maskrcnn                 | ❌              |
| timm_vision_transformer_large   | ✅              |
| hf_Bart                         | ✅              |
| shufflenet_v2_x1_0              | ✅              |
| llama_v2_7b_16h                 | ❌              |
| basic_gnn_gcn                   | ✅              |
| resnet18                        | ✅              |
| Background_Matting              | ✅              |
| doctr_reco_predictor            | ✅              |
| timm_vision_transformer         | ✅              |
| hf_Albert                       | ✅              |
| hf_Longformer                   | ✅              |

[1]: https://github.com/ascend/pytorch

[2]: https://github.com/cosdt/pytorch-integration-tests/actions/workflows/ascend_npu_test.yml/badge.svg

[3]: https://github.com/cosdt/pytorch-integration-tests/actions/workflows/ascend_npu_test.yml

<!-- End -->

## Overview

This repository contains workflows and scripts that automate the testing
process for integrating different hardware devices with PyTorch. The tests aim
to validate that PyTorch's device-specific functionalities are working
correctly and efficiently across different platforms.

### Key Features

- **Automated Integration Tests**: Run tests automatically for different devices using GitHub Actions.
- **Cross-Device Compatibility**: Ensure that PyTorch functions correctly on NPUs, GPUs, and other devices.
- **Reusable Workflows**: Leverage modular and reusable workflows to streamline the testing process.

## Usage

### Running Tests

To run the integration tests, the repository leverages GitHub Actions.
You can trigger the tests by pushing code to the repository or by manually
triggering the workflows.

### Customizing Workflows

The workflows are designed to be flexible. You can customize the parameters
such as the target branch, runner, and loop time by modifying the inputs in
the workflow files.

## Contributing

We welcome contributions to enhance the integration testing process. Feel free
to submit issues, pull requests, or suggestions to help us improve the
compatibility and performance of PyTorch on various devices.

### Reporting Issues

If you encounter any issues while using the workflows or integrating a device,
please report them via the [Issues](https://github.com/cosdt/pytorch-integration-tests/issues) tab.

## License

This project is licensed under BSD-3-Clause license. See the [LICENSE](LICENSE)
file for more details.
