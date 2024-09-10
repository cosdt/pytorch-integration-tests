# PyTorch Integration Tests

Welcome to the `pytorch-integration-tests` repository! This repository is
designed to facilitate the integration testing of different accelerators with
PyTorch. Our primary focus is to ensure seamless integration and compatibility
across various devices by running comprehensive GitHub workflows.

## Accelerator Integration Test Results

| [torch_npu][1]                   |
|----------------------------------|
| [![Ascend NPU Test Suite][2]][3] | 

[1]: https://github.com/ascend/pytorch

[2]: https://github.com/cosdt/pytorch-integration-tests/actions/workflows/ascend_npu_test.yml/badge.svg

[3]: https://github.com/cosdt/pytorch-integration-tests/actions/workflows/ascend_npu_test.yml

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
