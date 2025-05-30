# 贡献指南

## 贡献流程

1. 在本仓库提交 Pull Request 并得到充分验证
2. Pull Request 合入并稳定运行后再同步提交到 [pytorch-fdn/oota][0]

[0]: https://github.com/pytorch-fdn/oota

## 整体流程

触发任务后，会依次执行下列子工作流：

1. 编译 torch

   - 只有监听到上游 PyTorch 仓库 PR 事件时，才会基于上游 PR 的代码源码编译 torch
   - 否则会安装 torch_npu [requirements.txt][1] 中指定的 torch 版本

2. 编译 torch_npu

   - 核心是执行 `bash torch_npu/ci/build.sh` 进行编译

3. 执行 torch_npu 单元测试

   - 核心是执行 `python torch_npu/ci/access_control_test.py` 进行测试

4. 执行 torch_npu [torchbenchmark][2]

   - 核心是执行 `python benchmark/run_benchmark.py test_bench` 进行测试
   - 周期触发时会自动将 TorchBenchmark 的测试结果提交 PR，例如：[#46][3]

5. 执行其他 PyTorch 生态项目测试（如 [torchtitan][4], [torchtune][5], [torchchat][6]）

[1]: https://github.com/Ascend/pytorch/blob/master/requirements.txt
[2]: https://github.com/pytorch/benchmark
[3]: https://github.com/cosdt/pytorch-integration-tests/pull/46
[4]: https://github.com/pytorch/torchtitan
[5]: https://github.com/pytorch/torchtune
[6]: https://github.com/pytorch/torchchat

## 工作流触发条件

torch_npu 测试总入口工作流文件在 [.github/workflows/ascend_npu_test.yml](.github/workflows/ascend_npu_test.yml)，
其触发条件为：

1. `pull_request` 触发
2. `workflow_dispatch` 手动触发
3. `schedule` 周期触发
4. `pytorch-pr-event-redispatch` 事件触发

其区别如下：

|          事件类型           | 工作流所在分支 |                   触发时机                   | 是否源码编译 torch |
| :-------------------------: | :------------: | :------------------------------------------: | :----------------: |
|        pull_request         |   pr-branch    |                  提交 PR 时                  |         否         |
|      workflow_dispatch      |      main      |                   手动触发                   |         否         |
|          schedule           |      main      |              每晚定期触发 1 次               |         否         |
| pytorch-pr-event-redispatch |      main      | 每晚定期扫描 PyTorch 仓库的 PR，会触发此事件 |         是         |

## 代码结构

```
.
├── ascend_npu                                  // Ascend NPU 配置/文档等
├── .ci                                         // CI 配置/文档等
├── .github
│   ├── actions                                 // 自定义 action
│   └── workflows                               // 工作流文件
│       ├── ascend_npu_test.yml                 // torch_npu 测试总入口
│       ├── _ascend_npu_build_torch.yml         // torch 编译子工作流
│       ├── _ascend_npu_build_torch_npu.yml     // torch_npu 编译子工作流
│       ├── _ascend_npu_ut.yml                  // torch_npu ut 子工作流
│       ├── _ascend_npu_benchmark.yml           // torch_npu benchmark 子工作流
│       ├── dispatch-event.yml                  // 监听上游 PR 事件并分发
│       └── redispatch-event.yml                // 重新分发 PR 事件至其他仓库
├── requirements.txt                            // 本项目依赖的 python 包
├── src
│   └── benchmark                               // TorchBenchmark 流程代码
└── test                                        // 测试代码
```
