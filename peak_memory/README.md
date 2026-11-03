# 测量 Chromium 运行 SD1.5 的 Peak Memory

测试页面:
`https://10.239.115.25:8080/demos/stable-diffusion-1.5/?unetbatch=2&safetychecker=false&ort=latest`

脚本:`chrome-peak-memory.ps1`(本目录)

## 原理

Chrome 是多进程架构,子进程命令行里带有 `--type=` 标记:

| 标记 | 进程类型 |
|---|---|
| `--type=gpu-process` | GPU 进程 |
| `--type=renderer` | 标签页(渲染)进程 |
| `--type=utility` | 网络、存储等工具进程 |
| 无 | 浏览器主进程 |

脚本用 `Get-CimInstance Win32_Process` 取所有 `chrome.exe` 的命令行并分类,再用 `Get-Process` 读取每个进程的 `PeakWorkingSet64`(自进程启动以来的最大 working set)和 `WorkingSet64`(当前值)。

启用 `WebNNLiteRTGpuInRenderer` 时推理可能跑在 renderer 里,所以 GPU 进程和 renderer 都要看。

## 使用步骤

1. 启动 Chromium(PowerShell):

   ```powershell
   cd C:\Users\webnn\workspace\chromium\src
   .\out\Release\chrome.exe --no-sandbox --enable-features=WebMachineLearningNeuralNetwork,WebNNLiteRTGpuInRenderer --disable-features=WebNNDirectML,WebNNOnnxRuntime --enable-unsafe-webgpu
   ```

2. 在浏览器里打开上面的测试页面,点击 **Load Models**,加载完成后点击 **Generate Image**,等图片生成完成。
3. **不要关闭浏览器**(关闭后峰值读不到)。另开一个 PowerShell 窗口:

   ```powershell
   cd C:\Users\webnn\workspace\webnn\peak_memory
   powershell -ExecutionPolicy Bypass -File .\chrome-peak-memory.ps1
   ```

   保存输出:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\chrome-peak-memory.ps1 | Tee-Object peak-memory.txt
   ```

4. 每次复测前先关闭 Chromium 再重新启动,因为 `PeakWorkingSet64` 会累积,上一轮的峰值会留在进程里。

### 参数

- `-PathPattern`:只统计 `ExecutablePath` 匹配该通配符的进程,默认 `*chromium\src\out\Release*`。
  如果输出为空,脚本会列出实际的 chrome.exe 路径,按需调整,例如:

  ```powershell
  .\chrome-peak-memory.ps1 -PathPattern '*你的路径*'
  ```

## 输出说明

输出一张按 PeakMB 降序排列的进程表,以及汇总行:

| 汇总行 | 含义 |
|---|---|
| GPU process peak (sum) | GPU 进程峰值 |
| Renderer peak (largest) | 最大 renderer 的峰值,通常就是 SD 页面所在进程 |
| Renderer peak (all, sum) | 所有 renderer 峰值之和 |
| GPU + largest renderer peak | GPU 进程 + 最大 renderer 的峰值之和 |
| All chrome processes peak (sum) | 所有 chrome 进程峰值之和 |

## 如何选取数值

- SD 页面对应 **最大的 renderer** 和 **gpu-process**,其余进程很小,可忽略。
- 两个进程峰值之和是**上界**:两个峰值不一定同时出现(例如 renderer 的高点常出在模型下载、解析、创建 session 阶段,之后会释放)。
- 同时占用的真实峰值介于「结束后当前值之和」(`CurMB`)与「峰值之和」之间。需要准确值时,应定时采样当前 working set 并对合计求最大值。
- 这里统计的是系统内存里的 working set,不含全部显存。Intel 集显使用共享内存,可能有一部分不计入。
- 峰值包含**模型加载阶段**。如果只关心生成图片这一步,需要在点 Generate 之前后分别采样。

## 示例结果

| 进程 | PeakMB | CurMB |
|---|---|---|
| renderer (最大) | 16104.6 | 7673.1 |
| gpu-process | 13278.4 | 11427.3 |
| 峰值之和 | 29383 | |
| 当前值之和 | | 19100.4 |

## 备注

- 工具 `ort-test-master` 的 `--tasks workload` 也可以打开页面并输出这两个进程的峰值(`src/workload.js`,新增了 `--browser-path`)。但 SD 页面没有 `#ortStatus` 元素,也需要手动点击 Load Models 和 Generate Image,所以目前直接用本脚本更方便。
- 用 `curl` 访问页面服务器时要加 `--noproxy '*'`,否则会被代理拦成 403。
