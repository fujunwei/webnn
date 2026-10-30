# sync_chromium

把本机 `C:\Users\junwei\workspace\chromium` 同步到远程机器
`10.239.182.81` 的 `C:\Users\webnn\workspace\chromium`。

脚本：`sync-to-remote.ps1`（PowerShell，使用 robocopy over SMB，增量复制）。

## 工作原理

- 通过远程的 `Users` 共享（`\\10.239.182.81\Users`，指向 `C:\Users`）连接，
  账户为 `webnn`，运行时会弹出密码框，密码不会写入文件，结束后自动断开连接。
- 用 `robocopy /E /MT:32` 复制，只传变化的文件，重复运行是增量的。
- 默认**不**同步 `src\out` 和 `src\out_*`（编译产物很大）。
- 默认**不**删除目标端多余文件，加 `-Mirror` 才会。

## 前提

- 本机能访问对方 445 端口（SMB）：`Test-NetConnection 10.239.182.81 -Port 445`
- 对方存在 `Users` 共享（`Get-SmbShare` 可查），`webnn` 对其有读写权限。
- 在 PowerShell 里运行（不是 cmd）。

## 用法

```powershell
cd C:\Users\junwei\workspace\webnn\sync_chromium

# 1. 先预览，不实际复制
.\sync-to-remote.ps1 -DryRun

# 2. 正式同步整个 chromium 目录
.\sync-to-remote.ps1

# 3. 只同步某个子目录（相对于 chromium 根目录，源端和目标端同时生效）
.\sync-to-remote.ps1 -SubPath src\third_party\litert
.\sync-to-remote.ps1 -SubPath src\services\webnn -DryRun

# 4. 镜像：删除目标端多余的文件（建议先 -DryRun）
.\sync-to-remote.ps1 -SubPath src\third_party\ml-drift -Mirror

# 5. 连同编译产物 src\out* 一起同步
.\sync-to-remote.ps1 -IncludeOut
```

如果执行策略不允许运行脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\sync-to-remote.ps1 -DryRun
```

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-Source` | `C:\Users\junwei\workspace\chromium` | 本机源目录 |
| `-RemoteHost` | `10.239.182.81` | 远程主机 |
| `-RemoteUser` | `webnn` | 远程账户（域账户写成 `域名\用户`） |
| `-ShareName` | `Users` | 远程共享名 |
| `-RemotePath` | `webnn\workspace\chromium` | 目标目录，相对于共享 |
| `-SubPath` | 空 | 只同步该子目录，相对于 `-Source` |
| `-Mirror` | 关 | 删除目标端多余文件（robocopy `/PURGE`） |
| `-IncludeOut` | 关 | 同时同步 `src\out`、`src\out_*` |
| `-DryRun` | 关 | 只列出会做什么，不复制（robocopy `/L`） |

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `System error 67` | 对方没有该共享名。在对方运行 `Get-SmbShare` 确认 `-ShareName`。 |
| `System error 1326` | 用户名或密码错误；域账户用 `域名\webnn`。 |
| `System error 5` | 权限不足。在对方运行 `Grant-SmbShareAccess -Name Users -AccountName webnn -AccessRight Full -Force`，并确认 NTFS 权限允许 `webnn` 写入。 |
| `System error 1219` | 本机已用别的账户连着对方。先 `net use * /delete /y`。 |
| `Invalid Parameter` | robocopy 参数问题；`/XD` 不接受带通配符的完整路径，脚本已自行展开 `out_*`。 |
| robocopy 退出码 | 0–7 为成功（含有文件被复制/额外文件等信息），≥8 表示有失败。 |

## 注意

- 同步期间不要在源端做 git 操作，否则目标端 `src\.git` 可能不一致。
- `-Mirror` 会删除目标端文件，先用 `-DryRun` 确认。
- 只支持目录，不支持单个文件。
- 每次运行都会提示输入密码。
