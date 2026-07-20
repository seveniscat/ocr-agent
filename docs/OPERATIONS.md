# 运维手册（Windows · NSSM 部署）

本服务在 Windows 上通过 [NSSM](https://nssm.cc) 注册为系统服务 `ocr-agent`，开机自启、崩溃自动重启，不依赖任何终端窗口。本文档列出日常运维常用的命令与说明。

> ⚠️ **关于 Makefile**：项目根目录的 `Makefile` 是为 **Linux/macOS** 写的（用了 `lsof` / `nohup` / `$(VENV)/bin/`），**Windows 上不可用**。Windows 部署请使用本文档的命令。
>
> ⚠️ 所有 `sc` / `net start|stop` / `nssm` 命令都需要**管理员权限**。请用"管理员身份"打开 cmd 或 PowerShell 后再执行。

---

## 0. 服务信息速查

| 项 | 值 |
|---|---|
| 服务名 | `ocr-agent` |
| 显示名 | `ocr-agent (uvicorn :48763)` |
| 监听地址 | `0.0.0.0:48763`（本机 + 局域网均可访问） |
| 本机访问 | `http://127.0.0.1:48763/` |
| 局域网访问 | `http://10.1.74.31:48763/`（本机 LAN IP） |
| Web UI | 根路径 `/`（浏览器直接打开） |
| 健康检查 | `GET /healthz` → `{"status":"ok"}` |
| 安装目录 | `D:\bzdev\ocr-agent` |
| Python | `D:\bzdev\ocr-agent\.venv\Scripts\python.exe` |
| 日志目录 | `D:\bzdev\ocr-agent\logs\` |

---

## 1. 服务生命周期

### 1.1 查看状态

```cmd
sc query ocr-agent
```
或（PowerShell 更直观）：
```powershell
Get-Service ocr-agent | Format-Table Name,Status,StartType,DisplayName
```

输出 `RUNNING` / `STOPPED` 即正常。`Get-Service` 还能看到 `StartType`（应为 `Automatic`）。

### 1.2 启动服务

```cmd
net start ocr-agent
:: 或
sc start ocr-agent
:: 或
nssm start ocr-agent
```

> 这三种等价。`sc start` 在某些 Windows 版本上不会等待服务完全启动就返回，验证是否真的起来请配合 [§3 健康检查](#3-健康检查)。

### 1.3 停止服务

```cmd
net stop ocr-agent
:: 或
sc stop ocr-agent
:: 或
nssm stop ocr-agent
```

> 优雅停止（uvicorn 收到信号后会处理完在途请求再退出）。停止后端口 48763 释放，所有 HTTP 请求都会失败。

### 1.4 重启服务

NSSM 没有 `restart` 子命令，Windows 服务重启用停止 + 启动两步：
```cmd
net stop ocr-agent && net start ocr-agent
```

> ⚠️ **改完 `.env` 后必须重启服务才生效**（uvicorn 的 `--reload` 只监听 `.py`，不监听 `.env`）。详见 [§5 配置变更](#5-配置变更)。

---

## 2. 崩溃恢复与开机自启

这两项是 NSSM 注册时已经配好的，**无需手动操作**，这里说明机制和验证方法。

### 2.1 开机自启

服务启动类型已设为 `SERVICE_AUTO_START`：
```cmd
sc qc ocr-agent | findstr START_TYPE
```
应输出 `START_TYPE         : 2  AUTO_START`。机器开机/重启后服务会自动拉起。

### 2.2 崩溃自动重启

NSSM 配置了 `AppExit Default Restart` + `AppRestartDelay 5000`：
- uvicorn 进程无论以什么退出码结束，NSSM 都会在 **5 秒后自动重启**。
- 验证配置：
  ```cmd
  nssm get ocr-agent AppExit
  nssm get ocr-agent AppRestartDelay
  ```
  应分别输出 `Restart` 和 `5000`。

> 这意味着：即使 OCR 引擎崩溃、内存溢出、被误杀，服务都会自己回来。**不再需要人工值守**。

---

## 3. 健康检查

### 3.1 最简健康检查（HTTP 200 即健康）

```cmd
curl http://127.0.0.1:48763/healthz
```
正常返回：`{"status":"ok","version":"0.1.0"}`

### 3.2 完整状态检查（服务 + 端口 + 本机 + 局域网）

```cmd
:: 1. 服务状态
sc query ocr-agent | findstr STATE

:: 2. 端口是否在监听
netstat -ano | findstr :48763

:: 3. 本机访问
curl http://127.0.0.1:48763/healthz

:: 4. 局域网 IP 访问
curl http://10.1.74.31:48763/healthz
```

四项全绿即部署完整、可对外服务。

### 3.3 用浏览器验证 Web UI

直接打开 `http://127.0.0.1:48763/`（本机）或 `http://10.1.74.31:48763/`（局域网），应看到"ocr-agent · 图片理解"页面。

---

## 4. 日志

### 4.1 日志位置

| 文件 | 内容 |
|---|---|
| `D:\bzdev\ocr-agent\logs\stdout.log` | 标准输出（通常为空，uvicorn 主要输出到 stderr） |
| `D:\bzdev\ocr-agent\logs\stderr.log` | **主要日志**：uvicorn 启动信息、请求日志、`app.*` 业务日志 |

日志已配置 **10MB 自动轮转**（`AppRotateBytes 10485760`），不会写满磁盘。

### 4.2 实时查看日志（跟踪模式）

```cmd
:: cmd（PowerShell 用 Get-Content -Wait）
tail -f D:\bzdev\ocr-agent\logs\stderr.log
```
PowerShell：
```powershell
Get-Content D:\bzdev\ocr-agent\logs\stderr.log -Wait -Tail 50
```
按 Ctrl+C 退出跟踪。

### 4.3 查看最近若干行

```cmd
:: cmd 自带（无 tail，用 more 或 PowerShell）
powershell -c "Get-Content D:\bzdev\ocr-agent\logs\stderr.log -Tail 50"
```

### 4.4 调整轮转大小

默认 10MB。如想改成 50MB：
```cmd
nssm set ocr-agent AppRotateBytes 52428800
```
改完 `nssm restart ocr-agent` 生效。

---

## 5. 配置变更

### 5.1 配置文件位置

`D:\bzdev\ocr-agent\.env`（已被 `.gitignore` 排除，不会进 git）。

字段含义见 `.env.example`（同目录下，有详细注释）。

### 5.2 修改 `.env` 后让它生效

```cmd
:: 1. 编辑 .env（记事本或任意编辑器）
notepad D:\bzdev\ocr-agent\.env

:: 2. 重启服务（必须，否则用的是旧值）
net stop ocr-agent && net start ocr-agent

:: 3. 验证（健康检查 + 看日志没报错）
curl http://127.0.0.1:48763/healthz
```

> 💡 **VLM 配置的热更新特例**：`POST /config/vlm` 接口支持运行时修改部分 VLM 字段（base_url / model / api_key / understand_enabled / enable_thinking），写 `.env` 后立刻生效，**无需重启**。但直接编辑 `.env` 文件则必须重启。

---

## 6. 防火墙

### 6.1 查看现有规则

```powershell
Get-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)" | Format-List DisplayName,Enabled,Direction,Action
```

### 6.2 临时关闭 / 开启规则（不断开 TCP）

```powershell
:: 临时禁用（局域网将无法访问）
Disable-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)"

:: 重新启用
Enable-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)"
```

### 6.3 收紧来源 IP（只允许特定网段访问）

默认规则放行了**任意来源**。如想只允许某个 IP 段（例如 `10.1.0.0/16`）访问：
```powershell
Remove-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)"
New-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 48763 `
    -RemoteAddress 10.1.0.0/16
```

---

## 7. 故障排查

### 7.1 服务起不来 / 一直崩溃重启

**第一步**：看 stderr 日志最后 50 行
```cmd
powershell -c "Get-Content D:\bzdev\ocr-agent\logs\stderr.log -Tail 50"
```
绝大多数启动失败原因（端口占用、`.env` 字段非法、依赖缺失）这里都有明确报错。

**第二步**：如果日志是空的，说明进程根本没起来。手动跑一次 uvicorn 看报错：
```cmd
cd /d D:\bzdev\ocr-agent
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 48763
```
这样错误会直接打到屏幕上。

### 7.2 端口被占用

服务起不来报 "Address already in use"：
```cmd
netstat -ano | findstr :48763
```
找到占用进程的 PID 后：
```cmd
taskkill /PID <那个PID> /F
```
然后 `net start ocr-agent`。

### 7.3 局域网访问不通（本机 OK，别的机器连不上）

按顺序排查：
1. 服务是否在跑：`sc query ocr-agent`
2. 防火墙规则是否启用：`Get-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)"`
3. 调用方网络是否可达本机：让调用方 `ping 10.1.74.31`
4. 调用方所在网段是否在防火墙白名单内（如果做过 [§6.3 收紧](#63-收紧来源-ip只允许特定网段访问)）

### 7.4 OCR 推理报错（paddlepaddle 版本问题）

如果 `/analyze` 报 `ConvertPirAttribute2RuntimeAttribute not support` 之类错误，是 paddlepaddle 版本不对。Windows CPU 上**验证可用的是 3.2.x**（3.3.x 有 PIR/oneDNN 回归 bug，3.0.0 有 strides bug）。

修复：
```cmd
cd /d D:\bzdev\ocr-agent
.venv\Scripts\python.exe -m pip install "paddlepaddle>=3.2,<3.3"
net stop ocr-agent && net start ocr-agent
```

### 7.5 模型缓存被清空导致启动变慢或失败

PaddleOCR 模型缓存在 `C:\Users\<用户>\.paddlex\official_models\`。如被清空，首次启动会重新下载（需要联网到 HuggingFace/ModelScope/AIStudio/BOS 之一）。若网络访问这些站困难，启动会卡。

---

## 8. 服务卸载 / 重装

### 8.1 卸载服务

```cmd
:: 1. 停止并删除服务
nssm stop ocr-agent
nssm remove ocr-agent confirm

:: 2.（可选）删除防火墙规则
:: Remove-NetFirewallRule -DisplayName "ocr-agent (TCP 48763)"
```

卸载后 `.venv`、代码、`.env`、日志文件都保留，只是不再作为 Windows 服务存在。

### 8.2 重新安装服务

直接跑部署脚本即可（脚本是幂等的，会先删掉同名旧服务）：
```cmd
powershell -ExecutionPolicy Bypass -File D:\bzdev\ocr-agent\scripts\install_service.ps1
```
> 该脚本需要管理员权限（会弹 UAC）。脚本内部已包含"注册 + 配置 + 启动"全套步骤。

---

## 9. 常用命令速查卡

| 任务 | 命令 |
|---|---|
| 查状态 | `sc query ocr-agent` |
| 启动 | `net start ocr-agent` |
| 停止 | `net stop ocr-agent` |
| 重启 | `net stop ocr-agent && net start ocr-agent` |
| 健康检查 | `curl http://127.0.0.1:48763/healthz` |
| 看日志 | `powershell -c "Get-Content D:\bzdev\ocr-agent\logs\stderr.log -Tail 50 -Wait"` |
| 改配置后生效 | 编辑 `.env` → `net stop ocr-agent && net start ocr-agent` |
| 图形化编辑服务配置 | `nssm edit ocr-agent` |
| 查端口占用 | `netstat -ano \| findstr :48763` |
| 杀进程 | `taskkill /PID <PID> /F` |
