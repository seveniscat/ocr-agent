# ocr-agent 运维命令集
#
# 用法:在项目根目录运行 `make <target>`,例如:
#   make run         # 一键启动(本机)+ 自动打开浏览器 UI
#   make up          # 启动服务(对外监听,后台常驻)
#   make log         # 跟踪日志
#   make test URL=…  # 用一个图片 URL 跑一遍 OCR 验证
#   make help        # 列出所有命令
#
# 可用变量(均有默认值,用 `make TARGET VAR=value` 覆盖):
#   HOST    监听地址          (默认 0.0.0.0,对外开放)
#   PORT    监听端口          (默认 8000)
#   LOG     日志文件          (默认 ocr.log)
#   URL     测试用图片 URL    (make test/understand 用)
#   FILE    测试用本地图片    (make test/understand 用,优先于 URL)
#   ENGINE  OCR 引擎          (make test/annotate 用;paddleocr 默认 | vlm 走 Qwen-VL)

# ---- 配置(可在命令行覆盖)----
HOST        ?= 0.0.0.0
PORT        ?= 8000
LOG         ?= ocr.log
VENV        := .venv
PY          := $(VENV)/bin/python
UVICORN     := $(VENV)/bin/uvicorn
PID_FILE    := .uvicorn.pid
PP          := $(PY) scripts/pp_analyze.py
# OCR 引擎:paddleocr(默认,本地) | vlm(Qwen-VL 视觉定位,需 OCR_VLM_OCR_ENABLED=true)
ENGINE      ?= paddleocr
# 本机 UI 地址(run 命令打开浏览器到这里)
LOCAL_URL   := http://127.0.0.1:$(PORT)
# 内网访问地址(对外监听时其它系统用它)
LAN_IP      := $(shell ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $$1}' 2>/dev/null)
BASE        := http://$(LAN_IP):$(PORT)

# 默认目标
.DEFAULT_GOAL := help

# 标记为"伪目标"(不对应真实文件)
.PHONY: help install venv run up dev-up down restart status health log \
        test understand annotate docs pytest clean

# ===========================================================================
# 安装 / 环境
# ===========================================================================

install: ## 安装依赖(首次部署用)
	$(PY) -m pip install -r requirements.txt

venv: ## 创建虚拟环境
	python3 -m venv $(VENV)
	@echo "虚拟环境已创建。激活:source $(VENV)/bin/activate"

# ===========================================================================
# 服务生命周期
# ===========================================================================

up: ## 启动服务(对外监听 + 后台常驻,关终端不掉)。不含热重载,适合对外服务
	@$(UVICORN) app.main:app --host $(HOST) --port $(PORT) --log-level info \
		> $(LOG) 2>&1 & echo $$! > $(PID_FILE)
	@sleep 2
	@$(MAKE) --no-print-directory status

run: ## 一键启动(本机 127.0.0.1)+ 自动打开浏览器 UI。已在跑则直接打开
	@if lsof -ti tcp:$(PORT) >/dev/null 2>&1; then \
		echo "✓ 端口 $(PORT) 已有服务,直接打开 UI"; \
	else \
		echo "→ 启动服务(本机 127.0.0.1:$(PORT),后台)..."; \
		$(UVICORN) app.main:app --host 127.0.0.1 --port $(PORT) --log-level info \
			> $(LOG) 2>&1 & echo $$! > $(PID_FILE); \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
			if curl -sf --max-time 1 $(LOCAL_URL)/healthz >/dev/null 2>&1; then \
				echo "✓ 服务已就绪(等待 $${i}s)"; break; \
			fi; \
			sleep 1; \
		done; \
		curl -sf --max-time 1 $(LOCAL_URL)/healthz >/dev/null 2>&1 \
			|| { echo "✗ 服务未就绪,查看日志:make log"; exit 1; }; \
	fi
	@echo "→ 打开 $(LOCAL_URL)"
	@open "$(LOCAL_URL)" 2>/dev/null || xdg-open "$(LOCAL_URL)" 2>/dev/null || true

dev-up: ## 启动开发服务(--reload 热重载,改 app/ 自动重启)。reload 期间请求会短暂失败
	@$(UVICORN) app.main:app --host $(HOST) --port $(PORT) \
		--reload --reload-dir app --log-level info
	@echo "(前台运行,Ctrl+C 退出)"

down: ## 停止服务
	@if [ -f $(PID_FILE) ]; then \
		kill $$(cat $(PID_FILE)) 2>/dev/null; \
		rm -f $(PID_FILE); \
	fi
	@# Kill ALL processes on the port (--reload spawns a parent+child tree;
	@# killing only the child makes the parent respawn it). Loop until the port
	@# is free, in case the parent takes a moment to release.
	@for i in 1 2 3 4 5; do \
		PIDS=$$(lsof -ti tcp:$(PORT) 2>/dev/null); \
		if [ -z "$$PIDS" ]; then echo "端口 $(PORT) 已无进程 ✓"; break; fi; \
		echo "$$PIDS" | xargs kill 2>/dev/null; \
		sleep 1; \
	done
	@PIDS=$$(lsof -ti tcp:$(PORT) 2>/dev/null); \
	if [ -n "$$PIDS" ]; then echo "仍有残留,强制结束..."; echo "$$PIDS" | xargs kill -9 2>/dev/null; fi

restart: down up ## 重启服务(down + up)

status: ## 查看服务状态(进程 + 监听端口)
	@echo "=== 进程 ==="
	@if [ -f $(PID_FILE) ]; then \
		ps -o pid,command -p $$(cat $(PID_FILE)) 2>/dev/null || echo "$(PID_FILE) 记录的进程已不在"; \
	else echo "无 $(PID_FILE)(可能是 dev-up 前台运行,或未启动)"; fi
	@echo "=== 监听端口 $(PORT) ==="
	@lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2>/dev/null || echo "$(PORT) 端口无监听"

health: ## 健康检查(本机 + 内网 IP)
	@LOCAL=$$(curl -s --max-time 5 http://127.0.0.1:$(PORT)/healthz || printf '无响应 ✗'); \
	printf "  本机:    %s\n" "$$LOCAL"; \
	if [ -n "$(LAN_IP)" ]; then \
		LAN=$$(curl -s --max-time 5 http://$(LAN_IP):$(PORT)/healthz || printf '无响应 ✗'); \
		printf "  内网 IP: %s\n" "$$LAN"; \
	fi

# ===========================================================================
# 日志 / 文档
# ===========================================================================

log: ## 实时跟踪日志(Ctrl+C 退出)
	@tail -f $(LOG)

docs: ## 在浏览器打开交互式 API 文档(/docs)
	@echo "打开 $(BASE)/docs"
	@open "$(BASE)/docs" 2>/dev/null || xdg-open "$(BASE)/docs" 2>/dev/null || true

# ===========================================================================
# 测试 / 验证
# ===========================================================================

test: ## 跑一遍 OCR 验证。用法: make test URL=https://... [ENGINE=vlm] 或 make test FILE=本地图.png
	@if [ -z "$(URL)" ] && [ -z "$(FILE)" ]; then \
		echo "✗ 请提供图片:make test URL=https://... 或 make test FILE=本地图.png" >&2; exit 1; \
	fi
	@$(eval OPT := $(if $(filter-out paddleocr,$(ENGINE)),-F options='{\"engine\":\"$(ENGINE)\"}',))
	@if [ -n "$(FILE)" ]; then \
		echo "→ 上传文件:$(FILE) [engine=$(ENGINE)]" >&2; \
		curl -s --max-time 120 -F "file=@$(FILE)" $(OPT) http://127.0.0.1:$(PORT)/analyze; \
	else \
		echo "→ 下载 URL:$(URL) [engine=$(ENGINE)]" >&2; \
		curl -s --max-time 120 -F "url=$(URL)" $(OPT) http://127.0.0.1:$(PORT)/analyze; \
	fi | $(PP)

understand: ## AI 理解"这张图是什么"。用法同 test:make understand URL=...
	@if [ -z "$(URL)" ] && [ -z "$(FILE)" ]; then \
		echo "✗ 请提供图片:make understand URL=... 或 FILE=..." >&2; exit 1; \
	fi
	@if [ -n "$(FILE)" ]; then \
		echo "→ 上传文件:$(FILE)" >&2; \
		curl -s --max-time 120 -F "file=@$(FILE)" http://127.0.0.1:$(PORT)/understand; \
	else \
		echo "→ 下载 URL:$(URL)" >&2; \
		curl -s --max-time 120 -F "url=$(URL)" http://127.0.0.1:$(PORT)/understand; \
	fi | $(PY) -m json.tool

annotate: ## 生成带标注框的图。用法:make annotate URL=... OUT=out.png [ENGINE=vlm]
	@if [ -z "$(URL)" ] && [ -z "$(FILE)" ]; then echo "✗ 请提供图片 URL= 或 FILE=" >&2; exit 1; fi
	@$(eval OPT := $(if $(filter-out paddleocr,$(ENGINE)),-F options='{\"engine\":\"$(ENGINE)\"}',))
	@OUT="$(or $(OUT),annotated.png)"; \
	if [ -n "$(FILE)" ]; then \
		echo "→ 上传文件:$(FILE) [engine=$(ENGINE)]" >&2; \
		curl -s --max-time 120 -F "file=@$(FILE)" $(OPT) "http://127.0.0.1:$(PORT)/analyze?annotate=true"; \
	else \
		echo "→ 下载 URL:$(URL) [engine=$(ENGINE)]" >&2; \
		curl -s --max-time 120 -F "url=$(URL)" $(OPT) "http://127.0.0.1:$(PORT)/analyze?annotate=true"; \
	fi | $(PY) -c "import sys,base64,pathlib,json; d=json.load(sys.stdin); p='$$OUT'; pathlib.Path(p).write_bytes(base64.b64decode(d['annotated_image_b64'])); print('已保存', p)"

pytest: ## 跑测试套件
	$(PY) -m pytest -q

# ===========================================================================
# 清理
# ===========================================================================

clean: ## 停服务 + 清理日志/缓存
	@$(MAKE) --no-print-directory down 2>/dev/null || true
	@rm -f $(LOG) $(PID_FILE) annotated.png
	@rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	@echo "已清理日志与缓存"

# ===========================================================================
# help —— 自动从各 target 的 ## 注释生成
# ===========================================================================

help: ## 显示本帮助
	@echo "ocr-agent 运维命令"
	@echo
	@echo "用法: make <target> [VAR=value ...]"
	@echo
	@printf "可用变量:\n  HOST=$(HOST)  PORT=$(PORT)  LOG=$(LOG)\n  URL=图片URL   FILE=本地图片   OUT=输出文件名\n  ENGINE=$(ENGINE)(paddleocr|vlm)  测试/标注时指定 OCR 引擎\n"
	@echo
	@echo "常用命令:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "快速上手:make run                           # 启动 + 打开浏览器 UI"
	@echo "VLM 引擎:make test URL=... ENGINE=vlm       # 命令行验证 Qwen-VL OCR"
