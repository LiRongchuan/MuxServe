# 定义变量，方便修改路径
DATASET_SOURCE      := /data/lrc/workspace/dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
WORKLOAD_OUTPUT_FILE := examples/basic/sharedgpt_n3_rate_12_5_3.json
MODEL_YAML          := examples/basic/models.yaml
WORKLOAD_SCRIPT     := muxserve/muxsched/workload_utils.py
MPS_START_SCRIPT	:= scripts/start_mps.sh
MPS_STOP_SCRIPT		:= scripts/stop_mps.sh
MPS_DIR				:= examples/basic/mps
LOG_DIR             := log
VLLM_PROC_DIR       := $(LOG_DIR)/vllm_proc  # 新增vllm_proc目录变量
LOG_FILE            := $(LOG_DIR)/muxserve_test.log

# 默认目标
.DEFAULT_GOAL := help

# 生成工作负载的目标 - 移除了目录依赖，不自动创建目录
.PHONY: gen-workload
gen-workload:
	@echo "生成工作负载文件..."
	python $(WORKLOAD_SCRIPT) \
		--dataset-source $(DATASET_SOURCE) \
		--workload_info_from_yaml True \
		--output-file $(WORKLOAD_OUTPUT_FILE) \
		--model-yaml $(MODEL_YAML)
	@echo "工作负载文件已生成: $(WORKLOAD_OUTPUT_FILE)"

# 启动MPS
.PHONY: start-mps
start-mps:
	@echo "启动MPS服务..."
	sudo bash $(MPS_START_SCRIPT) $(MPS_DIR)
	@echo "MPS已启动"

# 停止MPS
.PHONY: stop-mps
stop-mps:
	@echo "停止MPS服务..."
	sudo bash $(MPS_STOP_SCRIPT) $(MPS_DIR)
	@echo "MPS已停止"

# 运行主程序 - 移除gen-workload依赖，添加vllm_proc目录依赖
.PHONY: run
run: | $(VLLM_PROC_DIR)
	@echo "开始运行程序..."
	python -m muxserve.launch examples/basic/model_config.yaml \
		--nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
		--nproc_per_node=2 \
		--server-port 4145 --flexstore-port 50025 \
		--workload-file $(WORKLOAD_OUTPUT_FILE) \
		2>&1 | tee $(LOG_FILE)
	@echo "程序运行结束，日志已保存至: $(LOG_FILE)"

# 清理生成的输出文件
.PHONY: clean
clean:
	@if [ -f "$(WORKLOAD_OUTPUT_FILE)" ]; then \
		rm -f $(WORKLOAD_OUTPUT_FILE); \
		echo "已删除工作负载文件: $(WORKLOAD_OUTPUT_FILE)"; \
	else \
		echo "工作负载文件不存在: $(WORKLOAD_OUTPUT_FILE)"; \
	fi

# 清理日志文件
.PHONY: clean-logs
clean-logs:
	@if [ -d "$(LOG_DIR)" ]; then \
		rm -rf $(LOG_DIR)/*; \
		echo "已清理日志目录: $(LOG_DIR)"; \
	else \
		echo "日志目录不存在: $(LOG_DIR)"; \
	fi

# 创建必要的目录 - 保留LOG_DIR和工作负载目录的创建能力
$(dir $(WORKLOAD_OUTPUT_FILE)) $(LOG_DIR) $(VLLM_PROC_DIR):
	@mkdir -p $@
	@echo "已创建目录: $@"

# 显示帮助信息
.PHONY: help
help:
	@echo "可用目标:"
	@echo "  make gen-workload    - 运行脚本生成工作负载文件"
	@echo "  make start-mps   	  - 启动MPS服务"
	@echo "  make stop-mps   	  - 停止MPS服务"
	@echo "  make run             - 运行主程序（包含日志记录，运行前会创建log/vllm_proc目录）"
	@echo "  make clean           - 删除生成的工作负载文件"
	@echo "  make clean-logs      - 清理所有日志文件"
	@echo "  make help            - 显示此帮助信息"