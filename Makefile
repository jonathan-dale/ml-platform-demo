# ==========================================
# Makefile for Local MLflow Server (uv)
# ==========================================

# 1. Configuration Variables
UV           := uv
HOST         := 127.0.0.1
PORT         := 5000
TRACKING_URI := http://$(HOST):$(PORT)
BACKEND_URI  := sqlite:///mlflow.db
ARTIFACT_DIR := ./artifacts

# 2. Phony targets
.PHONY: help install install-dev start stop status clean-data train-wine train-california train-iris

.DEFAULT_GOAL := help

# Default target displays available commands
help:
	@echo "Available commands:"
	@echo "  make install            - Install project dependencies (uv sync)"
	@echo "  make install-dev        - Install dependencies including dev tools (ruff)"
	@echo "  make start              - Start MLflow server in the background"
	@echo "  make stop               - Stop the running MLflow server"
	@echo "  make status             - Check if MLflow server is running"
	@echo "  make clean-data         - Wipe local MLflow database and artifacts"
	@echo "  make train-wine         - Run wine quality training against the local server"
	@echo "  make train-california   - Run California housing training against the local server"
	@echo "  make train-iris         - Run iris training against the local server"

# Install dependencies from pyproject.toml / uv.lock
install:
	$(UV) sync

install-dev:
	$(UV) sync --group dev

# Start the MLflow server in the background
start:
	@echo "Starting MLflow server on $(TRACKING_URI)..."
	@nohup $(UV) run mlflow server \
		--host $(HOST) \
		--port $(PORT) \
		--backend-store-uri $(BACKEND_URI) \
		--default-artifact-root $(ARTIFACT_DIR) > mlflow_server.log 2>&1 &
	@echo "Server started. Logs written to mlflow_server.log"

# Stop the MLflow server
stop:
	@echo "Stopping MLflow server..."
	@pkill -f "mlflow server" || echo "No running MLflow server found."

# Check server status
status:
	@pgrep -f "mlflow server" > /dev/null \
		&& echo "MLflow server is RUNNING at $(TRACKING_URI)." \
		|| echo "MLflow server is STOPPED."

# Permanently delete tracking data
clean-data: stop
	@echo "Deleting database and artifact directories..."
	rm -rf mlflow.db $(ARTIFACT_DIR) mlruns mlflow_server.log
	@echo "Data wiped successfully."

# Training examples (require a running MLflow server)
train-wine:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(UV) run python train/wine/train.py

train-california:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(UV) run python train/california/train.py

train-iris:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(UV) run python train/iris/train.py
