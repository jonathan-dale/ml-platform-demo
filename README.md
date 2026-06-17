# ml-platform-demo

End-to-end ML platform portfolio project: local MLflow training runs today, with serving, containers, Kubernetes, monitoring, and CI/CD planned downstream.

Each example under `train/` is a self-contained scikit-learn script that logs parameters, metrics, and models to MLflow. The goal is reproducible, inspectable runs and a codebase that grows into a full MLOps demo—not a collection of one-off notebooks.

## Quick start

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Python 3.14 (see `.python-version`).

```sh
git clone https://github.com/jonathan-dale/ml-platform-demo.git
cd ml-platform-demo
uv sync
```

**Terminal 1 — start the MLflow tracking server:**

```sh
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

**Terminal 2 — run a training example:**

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

uv run python train/wine/train.py
```

Run it again after changing hyperparameters in the script to compare runs in the MLflow UI. Repeat with different `alpha` values once `train/california/train.py` is wired up to MLflow.

| Example | Status | Command |
|---------|--------|---------|
| Wine quality (classification) | Ready | `uv run python train/wine/train.py` |
| California housing (regression) | In progress | `uv run python train/california/train.py` |
| Iris (classification) | In progress | `uv run python train/iris/train.py` |

Without `MLFLOW_TRACKING_URI`, runs are written to `./mlruns/` and can be viewed with `uv run mlflow ui`.

## Project layout

```
ml-platform-demo/
├── train/                  # MLflow training examples (Phase 1)
│   ├── wine/train.py
│   ├── california/train.py
│   └── iris/
│       ├── train.py
│       └── predict.py
├── serve/                  # FastAPI serving (Phase 2, planned)
├── charts/                 # Helm chart (Phase 4, planned)
├── workflows/              # Argo Workflow templates (Phase 6, planned)
├── monitoring/             # Grafana dashboards (Phase 5, planned)
├── .github/                # GitHub Actions (Phase 6, planned)
├── docker-compose.yml      # Local stack (Phase 3, planned)
├── pyproject.toml
└── uv.lock
```

Local MLflow state (`mlflow.db`, `artifacts/`, `mlruns/`) is gitignored.

## Roadmap

Six phases, each building on the last. Every phase should be independently demonstrable.

### Phase 1: Local MLflow + training scripts *(in progress)*

- [x] `train/` layout with per-dataset examples
- [x] Wine example: experiment tracking, params, metrics, model logging
- [ ] Standardize California housing on the wine pattern (Ridge regression, RMSE/R², model registry)
- [ ] Fix and standardize Iris example
- [ ] Register best runs in the Model Registry and transition to Staging

**Outcome:** Comfortable with MLflow as a user—prerequisite for everything downstream.

### Phase 2: FastAPI serving layer

Add `serve/main.py` that loads a model from the registry at startup:

```python
mlflow.pyfunc.load_model("models:/california-housing/Production")
```

Expose `/predict` (JSON in/out) and `/health` (model version metadata). The server should not know how the model was trained—only which registry name/stage to load. That separation is what makes retraining and promotion workable.

### Phase 3: Containerization + Docker Compose

- `docker/Dockerfile.train` and `docker/Dockerfile.serve`
- `docker-compose.yml`: MLflow (Postgres backend + artifact volume), one-shot trainer, FastAPI server

**Outcome:** `docker compose up` gives a self-contained local demo anyone can clone and run.

### Phase 4: Helm chart + Kubernetes

Chart under `charts/ml-serving/`: Deployment, Service, Ingress. Values for image tag, replicas, MLflow tracking URI, model name/stage. Deploy MLflow from a public chart on kind/minikube; make model promotion + rollout restart a clean, separated concern. Optional initContainer to verify the model exists before Ready.

### Phase 5: Prometheus monitoring

Instrument the FastAPI app with `prometheus-client`: request count, prediction latency histogram, and a gauge for loaded model version. Deploy kube-prometheus-stack; add a ServiceMonitor. Export Grafana dashboard JSON to `monitoring/dashboards/`.

### Phase 6: Argo Workflows CI/CD

`workflows/retrain.yaml` DAG: **train** → **evaluate** (compare new RMSE to Production; fail if worse) → **promote** (transition to Production + rollout restart). GitHub Actions builds/pushes images on push to main and submits the workflow.

**End state:** push code → build → train → evaluate → promote if better → serve.

---

Screenshots of the MLflow UI, Grafana dashboard, and a successful Argo run will make this land well as a portfolio piece—the signal is operational fluency across the full ML lifecycle, not research novelty.
