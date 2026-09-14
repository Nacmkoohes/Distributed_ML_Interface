# Distributed ML Interface

A distributed machine-learning inference platform designed to study **load balancing, scalability, fault tolerance, and performance** in a multi-worker inference system.

The project compares **Round Robin** and **Least Connections** load-balancing strategies under increasing workloads and evaluates their impact on latency, throughput, and tail performance.

---

## Research Question

> **How do different load-balancing strategies affect the scalability, performance, and resource efficiency of a distributed machine-learning inference system?**

The system was designed as an experimental platform rather than a production-ready cloud service. The primary focus is on distributed-systems behavior and performance analysis, while the machine-learning model provides a realistic inference workload.

---

## Project Goals

The project investigates:

* How requests should be distributed across multiple ML workers
* How different load-balancing strategies behave under increasing concurrency
* How worker failures affect request processing
* How retry and failover improve system availability
* How latency changes as workload increases
* How average latency differs from tail latency
* How P95 and P99 reveal slow requests
* How to monitor a distributed inference service using Prometheus and Grafana

---

## Architecture

```text
                         ┌───────────────┐
                         │    Client     │
                         └───────┬───────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │    FastAPI API    │
                       │   API Gateway     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Load Balancer   │
                       │                  │
                       │ Round Robin      │
                       │ Least Connections│
                       └────────┬─────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
           ┌──────────┐   ┌──────────┐   ┌──────────┐
           │ Worker 1 │   │ Worker 2 │   │ Worker 3 │
           └────┬─────┘   └────┬─────┘   └────┬─────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                         ┌────────────┐
                         │ ML Model   │
                         └────────────┘


                    Monitoring Pipeline

             FastAPI
                │
             /metrics
                │
                ▼
          ┌─────────────┐
          │ Prometheus  │
          └──────┬──────┘
                 │
                 │ PromQL
                 ▼
          ┌─────────────┐
          │   Grafana   │
          └─────────────┘
```

---

## Technology Stack

| Category            | Technology                   |
| ------------------- | ---------------------------- |
| API                 | FastAPI                      |
| Language            | Python                       |
| ML                  | scikit-learn                 |
| Data Processing     | Pandas                       |
| Model Serialization | Joblib                       |
| Load Balancing      | Custom Python implementation |
| Testing             | Pytest                       |
| Containerization    | Docker                       |
| Orchestration       | Docker Compose               |
| Monitoring          | Prometheus                   |
| Visualization       | Grafana                      |
| Benchmarking        | Custom Python benchmark      |
| Version Control     | Git / GitHub                 |

---

# Machine Learning Component

The project uses a lightweight movie-rating prediction model based on MovieLens-style user/movie interactions.

The ML model is intentionally simple because the main research focus is **distributed inference and system performance**, rather than model accuracy.

The model is loaded once by each worker and reused for inference.

The prediction service receives:

```json
{
  "user_id": 1,
  "movie_id": 10
}
```

and returns a predicted rating together with the worker that processed the request:

```json
{
  "user_id": 1,
  "movie_id": 10,
  "predicted_rating": 4.34,
  "worker_id": "worker-1"
}
```

---

# Distributed Worker Architecture

The system runs three independent ML workers:

```text
worker-1
worker-2
worker-3
```

Each worker:

* Runs a FastAPI application
* Loads the ML model
* Provides a `/predict` endpoint
* Provides a `/health` endpoint
* Identifies itself using `WORKER_ID`

Example:

```text
POST /predict
```

Response:

```json
{
  "predicted_rating": 4.34,
  "worker_id": "worker-2"
}
```

The worker ID makes request distribution observable during testing.

---

# Load Balancing

Two load-balancing strategies were implemented and evaluated.

## Round Robin

Round Robin distributes requests sequentially across workers.

For three workers, the sequence is approximately:

```text
worker-1
    ↓
worker-2
    ↓
worker-3
    ↓
worker-1
    ↓
...
```

Before selecting a worker, the load balancer checks worker availability through the `/health` endpoint.

### Advantages

* Simple
* Low scheduling overhead
* Predictable distribution
* Effective for homogeneous workers

### Limitation

Round Robin does not consider the current workload or number of active requests on each worker.

---

## Least Connections

Least Connections selects the worker with the lowest number of active requests.

For example:

```text
worker-1 → 3 connections
worker-2 → 1 connection
worker-3 → 4 connections
```

The next request is sent to:

```text
worker-2
```

The implementation maintains an in-memory connection counter and updates it when requests start and finish.

### Advantages

* Can adapt to uneven workloads
* Useful when request durations vary
* Can avoid sending work to already busy workers

### Limitation

The strategy introduces additional bookkeeping and health-check overhead.

For homogeneous workers with similar request durations, it may provide little benefit over Round Robin.

---

# Strategy Selection

The strategy can be selected through an environment variable:

```text
LOAD_BALANCER
```

For example:

```yaml
environment:
  LOAD_BALANCER: least_connections
```

Supported strategies:

```text
round_robin
least_connections
```

---

# Fault Tolerance

Worker failures were explicitly tested rather than assuming that the system would remain available.

The fault-tolerance experiment followed this process:

```text
1. Start all workers
        ↓
2. Stop one worker
        ↓
3. Send inference request
        ↓
4. Detect worker failure
        ↓
5. Retry using another worker
        ↓
6. Successful response
        ↓
7. Restart failed worker
        ↓
8. Worker becomes available again
```

The API contains retry/failover logic around worker requests.

When a worker becomes unavailable, the API can attempt another worker rather than immediately returning an error.

### Important limitation

The current retry mechanism is intentionally lightweight.

It is **not a full production-grade circuit breaker**. In particular, a failed worker may temporarily remain visible to health checks depending on the exact failure timing.

A production implementation could add:

* Failure thresholds
* Temporary worker quarantine
* Circuit breaker states
* Exponential backoff
* More sophisticated health checks

These are considered future improvements rather than part of the current experimental scope.

---

# Benchmarking

The system was benchmarked under increasing concurrency:

```text
1
5
10
20
50
```

For each configuration, the following metrics were collected:

* Average latency
* P50 latency
* P95 latency
* P99 latency
* Throughput
* Error rate

The same workload was used for both load-balancing strategies to make the comparison controlled.

---

# Benchmark Results

## Round Robin

| Concurrency | Avg Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Error Rate |
| ----------: | ---------------: | -------: | -------: | -------: | -----------------: | ---------: |
|           1 |            15.00 |    14.84 |    17.16 |    20.43 |              66.05 |         0% |
|           5 |            24.83 |    24.35 |    30.95 |    34.98 |             199.43 |         0% |
|          10 |            43.02 |    42.33 |    57.78 |    65.22 |             230.82 |         0% |
|          20 |            79.69 |    78.56 |   110.38 |   125.11 |             248.57 |         0% |
|          50 |           189.46 |   189.05 |   258.24 |   299.88 |             258.51 |         0% |

## Least Connections

| Concurrency | Avg Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Error Rate |
| ----------: | ---------------: | -------: | -------: | -------: | -----------------: | ---------: |
|           1 |            15.45 |    15.13 |    17.88 |    24.36 |              64.36 |         0% |
|           5 |            25.06 |    24.67 |    31.90 |    36.91 |             198.14 |         0% |
|          10 |            45.96 |    44.73 |    64.08 |    80.62 |             216.43 |         0% |
|          20 |            83.18 |    81.55 |   117.83 |   148.12 |             238.45 |         0% |
|          50 |           195.40 |   194.73 |   263.96 |   302.64 |             250.82 |         0% |

---

# Results Interpretation

Under the tested homogeneous workload, **Round Robin performed slightly better overall**.

At concurrency 50:

* Round Robin average latency: **189.46 ms**
* Least Connections average latency: **195.40 ms**

Round Robin also achieved higher throughput:

* Round Robin: **258.51 req/s**
* Least Connections: **250.82 req/s**

P95 and P99 latency were also slightly lower for Round Robin.

The most likely explanation is that all workers had approximately similar processing characteristics. Therefore, Least Connections had limited opportunity to improve scheduling based on worker load, while maintaining additional connection bookkeeping and health-check overhead.

### Important conclusion

The result should **not** be interpreted as:

> Round Robin is always better than Least Connections.

The more accurate conclusion is:

> Under the tested homogeneous workload, Round Robin achieved slightly better performance than Least Connections because the workers had similar processing characteristics and Least Connections provided limited scheduling benefit.

Different workloads, especially heterogeneous request durations or uneven worker capacities, could produce different results.

---

# Performance Metrics

## Average Latency

Average latency represents the mean response time.

It provides a useful overall measurement but can hide slow outliers.

---

## P50

P50 is the median latency.

Approximately half of the requests are at or below this value.

---

## P95

P95 indicates the latency below which approximately 95% of requests complete.

It is useful for observing tail behavior without being dominated by the extreme slowest requests.

---

## P99

P99 indicates the latency below which approximately 99% of requests complete.

It provides a stronger view of extreme tail latency.

---

## Throughput

Throughput measures how many requests the system processes per second.

Higher throughput is generally desirable, provided latency and error rate remain acceptable.

---

# Performance Visualization

Benchmark plots are generated for:

```text
benchmark/plots/
├── average_latency_ms.png
├── p50_ms.png
├── p95_ms.png
├── p99_ms.png
└── throughput_req_per_second.png
```

The plots show how each strategy behaves as concurrency increases.

---

# Monitoring

Runtime monitoring was implemented using:

```text
Prometheus
+
Grafana
```

The FastAPI application exposes metrics through:

```text
/metrics
```

using:

```text
prometheus-fastapi-instrumentator
```

Prometheus periodically scrapes the API.

Grafana queries Prometheus using PromQL and visualizes the resulting metrics.

---

# Prometheus Configuration

Current configuration:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "api"
    static_configs:
      - targets: ["api:8000"]
```

Prometheus therefore collects metrics from the API every five seconds.

Prometheus is available at:

```text
http://localhost:9090
```

---

# Grafana

Grafana is available at:

```text
http://localhost:3000
```

The dashboard currently monitors:

* Request Throughput
* P95 Request Latency
* P99 Request Latency

---

# PromQL Examples

## Request Throughput

```promql
sum(
  rate(
    http_requests_total{
      handler="/predict",
      status="2xx"
    }[5m]
  )
)
```

This estimates successful `/predict` requests per second over the last five minutes.

---

## P95 Latency

```promql
histogram_quantile(
  0.95,
  sum(
    rate(
      http_request_duration_highr_seconds_bucket[5m]
    )
  ) by (le)
)
```

This calculates the 95th percentile of the instrumented HTTP request-duration histogram.

---

## P99 Latency

```promql
histogram_quantile(
  0.99,
  sum(
    rate(
      http_request_duration_highr_seconds_bucket[5m]
    )
  ) by (le)
)
```

This calculates the 99th percentile.

---

# Understanding the Histogram

The latency metric:

```text
http_request_duration_highr_seconds_bucket
```

is a histogram bucket metric.

Conceptually, requests are grouped into latency boundaries such as:

```text
≤ 0.01 s
≤ 0.025 s
≤ 0.05 s
≤ 0.1 s
≤ 0.25 s
≤ 0.5 s
≤ 1 s
...
```

The `le` label represents:

```text
less than or equal to
```

The histogram allows Prometheus to estimate quantiles such as P95 and P99.

---

# Docker

The project uses Docker to package the application and its dependencies into reproducible environments.

Dockerfile:

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

---

# Docker Compose

The complete system is orchestrated using Docker Compose.

Services:

```text
api
worker1
worker2
worker3
prometheus
grafana
```

Ports:

| Service    | Port |
| ---------- | ---: |
| API        | 8000 |
| Worker 1   | 8001 |
| Worker 2   | 8002 |
| Worker 3   | 8003 |
| Prometheus | 9090 |
| Grafana    | 3000 |

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/Nacmkoohes/Distributed_ML_Interface.git
cd Distributed_ML_Interface
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run with Docker Compose

Start the complete system:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up -d --build
```

Check services:

```bash
docker compose ps
```

---

# API

## Health Check

```http
GET /health
```

Example:

```bash
curl http://localhost:8000/health
```

---

## Prediction

```http
POST /predict
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"movie_id":10}'
```

Example response:

```json
{
  "user_id": 1,
  "movie_id": 10,
  "predicted_rating": 4.34,
  "worker_id": "worker-1"
}
```

---

## Prometheus Metrics

```http
GET /metrics
```

Example:

```bash
curl http://localhost:8000/metrics
```

---

# Testing

Run the test suite with:

```bash
python -m pytest
```

Using:

```bash
python -m pytest
```

is preferred in this environment because the global `pytest` executable may point to a different Python installation.

---

# Project Structure

```text
Distributed_ML_Interface/
│
├── benchmark/
│   ├── results.csv
│   ├── benchmark.py
│   └── plots/
│       ├── average_latency_ms.png
│       ├── p50_ms.png
│       ├── p95_ms.png
│       ├── p99_ms.png
│       └── throughput_req_per_second.png
│
├── load_balancer/
│   ├── round_robin.py
│   └── least_connections.py
│
├── ml/
│   └── model.joblib
│
├── services/
│   └── prediction.py
│
├── workers/
│   └── worker.py
│
├── tests/
│   └── ...
│
├── main.py
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── README.md
└── READING.md
```

---

# Development Progress

The project was developed incrementally.

### Phase 1 — ML Inference

* ML model preparation
* Prediction service
* Model loading
* Prediction endpoint

### Phase 2 — Distributed Workers

* Worker abstraction
* Multiple worker instances
* Worker IDs
* Health endpoint

### Phase 3 — Load Balancing

* Round Robin
* Least Connections
* Strategy selection

### Phase 4 — Testing

* Worker tests
* Load balancer tests
* API tests
* Integration testing

### Phase 5 — Docker

* Dockerfile
* Docker Compose
* Multi-worker deployment
* Container networking

### Phase 6 — Benchmarking

* Concurrency experiments
* Latency measurement
* P50/P95/P99
* Throughput
* Error rate
* CSV results
* Performance plots

### Phase 7 — Fault Tolerance

* Worker failure injection
* Health checking
* Retry
* Failover
* Worker recovery

### Phase 8 — Monitoring

* Prometheus instrumentation
* `/metrics`
* Prometheus scraping
* Grafana integration
* Request throughput dashboard
* P95 dashboard
* P99 dashboard

---

# Design Decisions

## Why three workers?

Three workers provide enough parallelism to demonstrate load balancing and worker failure without making the experimental setup unnecessarily complex.

## Why compare Round Robin and Least Connections?

They represent two fundamentally different scheduling approaches:

* Round Robin uses a fixed rotation.
* Least Connections considers current active workload.

This creates a meaningful comparison for a distributed inference service.

## Why keep the ML model simple?

The research focus is system behavior rather than model architecture.

A lightweight model allows the experiments to focus on:

* scheduling
* concurrency
* latency
* throughput
* fault tolerance

rather than GPU/model-training complexity.

## Why Docker Compose instead of Kubernetes?

The project focuses on understanding distributed-system concepts first.

Docker Compose provides:

* isolated services
* reproducible environments
* service discovery
* multi-worker deployment

without introducing Kubernetes complexity before it is necessary.

Kubernetes is considered possible future work.

---

# Limitations

The current system is an experimental platform rather than a production-ready inference service.

Current limitations include:

* Workers are homogeneous.
* The load balancer runs in the API process.
* Least Connections uses in-memory connection counters.
* Retry logic is intentionally simple.
* Health checks are basic HTTP health checks.
* No persistent distributed state exists for load-balancer counters.
* No authentication or authorization layer is implemented.
* No production-grade circuit breaker is implemented.
* Resource monitoring is not yet a complete CPU/memory observability system.
* The benchmark workload is controlled and relatively small.
* Grafana latency monitoring currently uses the available HTTP histogram rather than a dedicated inference-latency metric.

These limitations define the boundary of the current experiment rather than being hidden assumptions.

---

# Future Work

Possible extensions include:

* Kubernetes deployment
* Horizontal Pod Autoscaling
* Heterogeneous worker capacities
* Variable inference workloads
* More realistic workload generation
* Circuit breaker implementation
* Exponential backoff
* Worker quarantine
* Distributed load-balancer state
* CPU and memory monitoring
* OpenTelemetry integration
* Alerting
* Distributed tracing
* Persistent experiment storage

These are intentionally outside the current project scope.

---

# Research Focus

The main contribution of the project is not the ML model itself.

The project focuses on understanding how a distributed inference system behaves when:

```text
Workload increases
        ↓
Concurrency increases
        ↓
Workers become more loaded
        ↓
Latency increases
        ↓
Tail latency becomes important
        ↓
Load-balancing strategy affects performance
```

The experimental results show that under the current homogeneous workload, Round Robin performs slightly better than Least Connections.

However, the results are workload-dependent and should not be generalized to all distributed systems.

---

# Key Takeaways

This project demonstrates practical experience with:

* Distributed systems
* Load balancing
* Fault tolerance
* Failure recovery
* REST APIs
* ML inference
* Docker
* Docker Compose
* Performance benchmarking
* Prometheus
* PromQL
* Grafana
* Latency analysis
* Throughput analysis
* Python testing
* Git/GitHub

The project combines machine learning with backend and distributed-systems engineering, with the primary emphasis on **system performance and reliability**.

---

# Author

**Nasim Koohestani**

Computer Engineering / Computer Science

Amirkabir University of Technology

GitHub: [Nacmkoohes](https://github.com/Nacmkoohes)

