# Distributed ML Interface

A distributed machine learning inference platform built with **FastAPI, Docker, multiple ML workers, and load-balancing strategies**.

The project investigates how different load-balancing strategies affect the **performance, scalability, and fault tolerance** of a distributed machine learning inference system.

---

## Research Question

> How do different load-balancing strategies affect the scalability, performance, and resource efficiency of a distributed machine learning inference system?

The project currently compares:

* **Round Robin**
* **Least Connections**

under increasing concurrent workloads.

The system is also tested under worker failures to evaluate its fault-tolerance and failover behavior.

---

## Project Goals

The main goals of this project are:

* Build a distributed ML inference system.
* Deploy multiple independent ML workers.
* Implement and compare different load-balancing strategies.
* Measure latency and throughput under increasing workloads.
* Analyze tail latency using P50, P95, and P99.
* Measure request error rates.
* Test worker failure and recovery.
* Implement request retry/failover behavior.
* Containerize the complete system using Docker Compose.
* Produce reproducible benchmark results.
* Analyze the trade-offs between different load-balancing strategies.

---

## System Architecture

```text
                    Client
                      |
                      v
              +---------------+
              |   FastAPI API  |
              |    Gateway     |
              +-------+-------+
                      |
                      v
              +---------------+
              | Load Balancer  |
              +-------+-------+
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     +---------+ +---------+ +---------+
     | Worker 1| | Worker 2| | Worker 3|
     +----+----+ +----+----+ +----+----+
          |           |           |
          +-----------+-----------+
                      |
                      v
               ML Prediction
                      |
                      v
                JSON Response
```

Each worker runs as an independent container and loads the same machine learning model.

The API gateway selects a worker using the configured load-balancing strategy.

---

## Technologies

### Backend

* Python
* FastAPI
* Pydantic
* Requests
* Uvicorn

### Machine Learning

* Scikit-learn
* Pandas
* Joblib

### Distributed System

* Multiple ML workers
* Load balancing
* Health checking
* Retry/failover
* Fault testing

### Infrastructure

* Docker
* Docker Compose
* Linux-based containers

### Testing & Benchmarking

* Pytest
* Python `ThreadPoolExecutor`
* Pandas
* Matplotlib

### Development

* Git
* GitHub

---

## Machine Learning Model

The ML component is intentionally lightweight because the primary focus of this project is the **distributed inference architecture**, rather than model complexity.

The current model predicts a movie rating using:

* `user_id`
* `movie_id`

The model is trained using a regression approach and stored as:

```text
ml/model.joblib
```

A prediction request has the following structure:

```json
{
  "user_id": 1,
  "movie_id": 10
}
```

A worker returns a response similar to:

```json
{
  "user_id": 1,
  "movie_id": 10,
  "predicted_rating": 4.34,
  "worker_id": "worker-1"
}
```

---

## Project Structure

```text
Distributed_ML_Interface/
│
├── benchmark/
│   ├── baseline.py
│   ├── results.csv
│   └── plots/
│       ├── average_latency_ms.png
│       ├── p50_ms.png
│       ├── p95_ms.png
│       ├── p99_ms.png
│       └── throughput_req_per_sec.png
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
├── tests/
│   ├── test_least_connections.py
│   └── ...
│
├── workers/
│   └── worker.py
│
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
├── LEARNING.md
└── README.md
```

---

# Load Balancing

## Round Robin

Round Robin distributes requests sequentially among available workers.

For example:

```text
Request 1 → Worker 1
Request 2 → Worker 2
Request 3 → Worker 3
Request 4 → Worker 1
Request 5 → Worker 2
...
```

The implementation also performs a health check before selecting a worker.

---

## Least Connections

Least Connections tracks the number of active requests assigned to each worker.

The worker with the fewest active connections is selected.

For example:

```text
Worker 1 → 3 active requests
Worker 2 → 1 active request
Worker 3 → 2 active requests

Next request → Worker 2
```

This strategy can be useful when workers have different workloads or when request processing times vary significantly.

---

# Fault Tolerance

The system was also tested under worker failures.

The fault-tolerance workflow is:

```text
Normal Operation
       |
       v
Worker Failure
       |
       v
Health Check
       |
       v
Select Healthy Worker
       |
       v
Retry Request
       |
       v
Successful Response
```

For example, when `worker1` was stopped:

```bash
docker compose stop worker1
```

the API continued serving requests through another healthy worker.

A successful request returned:

```json
{
  "user_id": 1,
  "movie_id": 10,
  "predicted_rating": 4.34,
  "worker_id": "worker-2"
}
```

After restarting the failed worker:

```bash
docker compose up -d worker1
```

the worker became available again.

This demonstrates basic **failover and recovery behavior**.

> The current health mechanism primarily verifies worker reachability. More advanced mechanisms such as circuit breakers, persistent worker state, and automatic health-state management are planned improvements.

---

# Benchmarking

The system was benchmarked using increasing concurrency levels:

```text
1
5
10
20
50
```

Each configuration was executed multiple times.

The benchmark measures:

* Average latency
* P50 latency
* P95 latency
* P99 latency
* Throughput
* Error rate

---

## Benchmark Methodology

The benchmark uses Python's `ThreadPoolExecutor` to generate concurrent HTTP requests.

Each request sends:

```json
{
  "user_id": 1,
  "movie_id": 10
}
```

to:

```text
POST /predict
```

For each workload configuration, the benchmark records the response latency and HTTP status code.

The results are stored in:

```text
benchmark/results.csv
```

---

# Benchmark Results

The distributed inference system was evaluated by comparing:

* **Round Robin**
* **Least Connections**

under increasing concurrency.

| Strategy          | Concurrency | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Error Rate |
| ----------------- | ----------: | ---------------: | -------: | -------: | -----------------: | ---------: |
| Round Robin       |           1 |            15.00 |    17.16 |    20.43 |              66.05 |         0% |
| Round Robin       |           5 |            24.83 |    30.95 |    34.98 |             199.43 |         0% |
| Round Robin       |          10 |            43.02 |    57.78 |    65.22 |             230.82 |         0% |
| Round Robin       |          20 |            79.69 |   110.38 |   125.11 |             248.57 |         0% |
| Round Robin       |          50 |           189.46 |   258.24 |   299.88 |             258.51 |         0% |
| Least Connections |           1 |            15.45 |    17.88 |    24.36 |              64.36 |         0% |
| Least Connections |           5 |            25.06 |    31.90 |    36.91 |             198.14 |         0% |
| Least Connections |          10 |            45.96 |    64.08 |    80.62 |             216.43 |         0% |
| Least Connections |          20 |            83.18 |   117.83 |   148.12 |             238.45 |         0% |
| Least Connections |          50 |           195.40 |   263.96 |   302.64 |             250.82 |         0% |

---

## Results Analysis

Round Robin achieved slightly better performance across the tested workloads.

At concurrency 50:

```text
Round Robin
Throughput:       258.51 req/s
Average latency:  189.46 ms
P95 latency:      258.24 ms
P99 latency:      299.88 ms
```

Compared with:

```text
Least Connections
Throughput:       250.82 req/s
Average latency:  195.40 ms
P95 latency:      263.96 ms
P99 latency:      302.64 ms
```

Both strategies achieved:

```text
0% error rate
```

across all tested configurations.

### Interpretation

The current workers have relatively similar processing characteristics, and the benchmark workload is homogeneous.

Therefore, Least Connections does not gain a significant advantage from dynamically selecting the least busy worker.

Instead, connection tracking and additional health-check operations introduce some overhead.

In this particular workload, **Round Robin provides better throughput and slightly lower latency**.

However, this does not mean Round Robin is universally better.

With:

* heterogeneous workers,
* different worker capacities,
* variable request execution times,
* uneven workloads,

Least Connections may provide better load distribution.

The experiment therefore demonstrates an important distributed-systems principle:

> Load-balancing strategies should be evaluated experimentally under representative workloads rather than assuming that a more dynamic strategy will always provide better performance.

---

# Benchmark Visualization

The benchmark results are visualized using Matplotlib.

Generated plots include:

```text
benchmark/plots/
```

### Average Latency

```text
benchmark/plots/average_latency_ms.png
```

### P50 Latency

```text
benchmark/plots/p50_ms.png
```

### P95 Latency

```text
benchmark/plots/p95_ms.png
```

### P99 Latency

```text
benchmark/plots/p99_ms.png
```

### Throughput

```text
benchmark/plots/throughput_req_per_sec.png
```

These plots make it easier to observe how latency and throughput change as concurrency increases.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Nacmkoohes/Distributed_ML_Interface.git
cd Distributed_ML_Interface
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running with Docker Compose

Build the containers:

```bash
docker compose build
```

Start the system:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

The architecture exposes:

```text
API       → localhost:8000
Worker 1  → localhost:8001
Worker 2  → localhost:8002
Worker 3  → localhost:8003
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

# Selecting the Load-Balancing Strategy

The strategy is configured using the `LOAD_BALANCER` environment variable.

For Round Robin:

```yaml
environment:
  LOAD_BALANCER: round_robin
```

For Least Connections:

```yaml
environment:
  LOAD_BALANCER: least_connections
```

The current Docker Compose configuration uses:

```yaml
LOAD_BALANCER: least_connections
```

---

# Running Tests

Run the complete test suite with:

```bash
python -m pytest -q
```

The tests cover load-balancer behavior including:

* Worker selection
* Connection tracking
* Finishing requests
* Worker recovery
* Health checks
* Ignoring unhealthy workers
* Handling the case where no healthy workers are available

---

# Running Benchmarks

The benchmark script accepts:

```bash
python benchmark/baseline.py <concurrency> <strategy>
```

For example:

```bash
python benchmark/baseline.py 10 round_robin
```

or:

```bash
python benchmark/baseline.py 10 least_connections
```

Example workload:

```bash
python benchmark/baseline.py 50 round_robin
python benchmark/baseline.py 50 least_connections
```

The results are appended to:

```text
benchmark/results.csv
```

---

# Generating Plots

After collecting benchmark results:

```bash
python benchmark/plot_results.py
```

The generated plots are saved under:

```text
benchmark/plots/
```

---

# Learning Documentation

The development and learning process is documented in:

```text
LEARNING.md
```

The learning notes cover topics including:

* FastAPI
* Docker
* Distributed inference
* Load balancing
* Round Robin
* Least Connections
* Benchmarking
* Latency
* P50 / P95 / P99
* Throughput
* Fault tolerance
* Failover
* Automated testing
* Benchmark visualization

---

# Development Progress

The project has been developed incrementally.

### Completed

* [x] FastAPI inference API
* [x] ML prediction service
* [x] Multiple ML workers
* [x] Docker containerization
* [x] Docker Compose orchestration
* [x] Round Robin load balancing
* [x] Least Connections load balancing
* [x] Worker health checking
* [x] Request retry/failover
* [x] Worker failure testing
* [x] Worker recovery testing
* [x] Automated load-balancer tests
* [x] Benchmark automation
* [x] Benchmark result collection
* [x] P50/P95/P99 latency analysis
* [x] Throughput analysis
* [x] Benchmark visualization
* [x] Benchmark comparison

### Planned

* [ ] Prometheus metrics
* [ ] Grafana monitoring dashboard
* [ ] CPU and memory utilization analysis
* [ ] More realistic heterogeneous workloads
* [ ] Improved worker health-state management
* [ ] Circuit breaker
* [ ] More robust failure detection
* [ ] Locust-based load testing
* [ ] More advanced scalability experiments
* [ ] Additional load-balancing strategies

---

# Limitations

The current implementation has several limitations.

### Homogeneous Workers

All workers currently run the same model with similar computational characteristics.

This limits the advantage that Least Connections might provide in heterogeneous environments.

### Simple Health Checking

Worker health is currently based primarily on HTTP reachability.

A production system would require richer health signals and state management.

### In-Memory Connection Tracking

Least Connections maintains active connection counts in memory inside the API process.

This is sufficient for the current experimental setup but would require a more robust design for horizontally scaled API gateways.

### Synthetic Benchmark Workload

The benchmark uses a fixed prediction request and controlled concurrency.

Real-world inference workloads would contain different request patterns and execution times.

---

# Future Work

The next stages of the project focus on improving observability, realism, and fault tolerance.

## Monitoring

Prometheus and Grafana will be introduced to monitor:

* Request rate
* Latency
* Error rate
* Worker utilization
* CPU usage
* Memory usage
* Worker health

## More Realistic Workloads

Future experiments will introduce:

* Different request processing times
* Uneven worker capacities
* Different workload distributions
* Higher concurrency
* Longer-running inference requests

This will allow a more meaningful comparison between Round Robin and Least Connections.

## Fault Tolerance

Future fault-tolerance improvements include:

* Circuit breaker
* Better failure detection
* Worker state management
* Recovery-time measurement
* More controlled mid-request failure experiments

## Advanced Load Testing

Locust will eventually be used for more realistic distributed load testing.

---

# Research Focus

The project is primarily a **distributed systems and machine learning infrastructure experiment**, rather than a model-development project.

The main research dimensions are:

```text
Load Balancing
      |
      +---- Performance
      |
      +---- Scalability
      |
      +---- Fault Tolerance
      |
      +---- Resource Efficiency
      |
      +---- Reliability
```

The machine learning model serves as the inference workload, while the main engineering focus is the distributed system surrounding it.

---

# Author

**Nasim Koohestani**

Computer Engineering Graduate
Amirkabir University of Technology

GitHub:

```text
https://github.com/Nacmkoohes
```

---

# License

This project is intended for educational and research purposes.
