# Learning Log — Distributed ML Inference Platform

This document records the concepts, technical decisions, implementation steps, and lessons learned while building the Distributed ML Inference Platform.

The goal is not only to build a working application, but to understand how machine learning inference can be deployed, distributed, monitored, tested, and evaluated under different workloads.

---

# Day 1 — FastAPI and API Fundamentals

## Concepts Learned

* Client-server architecture
* REST API
* HTTP methods
* GET vs POST
* Request body
* JSON
* Pydantic models
* Input validation
* FastAPI
* Automatic API documentation
* HTTP status codes

## API Architecture

The first version of the system was a simple ML prediction API:

```text
Client
   ↓
FastAPI
   ↓
Prediction
   ↓
Response
```

The API provides two endpoints:

```text
GET  /health
POST /predict
```

## Health Endpoint

The `/health` endpoint is used to determine whether the API is running.

Example response:

```json
{
    "status": "ok"
}
```

## Prediction Endpoint

The `/predict` endpoint receives:

```json
{
    "user_id": 12,
    "movie_id": 55
}
```

and returns a prediction.

## Pydantic Validation

A Pydantic model is used to validate incoming requests.

The API automatically rejects invalid request data.

For example, invalid input can result in:

```text
HTTP 422 Unprocessable Entity
```

## Important Lesson

An API is an interface between a client and the underlying application logic.

The client should not need to know how the prediction is calculated.

---

# Day 2 — Separation of Concerns and Testing

## Concepts Learned

* Separation of Concerns
* Service layer
* Modular architecture
* Unit testing
* Pytest
* Dependency separation

## Problem

Initially, prediction logic was written directly inside the FastAPI endpoint.

This creates a problem because the API layer becomes responsible for too many things.

A better structure is:

```text
Client
   ↓
FastAPI
   ↓
Prediction Service
```

The API handles HTTP-related responsibilities, while the prediction service handles prediction logic.

## Prediction Service

Prediction logic was moved to:

```text
services/
└── prediction.py
```

The service exposes:

```python
predict_rating(user_id, movie_id)
```

This makes the prediction logic independent from FastAPI.

## Why Separation of Concerns Matters

Separating responsibilities makes the system:

* easier to test
* easier to modify
* easier to debug
* easier to scale
* easier to reuse

This architectural principle becomes especially important when the prediction logic is later deployed inside multiple workers.

## Automated Testing

A unit test was added to verify the prediction service.

Tests are executed using:

```bash
python -m pytest
```

### Important Environment Lesson

The system had multiple Python installations.

The global `pytest` command was associated with another Python installation.

Therefore, the project uses:

```bash
python -m pytest
```

This ensures that tests run using the Python interpreter from the project's virtual environment.

---

# Day 3 — Machine Learning Pipeline

## Concepts Learned

* Machine Learning training
* Machine Learning inference
* Features
* Target
* Regression
* Training dataset
* Test dataset
* Train/test split
* Random Forest Regressor
* Mean Squared Error
* Model persistence
* Offline training

## ML Problem

The project uses a simplified movie-rating prediction problem.

Input features:

```text
user_id
movie_id
```

Target:

```text
rating
```

The task is treated as a regression problem because the target is a numerical value.

## Training vs Inference

A major distinction learned during this stage was:

### Training

Training is the process where the model learns patterns from data.

```text
Dataset
   ↓
Features + Target
   ↓
Training
   ↓
Trained Model
```

### Inference

Inference happens when a trained model receives new input and produces a prediction.

```text
Input
   ↓
Trained Model
   ↓
Prediction
```

Training and inference are therefore separate processes.

## Dataset

A small MovieLens-style dataset was created for the initial implementation.

The dataset contains:

```text
user_id
movie_id
rating
```

The dataset is intentionally small because the goal at this stage is to build and understand the inference infrastructure rather than optimize recommendation accuracy.

## Training Pipeline

The training process is:

```text
ratings.csv
    ↓
Load Dataset
    ↓
Select Features and Target
    ↓
Train/Test Split
    ↓
Train Random Forest
    ↓
Evaluate with MSE
    ↓
Save Model
```

## Random Forest

A Random Forest Regressor was selected as the initial model.

It provides a simple way to create a real ML inference pipeline without making the ML model itself the main research contribution.

The main research focus will later be distributed inference and system performance.

## Model Evaluation

Mean Squared Error (MSE) is used as an initial evaluation metric.

MSE measures the average squared difference between the predicted values and the actual target values.

## Model Persistence

The trained model is saved using `joblib`:

```text
ml/model.joblib
```

This allows the inference service to load an already-trained model instead of training the model every time the API starts.

## Important Design Decision

Training is an offline process.

Inference is an online process.

Keeping these processes separate allows the trained model to later be deployed to multiple inference workers.

---

# Day 4 — ML Model Integration and Inference

## Concepts Learned

* Model loading
* ML inference
* Prediction service integration
* Feature consistency
* Model input format
* Unit testing ML inference

## Architecture

The system evolved into:

```text
Client
   ↓
FastAPI
   ↓
Prediction Service
   ↓
Saved ML Model
   ↓
Prediction
```

The API no longer returns a hard-coded prediction.

Instead, the prediction service loads the trained model and uses it to generate the result.

## Prediction Service

The inference logic is implemented in:

```text
services/
└── prediction.py
```

The service loads:

```text
ml/model.joblib
```

and receives:

```text
user_id
movie_id
```

It then returns the predicted rating.

## Feature Consistency

During inference, the model must receive features in the same structure used during training.

The model was trained using:

```text
user_id
movie_id
```

Therefore inference also provides these named features.

This prevents the warning:

```text
X does not have valid feature names
```

and keeps the training and inference interfaces consistent.

## Testing

The inference service is tested for:

* returning a floating-point value
* returning a prediction within the expected rating range

The tests are executed with:

```bash
python -m pytest
```

## Important Observation

The current dataset is intentionally small and artificial.

Therefore, predictions from this model should not be interpreted as a scientifically meaningful recommendation system.

The purpose of this stage is to create a complete ML inference pipeline that can later be distributed across multiple workers.

---

# Day 5 — ML Workers and Load Balancing

## Concepts Learned

* Distributed ML inference
* ML Worker
* Worker abstraction
* Multiple inference instances
* Load Balancer
* Request distribution
* Round Robin
* Circular scheduling
* Separation between routing and inference

## Architecture Evolution

Before introducing workers:

```text
Client
   ↓
FastAPI
   ↓
Prediction Service
   ↓
ML Model
```

After introducing the Worker abstraction:

```text
Client
   ↓
FastAPI
   ↓
ML Worker
   ↓
Prediction Service
   ↓
ML Model
```

The target distributed architecture is:

```text
                    ┌── ML Worker 1
                    │
Client → FastAPI → Load Balancer ── ML Worker 2
                    │
                    └── ML Worker 3
```

## Why Do We Need Workers?

A single inference process can become a bottleneck when many requests arrive.

Multiple workers provide multiple inference instances that can process requests independently.

This creates the foundation for studying:

* scalability
* latency
* throughput
* resource utilization
* fault tolerance

## Worker Responsibility

An `MLWorker` represents one inference worker.

Its responsibility is to:

1. receive an inference request
2. call the prediction service
3. return the prediction

The Worker does not decide which worker receives a request.

That responsibility belongs to the Load Balancer.

## Worker IDs

Each worker has a unique identifier:

```text
worker-1
worker-2
worker-3
```

The worker ID makes request distribution observable during development and testing.

Example:

```json
{
    "user_id": 12,
    "movie_id": 55,
    "predicted_rating": 3.15,
    "worker_id": "worker-1"
}
```

## Round Robin Load Balancing

Round Robin is a simple load-balancing strategy.

Requests are assigned to workers in a circular order.

For three workers:

```text
Request 1 → Worker 1
Request 2 → Worker 2
Request 3 → Worker 3
Request 4 → Worker 1
Request 5 → Worker 2
Request 6 → Worker 3
```

The implementation maintains a current index and moves to the next worker after each request.

The circular behavior is implemented using the modulo operation:

```python
(current_index + 1) % len(workers)
```

For three workers:

```text
0 → 1 → 2 → 0 → 1 → 2 → ...
```

## Load Balancer Responsibility

The Load Balancer is responsible for:

```text
Incoming Request
       ↓
Choose Worker
       ↓
Forward Request
```

The Worker is responsible for:

```text
Receive Request
       ↓
Run ML Inference
       ↓
Return Prediction
```

Keeping these responsibilities separate allows different load-balancing strategies to be implemented and compared without changing the ML inference logic.

## Current Implementation

The project now contains:

```text
load_balancer/
└── round_robin.py
```

and:

```text
workers/
└── worker.py
```

The FastAPI application creates three logical ML workers and passes them to the Round Robin load balancer.

## Testing

Tests currently cover:

* prediction service
* prediction range
* ML Worker
* Round Robin request distribution

The current test suite contains:

```text
4 tests
4 passed
```

Tests are executed using:

```bash
python -m pytest
```

## Important Architectural Limitation

At this stage, the three workers are logical worker objects inside the same Python process.

They are not yet independent processes or containers.

Therefore, the system is not yet a fully distributed deployment.

This is intentional.

The project first implements and tests the distributed-system logic at the application level.

Later, Docker will be used to run independent worker instances.

---

# Current System

The system currently follows:

```text
                    ┌── Worker 1
                    │
Client → FastAPI → Round Robin
                    │
                    ├── Worker 2
                    │
                    └── Worker 3
                           ↓
                    Prediction Service
                           ↓
                      ML Model
```

## Current Components

```text
main.py
services/
└── prediction.py

ml/
├── train.py
└── model.joblib

workers/
└── worker.py

load_balancer/
└── round_robin.py

tests/
├── test_prediction.py
├── test_worker.py
└── test_load_balancer.py

data/
└── ratings.csv
```

---

# Research Direction

The central research question of the project is:

> How do different load-balancing strategies affect the scalability, performance, and resource efficiency of a distributed machine learning inference system?

The initial comparison will be:

```text
Round Robin
     VS
Least Connections
```

The system will eventually be evaluated using:

* Average latency
* P50 latency
* P95 latency
* P99 latency
* Throughput
* CPU utilization
* Memory utilization
* Error rate
* Worker health
* Recovery time

---

# Next Steps

## Day 6+

The next stage will introduce:

```text
Health Checks
Fault Detection
Worker Failure
Fault Tolerance
```

The goal is to intentionally make a worker unavailable and investigate whether the system can continue serving requests.

Later stages will include:

```text
Least Connections
        ↓
Docker
        ↓
Multiple Real Worker Containers
        ↓
Load Testing with Locust
        ↓
Performance Benchmarking
        ↓
Prometheus + Grafana
        ↓
Fault-Tolerance Experiments
        ↓
Final Research Analysis
```

---

# GitHub Development Practice

Each meaningful development stage is committed separately.

The project uses descriptive commit messages such as:

```bash
git commit -m "Add ML worker abstraction"
git commit -m "Integrate round robin load balancing"
```

The purpose is to maintain a clear development history and make the architectural evolution of the project visible.

---

# Main Learning Goal

The ultimate goal is to understand not only how to build an ML model or an API, but how to turn an ML model into a reliable and measurable distributed inference system.

The project connects concepts from:

```text
Machine Learning
        +
Backend Development
        +
Distributed Systems
        +
Cloud Infrastructure
        +
Performance Engineering
        +
Fault Tolerance
        +
Observability
```

This combination forms the technical foundation for the research and experiments in the later stages of the project.
# Day 6 — Health Checks & Fault Tolerance

## 1. What did I learn today?

Today I learned how a distributed inference system can detect unavailable workers and avoid sending requests to them.

The main idea was:

> A Load Balancer should not treat every worker as available. It should consider the current health of each worker before routing a request.

---

## 2. What is a Health Check?

A **Health Check** is a mechanism used to determine whether a service or worker is available and able to process requests.

In our project, each `MLWorker` has a health state:

```text
Worker
├── worker_id
└── is_healthy
```

We added:

```python
def health_check(self) -> bool:
    return self.is_healthy
```

This allows the Load Balancer to ask:

```text
Is this worker healthy?
        ↓
      True / False
```

---

## 3. Why do we need Health Checks?

Without health checks, our Load Balancer assumes that every worker is always available.

For example:

```text
Worker 1 → healthy
Worker 2 → crashed
Worker 3 → healthy
```

A normal Round Robin algorithm could still select Worker 2:

```text
Worker 1
Worker 2 ❌
Worker 3
Worker 1
Worker 2 ❌
...
```

This would cause requests to fail.

With health-aware load balancing:

```text
Worker 1 → healthy
Worker 2 → unhealthy
Worker 3 → healthy
```

the Load Balancer skips Worker 2:

```text
Worker 1
Worker 3
Worker 1
Worker 3
...
```

---

## 4. Liveness vs Readiness

Two common concepts in distributed systems are:

### Liveness

Answers:

> Is the worker still alive?

A liveness check is mainly concerned with whether the process is running.

### Readiness

Answers:

> Is the worker ready to receive and process requests?

A worker could be alive but temporarily not ready.

For example:

```text
Worker process → running
Model loading → not finished
```

The process is alive, but it should not receive inference requests yet.

For this project, we currently use a simplified health state. Later, Docker/Kubernetes can provide more realistic health and readiness mechanisms.

---

## 5. Changing Worker Health

We added methods to control the Worker state:

```python
def mark_unhealthy(self):
    self.is_healthy = False


def mark_healthy(self):
    self.is_healthy = True
```

The important design principle is that:

```text
mark_unhealthy()
        ↓
changes state

health_check()
        ↓
checks state
```

These methods have different responsibilities.

We should not use:

```python
worker.mark_healthy()
```

as a way to check whether a worker is healthy.

Instead:

```python
worker.health_check()
```

should be used for checking the state.

---

## 6. Health-aware Round Robin

The original Round Robin algorithm simply selected the next worker:

```text
1 → 2 → 3 → 1 → 2 → 3
```

We changed it so that it checks worker health.

Conceptually:

```text
Select next worker
       ↓
Is it healthy?
   ↙        ↘
 Yes         No
 ↓           ↓
Return      Skip
worker      worker
```

The implementation checks at most `len(workers)` workers.

This is important because the Load Balancer must not search forever if every worker is unavailable.

---

## 7. What happens when all workers are unhealthy?

Consider:

```text
Worker 1 ❌
Worker 2 ❌
Worker 3 ❌
```

There is no valid destination for the request.

Therefore the Load Balancer raises:

```python
RuntimeError("No healthy workers available")
```

This is better than an infinite loop.

The system explicitly reports:

> There is currently no available worker to handle the request.

---

## 8. Testing Exceptions with pytest

Today I also learned a better way to test expected exceptions.

Instead of manually writing:

```python
try:
    ...
except RuntimeError:
    ...
```

we can use:

```python
with pytest.raises(
    RuntimeError,
    match="No healthy workers available"
):
    load_balancer.get_next_worker()
```

This tells pytest:

1. A `RuntimeError` must occur.
2. Its message must contain the expected text.

If no exception occurs, the test fails.

If a different exception occurs, the test fails.

If the expected exception occurs with the expected message, the test passes.

---

## 9. Recovery

A distributed system should not only detect failure.

It should also be able to handle recovery.

Example:

```text
Before:

Worker 1 ✅
Worker 2 ❌
Worker 3 ✅
```

The Load Balancer uses:

```text
Worker 1 → Worker 3 → Worker 1 → Worker 3
```

After Worker 2 recovers:

```text
Worker 1 ✅
Worker 2 ✅
Worker 3 ✅
```

Worker 2 becomes available again.

This introduces an important distributed-systems concept:

> A worker can transition between healthy and unhealthy states during the lifetime of the system.

---

## 10. Important Design Lesson

Today I learned that the Load Balancer and Worker have different responsibilities.

### Worker

Responsible for:

```text
Prediction
Health state
Health checking
Health state transitions
```

### Load Balancer

Responsible for:

```text
Selecting a worker
Skipping unhealthy workers
Reporting when no healthy worker exists
```

This is another example of **Separation of Concerns**.

---

## 11. Tests Added

Day 6 added tests for:

* Worker being healthy by default
* Skipping unhealthy workers
* Handling the case where all workers are unhealthy
* Worker recovery
* Correct Round Robin behavior with healthy workers

Current test suite:

```text
8 passed
```

---

## 12. Current Architecture

After Day 6:

```text
                    Client
                      │
                      ▼
                 FastAPI API
                      │
                      ▼
             Round Robin LB
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Worker 1    Worker 2    Worker 3
       HEALTHY    HEALTHY     HEALTHY
          │           │           │
          └───────────┼───────────┘
                      ▼
               Prediction Service
                      │
                      ▼
                ML Model
```

The Load Balancer now considers Worker health before routing requests.

---

## 13. Important Limitation

The workers are **not yet truly distributed**.

Currently:

```text
One Python process
 ├── Worker 1 object
 ├── Worker 2 object
 └── Worker 3 object
```

So when we say a worker is "unhealthy", we are currently simulating failure using:

```python
worker.mark_unhealthy()
```

This is intentional.

The next phase will make the architecture more realistic by running workers as separate Docker containers:

```text
FastAPI container
       │
       ▼
Load Balancer
   │    │    │
   ▼    ▼    ▼
 W1    W2    W3
```

Then we can actually stop one container and observe how the system behaves.

---

## 14. Key Concepts Learned

```text
Health Check
Liveness
Readiness
Fault Tolerance
Failure Detection
Worker Availability
Recovery
Exception Handling
pytest.raises
Separation of Concerns
Health-aware Load Balancing
```

---

## 15. Research Connection

Day 6 directly supports the research question:

> How do different load-balancing strategies affect the scalability, performance, and resource efficiency of a distributed machine learning inference system?

Health-aware routing is important because a load-balancing strategy cannot be evaluated only under normal conditions.

Later experiments should also consider:

```text
Normal operation
       +
Worker failure
       +
Worker recovery
```

This will allow us to measure not only performance but also **fault tolerance and recovery behavior**.
