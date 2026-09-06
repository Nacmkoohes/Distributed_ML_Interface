# What I Learned

This document records the main concepts I learn while building
the Distributed ML Inference Platform.

## Day 1 — Client, Server, API, and ML Inference

### Client

A client is a program that sends a request to a server.

Examples include:
- Web applications
- Mobile applications
- Postman
- Python programs

### Server

A server is a program that receives requests and sends responses.

In this project, FastAPI will be used to build the server.

### API

An API provides a defined way for different software components
to communicate with each other.

For example:

    POST /predict

can be used to send a prediction request.

### ML Inference

Inference is the process of using a trained machine learning model
to generate a prediction for new input data.

The basic flow is:

    Client
      ↓
    API
      ↓
    ML Model
      ↓
    Prediction
      ↓
    Client
## Day 2 — API Architecture and Separation of Concerns

### Separation of Concerns

The API layer should handle HTTP requests and responses,
while prediction logic should be isolated in a separate service.

Current architecture:

Client → FastAPI → Prediction Service

### Prediction Service

The prediction service contains the logic responsible for
generating predictions.

For now, the project uses a temporary fake prediction.
This will later be replaced by a real machine learning model.

### Automated Testing

Pytest is used to verify that the prediction service
returns the expected data type.

This allows the project to detect regressions automatically.
## Day 3 — Machine Learning Pipeline

### Concepts Learned

* Training vs. inference
* Features and target
* Regression
* Training and test datasets
* Train/test split
* Random Forest Regressor
* Mean Squared Error (MSE)
* Model persistence
* Why model training and inference should be separated

### ML Problem

The project uses a simplified movie-rating prediction problem.

Input features:

* `user_id`
* `movie_id`

Target:

* `rating`

The model learns a relationship between the input features and
the user's movie rating.

### Training Pipeline

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

The trained model is saved as:

```text
ml/model.joblib
```

### Important Design Decision

Model training is separated from inference.

Training is an offline process, while inference happens when
the API receives a prediction request.

This separation will become important when the model is deployed
across multiple inference workers.

### Testing

A test was added to verify that the training/inference pipeline
produces a valid prediction.

Tests are executed with:

```bash
python -m pytest
```

---

## Day 4 — ML Model Integration with FastAPI

### Concepts Learned

* Model loading
* ML inference
* Prediction service
* Connecting FastAPI to a trained ML model
* Feature names
* Model input consistency
* Unit testing ML inference

### Architecture

The prediction pipeline became:

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

The API no longer returns a hard-coded rating.

Instead, it loads the trained model and uses it to generate
a prediction.

### Prediction Service

The prediction logic is implemented in:

```text
services/prediction.py
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

### Feature Consistency

The model was trained using named features:

```text
user_id
movie_id
```

Therefore, inference also provides the same feature names.

This prevents the following warning:

```text
X does not have valid feature names
```

and keeps the training and inference interfaces consistent.

### Testing

The prediction service is tested for:

* returning a `float`
* returning a rating between `1.0` and `5.0`

Tests are executed with:

```bash
python -m pytest
```

### Important Observation

The current dataset is intentionally small and artificial.

Therefore, the numerical prediction should not be interpreted
as a scientifically meaningful recommendation result.

The main purpose at this stage is to build a complete ML inference
pipeline that can later be distributed across multiple workers.

---

## Day 5 — ML Worker Abstraction

### Concepts Learned

* Distributed ML inference
* ML Worker
* Worker abstraction
* Separation between API and inference workers
* Multiple inference instances
* Basic role of a Load Balancer
* Request distribution
* Round Robin as a load-balancing strategy

### Architecture Evolution

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

After introducing a Worker:

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

The target architecture is:

```text
                    ┌── ML Worker 1
                    │
Client → FastAPI → Load Balancer ── ML Worker 2
                    │
                    └── ML Worker 3
```

### Why Do We Need Workers?

A single ML inference process can become a bottleneck when many
requests arrive at the same time.

Multiple workers allow inference requests to be distributed
across independent processing instances.

This creates the foundation for studying:

* scalability
* latency
* throughput
* resource utilization
* fault tolerance

### Worker Responsibility

An `MLWorker` represents one inference worker.

Its responsibility is to:

1. receive an inference request
2. call the prediction service
3. return the prediction

The Worker does **not** decide which worker should receive the
request.

That responsibility belongs to the Load Balancer.

### Code Structure

The Worker abstraction is implemented in:

```text
workers/
└── worker.py
```

The main class is:

```text
MLWorker
```

Its prediction method delegates the actual ML inference to:

```text
services/prediction.py
```

### Testing

A unit test verifies that the Worker:

* produces a floating-point prediction
* produces a prediction within the expected rating range

Tests are executed with:

```bash
python -m pytest
```

### Key Research Question

Now that the system can represent an individual ML Worker,
the next question is:

> How should incoming requests be distributed efficiently
> across multiple ML Workers?

This leads to the next stage:

**Load Balancing.**

### GitHub Practice

Each meaningful architectural step is committed separately.

Example:

```bash
git add .
git commit -m "Document ML worker architecture"
git push
```

This creates a clear development history and makes the evolution
of the distributed system visible in the GitHub repository.
