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