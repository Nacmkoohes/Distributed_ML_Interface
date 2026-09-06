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