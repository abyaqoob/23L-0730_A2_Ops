# VisionGuard

[![VisionGuard CI](https://github.com/abyaqoob/23L-0730_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/abyaqoob/23L-0730_A2_Ops/actions/workflows/ci.yml)

VisionGuard is an automated MLOps pipeline and web service for AI-generated image classification and prediction auditing.

## Architecture & Features
- **Hardened Container**: Built using `python:3.11-slim`, layer-cached dependencies, and executed under an unprivileged `appuser` (UID 1000).
- **Multi-Container Orchestration**: Docker Compose stack connecting the web service to an isolated PostgreSQL audit database via an internal bridge network.
- **Continuous Integration**: GitHub Actions workflow running automated linting (`flake8`), unit tests (`pytest`), and Docker image builds on every push.
