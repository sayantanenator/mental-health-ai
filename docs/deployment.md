# Deployment Guide

This guide covers deployment for API and model serving.

## Prerequisites

- Docker and Docker Compose
- Cloud credentials (if deploying to cloud)

## Quick Start (Local)

1. Build and run containers:
   - See `docker/docker-compose.yml`
2. Access API at http://localhost:8000

## CI/CD

- GitHub Actions workflows in `.github/workflows/` provide test and deployment scaffolding.
