# Project Hermes

Project Hermes is an end-to-end machine learning and operations research prototype for modernizing customer support routing in the BPO sector. The system explores a smarter alternative to traditional automatic call distribution by combining intent understanding, dynamic agent profiling, and queue-aware routing decisions.

## Overview

Project Hermes is designed to improve customer experience and operational efficiency by routing each interaction to the most suitable agent based on:

- the customer's inferred intent and urgency
- the agent's current performance profile
- workload and stress-aware decision factors
- queue considerations that balance speed and service quality

The project sits at the intersection of machine learning, MLOps, and intelligent operations design.

## What the system aims to solve

Traditional routing systems often rely on rigid availability-based logic, which can lead to:

- repeated transfers or poor first-contact resolution
- mismatches between customer issues and agent expertise
- inefficient handling of high-priority or sensitive cases
- limited adaptability to changing agent performance over time

Project Hermes aims to address these issues through a more context-aware routing strategy.

## Core ideas

The current system explores three major components:

1. Intent inference
   - A natural language model classifies incoming customer utterances into issue categories and severity-related signals.

2. Dynamic agent profiling
   - Agent performance is represented through historical and live metrics that reflect how well a person may handle different issue types.

3. Smart routing decisions
   - The routing layer evaluates candidate agents and selects the most suitable assignment based on both expected resolution quality and operational constraints.

## Repository structure

The repository is organized around a modular architecture:

- apps/mock-bpo-simulator: simulation of synthetic BPO traffic and customer interactions
- apps/routing-engine: core routing service, classifier logic, and optimization components
- infrastructure: feature store, model assets, orchestration helpers, and supporting services
- packages: shared workspace tooling and configuration

## Current development status

Project Hermes is currently under active development. The codebase already contains the foundations for:

- an NLP-based intent classifier
- a routing microservice
- simulated traffic generation
- feature-store-backed agent profiling
- supporting MLOps-related artifacts and model resources
- Core operations pipeline

This project is still evolving as a research and engineering prototype, so the implementation is intentionally being shaped and refined over time.

## Project direction

Future work will focus on strengthening the system's realism and operational readiness, including:

- more robust data pipelines
- richer agent performance modeling
- improved routing quality under real-world conditions
- broader MLOps integration and observability

## Note

This README is intentionally high-level while the system is still in development. Detailed installation, environment setup, and deployment instructions will be added as the platform matures.
