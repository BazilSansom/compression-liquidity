# compression-liquidity
Temporary private repository for finalising code for the compression, liquidity paper

# Compression & Liquidity Simulation

Code for the numerical experiments in the paper:

> **The consequences of portfolio compression for systemic liquidity risk**  
> *Working title*

This repository contains a clean, modular, and reproducible implementation of the  
simulation environment, compression methods, and experimental pipeline used in the paper.

The design goal is to separate:

- a **reusable core layer** (`src/`), containing stable abstractions, models,  
  and algorithms (network representation, compression, payment dynamics, buffers);
- **experiment scripts** (`experiments/`), where specific paper experiments live;
- **notebooks** (`notebooks/`), for diagnostics, demonstrations, and exploratory analysis.

---

## 🗂 Repository Structure

