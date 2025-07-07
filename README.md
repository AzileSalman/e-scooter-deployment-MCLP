# Optimal Deployment of E-Scooter Stations Using MCLP

This repository contains the code for my **ongoing Honours research project** in Applied Mathematics at Stellenbosch University.

The project investigates mathematical methods for determining **optimal deployment locations for shared e-scooters** in urban areas. It applies the **Maximal Covering Location Problem (MCLP)** on graph-based representations of urban road networks with given demand distributions. The town of **Stellenbosch, South Africa**, is used as a case study to demonstrate the model’s application in a real-world setting.

## Project Status
Ongoing – additional data visualisations will be added in July 2025 for the interim report.

## What’s Included
- `mclp_stellenbosch.py`: Solves MCLP on Stellenbosch road network (case study)
- `grid_graphs_testing.py` and `mclp_stellenbosch_testing: Tests the grid graph networks and the stellenbosch network respectively
- Visualisation using `matplotlib`
- Demand weighting using real and synthetic Points of Interest (POIs)

## Tools & Skills
- Python (`pulp`, `osmnx`, `matplotlib`, `networkx`)
- Graph theory and network modelling
- Integer Linear Programming (ILP)

## Requirements
Install required packages:
```bash
pip install pulp osmnx matplotlib networkx pandas
