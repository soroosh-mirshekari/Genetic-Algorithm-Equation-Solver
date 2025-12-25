# Genetic Algorithm for Solving Systems of Equations

## Overview
This project presents a modular implementation of a Genetic Algorithm (GA) designed
to approximate solutions for systems of algebraic equations, including both linear
and non-linear systems. The project focuses on evolutionary optimization principles
and their application to problems where analytical solutions may be complex or
infeasible.

## Objective
The main objectives of this project are to:
- design a general-purpose genetic algorithm framework,
- formulate suitable fitness functions for equation systems,
- approximate variable values that satisfy multiple equations simultaneously.

## Approach
The algorithm represents each candidate solution as a real-valued genome corresponding
to the variables of the equation system. The optimization process includes:
- population initialization within bounded variable ranges,
- tournament-based parent selection,
- arithmetic crossover to generate offspring,
- mutation to preserve diversity and avoid premature convergence,
- elitism to retain the best solutions across generations.

Fitness is defined based on the aggregated residual error of the equation system,
encouraging convergence toward consistent solutions.

## Key Concepts
genetic algorithms • evolutionary optimization • fitness function design •
selection, crossover, and mutation • elitism • numerical approximation

## Problem Settings
The implementation supports multiple equation systems, including:
- linear systems with a small number of variables,
- non-linear systems with constraints requiring careful error handling.

The algorithm structure remains independent of the specific system being solved.

## Challenges
Key challenges addressed in this project include:
- designing a stable fitness formulation for heterogeneous equations,
- balancing exploration and exploitation through mutation and crossover rates,
- avoiding numerical instabilities such as division by zero in non-linear systems,
- maintaining convergence while preserving population diversity.

## How to Run
1. Ensure required Python dependencies are installed  
2. Run the main script:
   ```bash
   python GA.py
