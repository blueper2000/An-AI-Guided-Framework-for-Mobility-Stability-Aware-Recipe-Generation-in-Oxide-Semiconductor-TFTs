# An AI-Guided Framework for Mobility–Stability-Aware Recipe Generation in Oxide Semiconductor TFTs

This repository contains the full materials and tools associated with the research  
**“An AI-Guided Framework for Mobility–Stability-Aware Recipe Generation in Oxide Semiconductor TFTs.”**

The repository is organized into two main components:  
(1) Supporting Information used to construct and evaluate the dataset, and  
(2) A Recipe Generator that performs AI-based synthesis recipe prediction and evaluation.

---

## Repository Structure

### 1. Supporting Informations

This folder provides the Supporting Information for the study and includes curated datasets used for model construction and evaluation.

Contents include:
- Excel files of seed papers and author metadata  
- Extracted synthesis data from PVD processes  
- Extracted synthesis data from ALD processes  
- Quantitative extraction scores for PVD recipes  
- Quantitative extraction scores for ALD recipes  

These files form the empirical and literature-based foundation of the recipe generation framework.

---

### 2. Recipe Generator

This folder contains the implementation of the AI-based recipe generation and evaluation pipeline.

It includes:
- Structured recipe prediction using large language models  
- LLM-as-a-Judge–based evaluation of generated recipes  
- Scoring and aggregation scripts for quantitative comparison  

Detailed usage instructions, dependencies, and execution workflows are provided in  
`Recipe Generator/README.md`.

---

## Purpose

This repository is intended to support:
- Reproducible extraction of oxide semiconductor synthesis knowledge  
- Mobility–stability–aware recipe generation using AI models  
- Quantitative evaluation of generated synthesis recipes  

It serves both as the Supporting Information for the manuscript and as a reusable research toolkit.
