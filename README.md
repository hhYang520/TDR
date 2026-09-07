# TDR: Two-Dimensional Reachability Index for Pattern-Constrained Queries

This repository contains the **C++ implementation** of the algorithms presented in the paper:

> **"Fast Answering Pattern-Constrained Reachability Queries with Two-Dimensional Reachability Index"**

The program evaluates **pattern-constrained reachability queries** on directed graphs, supporting **DFS baseline**, **only-H (only the horizontal dimension)**, **only-V ( only the vertical dimension)**, and the full **TDR (Two-Dimensional Reachability)** approach.

---

## 📁 Input File Format

### Graph File
A plain text file where each line represents a directed edge:
> u v labels

text
- `u`, `v`: vertex IDs
- `labels`: an integer label on the edge

### Query File
A plain text file where each line represents one reachability query:
source target pattern_len label1 label2 ... labelk

text

- `source`, `target`: vertex IDs
- `pattern_len`: number of labels in the pattern (k)
- `label1 ... labelk`: the sequence of edge labels that must appear in order along the path

---
