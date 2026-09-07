# TDR: Two-Dimensional Reachability Index for Pattern-Constrained Queries

This repository contains the **C++ implementation** of the algorithms presented in the paper:

> **"Fast Answering Pattern-Constrained Reachability Queries with Two-Dimensional Reachability Index"**

The program evaluates **pattern-constrained reachability queries** on directed graphs, supporting **DFS baseline**, **only-H (only the horizontal dimension)**, **only-V ( only the vertical dimension)**, and the full **TDR (Two-Dimensional Reachability)** approach.

---

## 📁 Input File Format

### Graph File
A plain text file where each line represents a directed edge:
> u v label(s)

- `u`, `v`: vertex IDs
- `label(s)`: one or more labels on the edge
  - For **single-label** edges: directly specify the integer label, e.g., `1 2 5`
  - For **multi-label** edges: separate labels with **commas**, e.g., `1 2 3,5,7`

**Note:** If your raw graph file is not in this format, you can use the provided `format_graph` to convert it into the required structure.

---

### Query File
A plain text file where each line represents one query:
> source target pattern_type $l_1$, $l_2$, $\dots$, $l_k$


- `source`, `target`: vertex IDs
- `pattern_type`: an integer (1–10) specifying the query type (see table below)
- $l_1$, $l_2$, $\dots$, $l_k$: the labels forming the pattern constraint

#### Query Type Encoding
| Pattern Code | Query Type | Mark Set Size |
|:---:|:---|:---:|
| 1 | PCR($\mathbb{AND}$) | Small |
| 2 | PCR($\mathbb{OR}$) | Small |
| 3 | PCR($\mathbb{NOT}$) | Small |
| 4 | LCR | Small |
| 5 | PCR($\mathbb{AND}$) | Large |
| 6 | PCR($\mathbb{OR}$) | Large |
| 7 | PCR($\mathbb{OR}$) | Large |
| 8 | LCR | Large |
| 9 | $(l_0)^+(l_1)^+\dots$ | — |
| 10 | $(l_0 l_1 \dots)^+$ | — |

- **Small Mark Set**: graphs where the total number of distinct edge labels is **≤ 32**.
- **Large Mark Set**: graphs where the total number of distinct edge labels is **> 32**.

> For pattern codes **1–8**, $l_1$, $l_2$, $\dots$, $l_k$ denotes a **set** of labels (order does not matter).  
> For pattern codes **9 and 10**, $l_1$, $l_2$, $\dots$, $l_k$ denotes a **sequence** of labels (order matters and must be followed exactly).

---

## 🛠️ Compilation

Compile the code with a C++17-compliant compiler:
```bash
g++ -std=c++17 -O3 -o TDR TDR.cpp
```

---

## 🚀 Usage
```bash
./TDR <graph_file> <query_file> [--compare]
```

Run with full TDR index only:
```bash
./TDR ../datasets/citeseer ../datasets/citeseer_pcr
```

Run all methods and compare traversal costs:
```bash
> ./TDR ../datasets/citeseer ../datasets/citeseer_pcr --compare
```

---

## 📝 Notes
- The graph is assumed to be directed.
- The `--compare` mode may be slower on large datasets due to DFS traversals.
