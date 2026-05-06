#!/usr/bin/env bash
set -e

export RAG_PIXEL_BASE_DIR=${RAG_PIXEL_BASE_DIR:-$(pwd)}

python evaluate.py \
  --benchmark data/eval/benchmark_jinst.json \
  --out_dir data/eval/runs/jinst_v1 \
  --top_k 10 \
  --configs bm25,dense,hybrid,graph_rag,graph_path_rag,agentic_graph_rag
