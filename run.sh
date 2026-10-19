#!/bin/bash
pip install fastapi uvicorn sentence-transformers faiss-cpu
uvicorn main:app --reload
