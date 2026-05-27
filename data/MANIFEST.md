# Corpus MANIFEST

**Corpus name:** AI Developers Course — Comprehensive Textbook & Engineering Guide

**Domain:** Artificial Intelligence and Machine Learning (educational/technical). Covers
foundational ML theory, deep learning, NLP, transformer architectures, large language
models, and retrieval-augmented generation (RAG) systems.

**Source of documents:** A single PDF textbook (`AI_Developers_Course_Textbook-v2.pdf`)
structured as a 12-module engineering course. Text is extracted page-by-page using
`pymupdf` (`fitz`).

**Modules covered:**
1. Introduction to ML Paradigms
2. Linear Models & Optimization
3. Trees & Ensembles
4. Neural Networks & Classification
5. NLP Basics & Preprocessing
6. Word Embeddings & Vectors
7. Transformer Architecture
8. LLMs & Prompt Engineering
9. Introduction to RAG
10. Data Ingestion & Chunking
11. Vector Stores & Retrieval
12. Project Implementation & Evaluation

**Number of documents (pages):** 152 pages → one doc per non-empty page.

The pipeline samples (by default) all pages; the cap is configurable via
`--max-per-source` (interpreted as max pages per PDF file).

**Approximate number of tokens:** ~15 k tokens of extractable text across 152 pages
(the PDF contains diagrams, code blocks, and math formulas that yield sparse text).

**File types:** PDF (`.pdf`), one file.

**License / permission:** Course material provided for educational use within this
assignment. No private or personally identifiable information is included.

**Why this corpus is suitable for RAG:**
* The textbook is technical and module-dense, so a baseline LLM is unlikely to
  reproduce exact definitions, formulas, or code snippets without retrieval.
* Content spans math, pseudocode, Python implementations, and conceptual explanations —
  which makes grounding and citation meaningful.
* Module structure (12 distinct topics) supports multi-category evaluation questions
  (factual, numerical, comparison, negation).

**What kind of questions the system should answer:**
* Factual: "What is the purpose of the attention mechanism?" / "What does RAG stand for?"
* Numerical: "How many modules does the course have?" / "What is the MSE formula?"
* Temporal / structural: "In which module is chunking covered?"
* Comparison: "What is the difference between word embeddings and transformers?"
* Negation / absence: questions about topics *not* covered in the textbook, which the
  system must explicitly refuse to answer.

**Privacy:** The PDF contains no personal data, user accounts, or private identifiers.
