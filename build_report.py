"""Generate report.pdf for the mid-term assignment."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceAfter=6, spaceBefore=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=8)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
SMALL = ParagraphStyle("SMALL", parent=styles["BodyText"], fontSize=8.5, leading=11)
CODE = ParagraphStyle("CODE", parent=styles["Code"], fontSize=8.5, leading=11, leftIndent=8)

doc = SimpleDocTemplate(
    "report.pdf", pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.6*cm, bottomMargin=1.6*cm,
    title="Mid-Term Report: RAG Pipeline on AI Developers Course Textbook",
)

story = []

def P(text, style=BODY):
    story.append(Paragraph(text, style))

CELL = ParagraphStyle("CELL", parent=styles["BodyText"], fontSize=8.5, leading=11)
def C(text):
    return Paragraph(text, CELL)

def tbl(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))

# ---------------------- title ----------------------
P("Mid-Term Assignment Report: Custom RAG Pipeline on an AI Course Textbook", H1)
P("Author: Kosta &nbsp;&nbsp;|&nbsp;&nbsp; Date: 2026-05-26", SMALL)
P(
    "This report documents a complete Retrieval-Augmented Generation (RAG) "
    "pipeline built over a 152-page AI Developers Course Textbook (PDF). "
    "The system implements the full pipeline "
    "(loading &rarr; cleaning &rarr; chunking &rarr; embedding &rarr; indexing &rarr; "
    "retrieval &rarr; generation &rarr; citation &rarr; evaluation) and exposes the "
    "required <font face=\"Courier\">answer(question)</font> interface. "
    "Three retrieval strategies are compared: dense vector search, BM25 lexical "
    "retrieval, and a hybrid RRF fusion of both."
)

# ---------------------- 1. corpus ----------------------
P("1. Corpus Description", H2)
P(
    "The corpus is a single structured PDF: "
    "<b>AI Developers Course Textbook v2</b> (257 KB, 152 pages). "
    "It covers twelve modules of an AI/ML curriculum:"
)
tbl([
    ["Module", "Topic"],
    ["1", "Introduction to ML Paradigms"],
    ["2", "Linear Models &amp; Optimization"],
    ["3", "Trees &amp; Ensembles"],
    ["4", "Neural Networks &amp; Classification"],
    ["5", "NLP Basics &amp; Preprocessing"],
    ["6", "Word Embeddings &amp; Vectors"],
    ["7", "Transformer Architecture"],
    ["8", "LLMs &amp; Prompt Engineering"],
    ["9", "Introduction to RAG"],
    ["10", "Data Ingestion &amp; Chunking"],
    ["11", "Vector Stores &amp; Retrieval"],
    ["12", "Project Implementation &amp; Evaluation"],
], col_widths=[1.5*cm, 14*cm])
P(
    "<b>Suitability for RAG.</b> The textbook is structured with explicit "
    "module/concept/subsection headings, enabling precise provenance tracking. "
    "A baseline LLM may know general ML concepts but lacks the exact "
    "module-level structure, code examples, and numerical details present in "
    "this specific document, making retrieval load-bearing for precise answers. "
    "Rich structural metadata (module, concept, subsection, page range) enables "
    "fine-grained citation and grounding."
)
P(
    "<b>Extraction.</b> Text is extracted page-by-page via PyMuPDF "
    "(<font face=\"Courier\">fitz</font>). Font-size heuristics identify "
    "module markers (&ge;28 pt), section headings (17&ndash;21 pt), named "
    "subsections (13.5&ndash;16 pt), and body text (&le;11 pt). Page-number "
    "lines are discarded. Total indexed chunks: <b>182</b> (structured mode)."
)

# ---------------------- 2. architecture ----------------------
P("2. System Architecture", H2)
P("End-to-end pipeline:")
P(
    "<font face=\"Courier\">PDF &rarr; load_pdf_structured() &rarr; clean_text() &rarr; "
    "chunk_documents() &rarr; SentenceTransformer encode &rarr; Chroma (HNSW) "
    "&rarr; Retriever [dense | BM25 | hybrid-RRF] &rarr; build_prompt() "
    "&rarr; Ollama (gemma3:4b) &rarr; parse_citations() &rarr; answer()</font>",
    CODE,
)
P(
    "Concrete components: three document loaders and two chunkers in "
    "<font face=\"Courier\">src/utils.py</font>; index builder in "
    "<font face=\"Courier\">src/build_index.py</font>; three retriever classes "
    "(<font face=\"Courier\">Retriever</font>, "
    "<font face=\"Courier\">BM25Retriever</font>, "
    "<font face=\"Courier\">HybridRetriever</font>) in "
    "<font face=\"Courier\">src/retrieval.py</font>; LLM generation and citation "
    "parsing in <font face=\"Courier\">src/generation.py</font>; and the "
    "assignment-required <font face=\"Courier\">answer()</font> entry point in "
    "<font face=\"Courier\">src/rag_system.py</font>. "
    "Evaluation tooling lives in "
    "<font face=\"Courier\">eval/build_gold_set.py</font> and "
    "<font face=\"Courier\">eval/run_eval.py</font>."
)

# ---------------------- 3. chunking ----------------------
P("3. Chunking Strategy", H2)
P(
    "Three document <i>load modes</i> control document granularity before "
    "chunking, selectable via "
    "<font face=\"Courier\">--load-mode</font>:"
)
tbl([
    ["Load Mode", "Granularity", "Chunks"],
    [C("full"), C("One document for entire PDF"), C("Depends on chunk size")],
    [C("page"), C("One document per page"), C("Up to 152 docs")],
    [C("structured (default)"), C("One doc per logical section via font-size heuristics"), C("182 sections")],
], col_widths=[3.5*cm, 7*cm, 5.5*cm])
P(
    "Two <i>chunking strategies</i> are then applied, selectable via "
    "<font face=\"Courier\">--chunk-strategy</font>:"
)
tbl([
    ["Strategy", "Parameters", "Behavior"],
    [C("fixed (default)"), C("chunk_size=500 words, overlap=80"), C("Word-level sliding window. Cheap and predictable.")],
    [C("sentence"), C("max_words=120, overlap_sents=1"), C("Greedy sentence packing up to a word budget with one-sentence overlap. Preserves natural boundaries.")],
], col_widths=[3*cm, 4.5*cm, 9*cm])
P(
    "<b>Why this design.</b> In structured mode the textbook sections average "
    "~80 words, so nearly every section maps to a single chunk regardless of "
    "strategy &mdash; chunking parameters are effectively inert at this "
    "granularity. The sentence strategy is available for full-PDF or page "
    "modes where multi-paragraph sections can exceed the word budget."
)
P(
    "<b>Concrete example where chunking helped.</b> Module 9 (Introduction to "
    "RAG) spans multiple subsections including Theoretical Foundation and "
    "Python Implementation. Structured load keeps each subsection as a "
    "separate chunk with dedicated metadata, so a question about the "
    "implementation details retrieves the correct subsection rather than "
    "being drowned out by the broader theoretical content."
)
P(
    "<b>Where chunking is limited.</b> Very short subsections (under 30 words) "
    "produce embeddings dominated by structural boilerplate (\"Knowledge "
    "Check\", \"Learning Objective\"). These chunks may surface on broad "
    "queries even when content-irrelevant, adding noise to top-k results."
)

# ---------------------- 4. embedding/index ----------------------
P("4. Embedding &amp; Vector Index", H2)
P(
    "Embeddings: <font face=\"Courier\">sentence-transformers/all-MiniLM-L6-v2</font> "
    "(384-d, L2-normalized). Chosen for its strong cost/quality tradeoff on a "
    "CPU-only setup &mdash; encoding 182 chunks completes in seconds on a laptop "
    "CPU with no GPU required. The model is LRU-cached across retrieval calls."
)
P(
    "Vector store: <font face=\"Courier\">Chroma</font> with HNSW and cosine "
    "distance, persisted under <font face=\"Courier\">data/processed/chroma/</font>. "
    "Each document stores the embedding, chunk text, and a metadata dict "
    "(<font face=\"Courier\">doc_id, chunk_id, source, page_start, page_end, "
    "module, concept, subsection</font>). Storing structural metadata at index "
    "time allows the LLM to produce grounded citations with module-level "
    "provenance without re-fetching the source document."
)
P(
    "<b>Reproducibility.</b> Deleting "
    "<font face=\"Courier\">data/processed/chroma</font> and rerunning "
    "<font face=\"Courier\">python src/build_index.py --clean</font> rebuilds "
    "an identical index. Chunk IDs are deterministic strings of the form "
    "<font face=\"Courier\">pdf_&lt;filename&gt;_sec&lt;N&gt;_chunk_&lt;i&gt;</font>."
)

# ---------------------- 5. retrieval ----------------------
P("5. Retrieval", H2)
P(
    "Three retrieval strategies are implemented, all returning "
    "<font face=\"Courier\">{chunk_id, text, score, metadata}</font> dicts. "
    "Default top-k = 5, configurable via "
    "<font face=\"Courier\">RAG_TOP_K</font> or the "
    "<font face=\"Courier\">k</font> argument to "
    "<font face=\"Courier\">answer()</font>."
)
tbl([
    ["Class", "Method", "Description"],
    [C("Retriever"), C("Dense (Chroma + MiniLM)"),
     C("Encodes query with MiniLM, queries Chroma HNSW index. score = 1 - cosine_distance.")],
    [C("BM25Retriever"), C("Sparse (rank_bm25, in-memory)"),
     C("Tokenises on [A-Za-z0-9_]+, lowercased. Filters zero-score results. Build time &lt;50 ms.")],
    [C("HybridRetriever"), C("RRF fusion of Dense + BM25"),
     C("Reciprocal Rank Fusion (K=60). Retrieves 4&times;k from each retriever, fuses, re-ranks. No score normalisation needed.")],
], col_widths=[3*cm, 4*cm, 9*cm])
P(
    "<b>Reported metrics.</b> <b>Hit@k</b>: fraction of grounded questions "
    "where at least one retrieved chunk shares a chunk_id or doc_id with the "
    "gold entry. <b>MRR@k</b>: mean reciprocal rank of the first correct "
    "result. <b>Answer Accuracy</b>: token-level F1 (stop-words removed) "
    "between generated answer and reference answer, sampled over the first "
    "30 grounded questions. Negation/absence questions are evaluated "
    "separately (correct iff the system returns the refusal string)."
)

# ---------------------- 6. prompt design ----------------------
P("6. Prompt Design", H2)
P(
    "The system prompt enforces five hard rules: (1) answer only from the "
    "provided context chunks, (2) say exactly “The information was not "
    "found in the provided context.” if absent, (3) cite chunks inline "
    "using <font face=\"Courier\">[chunk_id]</font> in square brackets, "
    "(4) never invent facts, numbers, or names, (5) keep answers concise "
    "(1&ndash;4 sentences unless a list is required)."
)
P(
    "Context is formatted as one block per chunk with a structured header:"
)
P(
    "<font face=\"Courier\">[chunk_id] (section=module &gt; concept &gt; subsection, "
    "pages=p&lt;start&gt;-&lt;end&gt;)</font>",
    CODE,
)
P(
    "This surfaces the structural hierarchy (module, concept, subsection) and "
    "page range so the LLM can reason about provenance without additional "
    "look-ups. Citations are extracted from the LLM reply with the regex "
    "<font face=\"Courier\">\\[([^\\[\\]\\s,]+_chunk_\\d+)\\]</font> and "
    "intersected with the set of retrieved IDs &mdash; any ID the model "
    "invents but did not receive in context is silently dropped."
)

# ---------------------- 7. evaluation ----------------------
P("7. Evaluation Results", H2)
P(
    "Gold set: <b>~182 questions</b> (one per chunk) generated by "
    "<font face=\"Courier\">eval/build_gold_set.py</font> using the same "
    "gemma3:4b LLM. Questions cycle through five required categories "
    "(factual, numerical, temporal, comparison, negation/absence) with a "
    "fixed seed for reproducibility. Negation entries carry empty "
    "<font face=\"Courier\">must_cite_chunk_ids</font> by design."
)
P("<b>Retrieval and generation metrics across all ablation configs:</b>")
tbl([
    ["Config", "Retriever", "k", "Chunks", "Hit@k", "MRR@k", "Ans Acc"],
    ["dense_k3", "dense", "3", "182", "0.620", "0.522", "0.261"],
    ["dense_k5 (baseline)", "dense", "5", "182", "0.672", "0.532", "0.287"],
    ["dense_k8", "dense", "8", "182", "0.693", "0.535", "0.294"],
    ["bm25_k5", "BM25", "5", "182", "0.679", "0.609", "0.364"],
    ["hybrid_k5", "hybrid RRF", "5", "182", "0.693", "0.588", "0.340"],
], col_widths=[4*cm, 2.5*cm, 0.8*cm, 1.5*cm, 1.4*cm, 1.4*cm, 1.8*cm])

P("<b>Manual answer classification</b> (10 sampled answers from "
  "<font face=\"Courier\">eval/answers_sample.jsonl</font>):")
tbl([
    ["Class", "Count", "Notes"],
    [C("Correct"), "7", C("Right answer with valid in-context citation.")],
    [C("Partially correct"), "0", C("")],
    [C("Incorrect"), "2", C("Generation returned refusal despite gold chunk appearing in top-5 (retrieval noise).")],
    [C("Hallucinated"), "1", C("Model cited a chunk ID not present in the retrieved set (dropped by citation filter).")],
], col_widths=[4*cm, 1.5*cm, 11*cm])

# ---------------------- 8. ablation ----------------------
P("8. Ablation Study", H2)
P(
    "Two ablation axes were varied: <b>retriever type</b> (dense, BM25, hybrid) "
    "and <b>top-k</b> (3, 5, 8 with dense retrieval). Chunk size was not "
    "varied because in structured load mode all sections average ~80 words "
    "&mdash; both 300-word and 700-word windows produce identical indices."
)
tbl([
    ["Experiment", "Hit@k", "MRR@k", "Ans Acc", "Key finding"],
    [C("dense, k=3"), "0.620", "0.522", "0.261", C("Focused window; misses when gold chunk ranked 4+.")],
    [C("dense, k=5 (baseline)"), "0.672", "0.532", "0.287", C("Balanced precision/recall baseline.")],
    [C("dense, k=8"), "0.693", "0.535", "0.294", C("Marginal recall gain; MRR barely moves.")],
    [C("BM25, k=5"), "0.679", "0.609", "0.364", C("Best MRR and answer accuracy. Exact-term queries excel.")],
    [C("Hybrid RRF, k=5"), "0.693", "0.588", "0.340", C("Best Hit@k (tied dense_k8). RRF smooths BM25 sparsity.")],
], col_widths=[4.2*cm, 1.3*cm, 1.3*cm, 1.3*cm, 8*cm])
P(
    "<b>Reading the results.</b> The retriever-type axis dominates: BM25 "
    "gains +7.7 points of MRR over the dense baseline, because many gold "
    "questions reference exact module names or subsection titles that BM25 "
    "matches lexically. Hybrid RRF achieves the highest Hit@k by capturing "
    "both lexical and semantic matches. Increasing k from 5 to 8 in dense "
    "retrieval improves Hit@k by ~2 points but does not recover the MRR gap "
    "with BM25, confirming that rank quality matters more than recall breadth. "
    "Answer accuracy tracks MRR closely, suggesting generation quality is "
    "gated by rank-1 retrieval precision."
)

# ---------------------- 9. failure analysis ----------------------
P("9. Failure Analysis", H2)
P(
    "From the 10 manually classified samples, 3 errors were observed:"
)
P(
    "&bull; <b>Retrieval noise causing false refusals (2 cases).</b> "
    "The gold chunk was present in the top-5 retrieved set but ranked "
    "below a cluster of topically similar chunks. The LLM, seeing no "
    "clear answer in the highest-scored passages, issued the refusal "
    "string. A cross-encoder reranker over top-20 would likely surface "
    "the gold chunk to rank 1."
)
P(
    "&bull; <b>Citation hallucination (1 case).</b> On a code-snippet "
    "question, the model cited a chunk ID "
    "(<font face=\"Courier\">sec124</font>) that was not in the retrieved "
    "set. The citation filter silently dropped the invented ID, so the "
    "final answer was returned without a source. This reveals that the "
    "LLM occasionally confabulates IDs when the context contains many "
    "similar-looking code blocks."
)
P(
    "&bull; <b>Answer accuracy plateau.</b> Despite Hit@k reaching 0.693, "
    "token-level F1 peaks at 0.364. The gap indicates a generation "
    "bottleneck: gemma3:4b occasionally paraphrases rather than quotes "
    "numbers or exact subsection names, causing partial F1 matches on "
    "numerical and temporal questions. A larger or instruction-tuned model "
    "would reduce this gap."
)
P(
    "&bull; <b>Top-k ceiling for dense retrieval.</b> Raising k from 5 "
    "to 8 did not recover any additional grounded questions beyond what "
    "BM25 already retrieves, confirming the issue is embedding "
    "representation quality rather than recall breadth."
)

# ---------------------- 10. improvements ----------------------
P("10. What I Would Improve Next", H2)
P(
    "&bull; <b>Cross-encoder reranker</b> (e.g. "
    "<font face=\"Courier\">ms-marco-MiniLM-L-6-v2</font>) over the "
    "top-20 hybrid results to improve rank-1 precision and close the "
    "MRR gap. Expected to directly reduce the false-refusal failure mode."
)
P(
    "&bull; <b>Metadata-filtered retrieval</b>: when the question "
    "mentions “Module X” or a specific subsection name, pre-filter "
    "the Chroma query on the "
    "<font face=\"Courier\">module</font> metadata field before dense "
    "search. This eliminates cross-module false positives."
)
P(
    "&bull; <b>LLM-as-judge evaluation</b> across the full ~182-question "
    "gold set to replace the 10-sample manual classification with a "
    "scalable correctness signal."
)
P(
    "&bull; <b>Larger generator</b> (e.g. "
    "<font face=\"Courier\">gemma3:12b</font> or "
    "<font face=\"Courier\">llama3:8b-instruct</font>) to close the "
    "answer-accuracy gap; retrieval is already at 0.693 Hit@5, so "
    "generation is the active bottleneck."
)
P(
    "&bull; <b>Chunk-size ablation in full/page load modes</b> to "
    "measure the actual impact of window size on this corpus, since "
    "structured mode makes all sections too short for chunk-size "
    "variation to matter."
)
P(
    "&bull; <b>Query-expansion or HyDE</b> (Hypothetical Document "
    "Embeddings): generate a hypothetical answer and embed it as the "
    "query. Particularly promising for numerical and comparison questions "
    "where the query phrasing diverges from the document phrasing."
)

doc.build(story)
print("Wrote report.pdf")
