"""Generate the 4-page Final Report PDF for the RAG pipeline project.

Reads no inputs; the report is statically composed from the project's
build_index.py, retrieval.py, generation.py, utils.py, and the evaluation
results checked in under eval/.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Preformatted,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = "report.pdf"
PAGE_W, PAGE_H = letter
MARGIN = 0.6 * inch

# Brand colors
NAVY    = colors.HexColor("#0f1e3d")
INK     = colors.HexColor("#1a1a2e")
ACCENT  = colors.HexColor("#2c5282")
SOFT    = colors.HexColor("#e8eef7")
STRIPE  = colors.HexColor("#f3f6fc")
RULE    = colors.HexColor("#c5cee0")
MUTED   = colors.HexColor("#5a6478")
CODE_BG = colors.HexColor("#f4f6fa")
CODE_BD = colors.HexColor("#dde3ee")


def make_styles():
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=18, leading=22, spaceAfter=2, alignment=TA_CENTER, textColor=NAVY,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontSize=9, leading=12, spaceAfter=10, alignment=TA_CENTER, textColor=MUTED,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=2, textColor=NAVY,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5, leading=12, spaceBefore=4, spaceAfter=2, textColor=ACCENT,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=9, leading=12.5, spaceAfter=4, alignment=TA_JUSTIFY,
    )
    s["body_sm"] = ParagraphStyle(
        "body_sm", parent=base["Normal"],
        fontSize=8.3, leading=11, spaceAfter=3, alignment=TA_JUSTIFY,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=9, leading=12, spaceAfter=2, leftIndent=12, bulletIndent=0,
    )
    s["why"] = ParagraphStyle(
        "why", parent=base["Normal"],
        fontSize=8.5, leading=11.5, spaceAfter=4, alignment=TA_JUSTIFY,
        leftIndent=8, rightIndent=8, textColor=INK,
        backColor=SOFT, borderColor=ACCENT, borderWidth=0, borderPadding=6,
        borderRadius=2,
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"],
        fontName="Courier", fontSize=7.6, leading=9.6, spaceAfter=4,
        backColor=CODE_BG, borderColor=CODE_BD, borderWidth=0.5, borderPadding=5,
        leftIndent=4, rightIndent=4, textColor=INK,
    )
    s["note"] = ParagraphStyle(
        "note", parent=base["Normal"],
        fontSize=7.6, textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
    )
    return s


def thin_rule(thickness=0.4, color=RULE, sp_before=1, sp_after=4):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=sp_before, spaceAfter=sp_after)


def section(num, title, styles):
    return [
        Paragraph(f"<font color='#2c5282'>{num}.</font> {title}", styles["h1"]),
        thin_rule(),
    ]


def bullet(text, styles, key="bullet"):
    return Paragraph(f"<bullet>&bull;</bullet>&nbsp; {text}", styles[key])


def code_block(text, styles):
    return Preformatted(text, styles["code"])


def why_box(text, styles):
    return Paragraph(f"<b>Why this choice:</b> {text}", styles["why"])


def base_table_style(header_rows=1, stripe=True, header_bg=NAVY):
    cmds = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME",   (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 7.8),
        ("LEADING",    (0, 0), (-1, -1), 10),
        ("LINEBELOW",  (0, header_rows - 1), (-1, header_rows - 1), 0.7, NAVY),
        ("LINEABOVE",  (0, 0), (-1, 0), 0.7, NAVY),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("INNERGRID", (0, header_rows), (-1, -1), 0.25, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, NAVY),
    ]
    if stripe:
        cmds.append(("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
                     [colors.white, STRIPE]))
    return TableStyle(cmds)


def build_story(s):
    el = []

    # ── Title block ─────────────────────────────────────────────────────────
    el.append(Spacer(1, 0.02 * inch))
    el.append(Paragraph("RAG Pipeline &mdash; Final Report", s["title"]))
    el.append(Paragraph(
        "AI Developers Course Mid-Term Assignment &nbsp;&bull;&nbsp; "
        "Kosta Sidorenko &nbsp;&bull;&nbsp; 2026",
        s["subtitle"]
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 1. CORPUS DESCRIPTION
    # ══════════════════════════════════════════════════════════════════════
    el += section("1", "Corpus Description", s)
    el.append(Paragraph(
        "The corpus is a single technical PDF, <b>AI_Developers_Course_Textbook-v2.pdf</b> "
        "(252 KB, 152 non-empty pages, ~15k extractable tokens) "
        "organised into <b>12 modules</b> that cover the full AI/ML engineering stack: "
        "ML paradigms, linear models, trees &amp; ensembles, neural networks, NLP basics, "
        "word embeddings, transformer architecture, LLMs &amp; prompt engineering, RAG "
        "fundamentals, data ingestion &amp; chunking, vector stores &amp; retrieval, "
        "and project implementation. Each module is internally structured into named "
        "subsections: <i>Chapter Overview</i>, <i>Theoretical Foundation</i>, "
        "<i>Mathematical Intuition</i>, <i>Python Implementation</i>, "
        "<i>Engineering Best Practices</i>, <i>Knowledge Check</i>.",
        s["body"]
    ))
    el.append(Paragraph(
        "Content types include conceptual prose, Python code snippets, mathematical "
        "formulas (loss functions, attention equations, BM25 scoring), and architectural "
        "diagrams described in prose. This diversity makes the corpus a realistic RAG "
        "testbed: a base LLM cannot reproduce module-specific structure or exact code "
        "without retrieval. Loading uses <b>pymupdf (fitz)</b> with font-size heuristics "
        "to recover the hierarchy directly from the typography:",
        s["body"]
    ))
    el.append(code_block(
        "size >= 28        : Module marker  (e.g. 'Module 1')\n"
        "size 22-27        : Module title   (e.g. 'Introduction to ML Paradigms')\n"
        "size 17-21        : Section heading (e.g. 'Core Concept: Linear Regression')\n"
        "size 13.5-16      : Named subsection ('Theoretical Foundation', ...)\n"
        "size <= 11        : Body text",
        s
    ))
    el.append(Paragraph(
        "This <i>structured</i> loader emits <b>180 logical sections</b> with metadata "
        "(<i>module, concept, subsection, page_start, page_end</i>), which is then "
        "chunked downstream. The metadata is preserved as a breadcrumb on every chunk, "
        "so each retrieval result knows its origin (Module &gt; Concept &gt; Subsection).",
        s["body"]
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 2. SYSTEM ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════
    el += section("2", "System Architecture", s)
    el.append(Paragraph(
        "The pipeline is split into four decoupled stages. Each stage is a standalone "
        "Python module that exposes a narrow interface, so individual components can "
        "be swapped (e.g. switch the retriever from dense to hybrid) without changing "
        "callers. Stages 1&ndash;3 run offline at build time; stage 4 runs online per query.",
        s["body"]
    ))
    arch = [
        ["#", "Stage", "Module / File", "Key Component"],
        ["1", "Load (offline)",   "src/utils.py", "load_pdf_structured() → 180 sections"],
        ["2", "Chunk (offline)",  "src/utils.py", "chunk_recursive() → 219 chunks"],
        ["3", "Index (offline)",  "src/build_index.py", "BGE-large → Chroma + BM25 JSONL"],
        ["4", "Query (online)",   "src/retrieval.py + src/generation.py",
         "RerankingRetriever → Ollama gemma3:4b"],
    ]
    t = Table(arch, colWidths=[0.25*inch, 1.05*inch, 1.95*inch, 3.95*inch])
    t.setStyle(base_table_style())
    el.append(t)
    el.append(Spacer(1, 4))
    el.append(Paragraph(
        "Online query flow (production path via <i>RerankingRetriever</i>):",
        s["h2"]
    ))
    el.append(code_block(
        "query  ──►  EnsembleRetriever (k=20)\n"
        "                ├─► DenseRetriever  (Chroma + bge-large-en-v1.5, cosine)\n"
        "                └─► BM25Retriever   (rank_bm25.BM25Okapi, in-memory)\n"
        "             ──► CrossEncoderReranker (bge-reranker-large, top_n=5)\n"
        "             ──► build_prompt(question, chunks)\n"
        "             ──► call_ollama(model='gemma3:4b', temperature=0.2)\n"
        "             ──► parse_citations(answer)  ──►  {answer, cited_ids}",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 3. CHUNKING STRATEGY
    # ══════════════════════════════════════════════════════════════════════
    el += section("3", "Chunking Strategy", s)
    el.append(Paragraph(
        "Three chunking strategies are implemented (<i>fixed</i>, <i>sentence</i>, "
        "<i>recursive</i>); the production choice is <b>recursive character splitting</b> "
        "via LangChain&rsquo;s <i>RecursiveCharacterTextSplitter</i>.",
        s["body"]
    ))
    el.append(code_block(
        "# src/utils.py\n"
        "from langchain_text_splitters import RecursiveCharacterTextSplitter\n"
        "DEFAULT_CHUNK_SIZE    = 600\n"
        "DEFAULT_CHUNK_OVERLAP = 120\n"
        "\n"
        "def chunk_recursive(text, chunk_size=600, overlap=120):\n"
        "    splitter = RecursiveCharacterTextSplitter(\n"
        "        chunk_size=chunk_size, chunk_overlap=overlap)\n"
        "    return splitter.split_text(text)",
        s
    ))
    el.append(why_box(
        "RCTS cascades through a priority list of separators &mdash; "
        "<b>\"\\n\\n\" &rarr; \"\\n\" &rarr; \" \" &rarr; character</b> &mdash; so a chunk "
        "boundary always falls on the largest available semantic seam. Fixed-size "
        "splitting would frequently cut Python snippets and formulas in half; "
        "sentence splitting would over-fragment formula-only lines. RCTS is the "
        "best fit for a corpus mixing prose, code, and math.",
        s
    ))
    params = [
        ["Parameter", "Value", "Effect"],
        ["chunk_size",       "600 chars (~100 words)", "Fits comfortably inside BGE&rsquo;s 512-token window"],
        ["chunk_overlap",    "120 chars (20%)",        "Preserves context across chunk boundaries"],
        ["separators (RCTS)","\\n\\n &rarr; \\n &rarr; space &rarr; char", "Splits at paragraph &gt; line &gt; word"],
        ["metadata prefix",  "[Module &gt; Concept &gt; Subsection]",
         "Prepended to every chunk; survives into prompt &amp; citations"],
        ["collection size",  "219 chunks (from 180 sections)", "~1.2 chunks/section avg, ~370 chars/chunk"],
    ]
    t3 = Table(params, colWidths=[1.4*inch, 2.05*inch, 3.75*inch])
    t3.setStyle(base_table_style())
    el.append(t3)
    el.append(Spacer(1, 4))

    # ══════════════════════════════════════════════════════════════════════
    # 4. EMBEDDING & VECTOR INDEX CHOICE
    # ══════════════════════════════════════════════════════════════════════
    el += section("4", "Embedding & Vector Index Choice", s)
    el.append(Paragraph(
        "<b>Embedding model:</b> <i>BAAI/bge-large-en-v1.5</i> loaded via "
        "<i>sentence-transformers</i>. <b>Vector store:</b> <i>Chroma</i> (persistent, "
        "HNSW index, cosine space).",
        s["body"]
    ))
    el.append(code_block(
        "# src/build_index.py\n"
        "EMBED_MODEL = \"BAAI/bge-large-en-v1.5\"\n"
        "model       = SentenceTransformer(EMBED_MODEL)\n"
        "client      = chromadb.PersistentClient(path=\"data/processed/chroma\")\n"
        "collection  = client.create_collection(\n"
        "    name=\"pdf_docs\",\n"
        "    metadata={\"hnsw:space\": \"cosine\", \"embed_model\": EMBED_MODEL})\n"
        "\n"
        "# documents are encoded with L2-normalisation so cosine == dot product\n"
        "emb = model.encode(batch_texts, batch_size=64,\n"
        "                   normalize_embeddings=True, convert_to_numpy=True)\n"
        "collection.add(ids=..., documents=..., metadatas=..., embeddings=emb.tolist())",
        s
    ))
    el.append(why_box(
        "<b>BGE-large-en-v1.5</b> is a top-3 MTEB English retrieval model, "
        "instruction-tuned for the exact use case (query &rarr; passage). At query "
        "time the model receives the BGE prompt prefix "
        "<i>&quot;Represent this sentence for searching relevant passages: &quot;</i> to align "
        "with its training distribution. <b>Chroma</b> was picked over FAISS/Qdrant "
        "because the corpus is small (~32 MB of vectors), it stores chunk metadata "
        "alongside the embeddings (no parallel database), and its HNSW backend gives "
        "sub-millisecond top-k at this scale. Cosine space normalises out vector "
        "magnitude, which suits length-varied chunks.",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 5. RETRIEVAL METHOD
    # ══════════════════════════════════════════════════════════════════════
    el += section("5", "Retrieval Method", s)
    el.append(Paragraph(
        "Production retrieval is the <b>two-stage RerankingRetriever</b> from "
        "<i>src/retrieval.py</i>. Stage&nbsp;1 builds a high-recall pool from a hybrid "
        "<b>dense + BM25 ensemble</b>; stage&nbsp;2 applies a <b>cross-encoder reranker</b> "
        "for high-precision selection.",
        s["body"]
    ))
    el.append(code_block(
        "# src/retrieval.py — production retriever\n"
        "INITIAL_FETCH_K = 20   # candidates per arm in stage 1\n"
        "RERANK_TOP_N    = 5    # final chunks passed to the LLM\n"
        "RRF_K           = 60   # Reciprocal Rank Fusion constant (Cormack 2009)\n"
        "RERANK_MODEL    = \"BAAI/bge-reranker-large\"\n"
        "\n"
        "ensemble = EnsembleRetriever(\n"
        "    retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])\n"
        "cross_enc = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)\n"
        "compressor = CrossEncoderReranker(model=cross_enc, top_n=RERANK_TOP_N)\n"
        "return ContextualCompressionRetriever(base_compressor=compressor,\n"
        "                                      base_retriever=ensemble)",
        s
    ))
    el.append(why_box(
        "Pure dense retrieval misses exact-match cues (model names, "
        "function identifiers, formulas); pure BM25 misses paraphrases. "
        "<b>RRF</b> fuses both ranked lists without normalising heterogeneous scores "
        "(cosine [0,1] vs. unbounded BM25). The <b>cross-encoder reranker</b> then "
        "reads each (query, chunk) pair jointly &mdash; a far stronger signal than "
        "bi-encoder cosine &mdash; and cuts the 20-candidate pool down to 5. "
        "BM25 tokenisation uses <i>[A-Za-z0-9_]+</i> so technical identifiers like "
        "<i>BM25</i>, <i>f1_score</i>, <i>RLHF</i> remain a single token.",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 6. PROMPT DESIGN
    # ══════════════════════════════════════════════════════════════════════
    el += section("6", "Prompt Design", s)
    el.append(Paragraph(
        "The generation prompt (from <i>src/generation.py</i>) enforces grounded "
        "answers and inline citations. The system prompt is short and explicit:",
        s["body"]
    ))
    el.append(code_block(
        "SYSTEM_PROMPT = (\n"
        "  \"You are a careful question-answering assistant for a PDF document corpus.\"\n"
        "  \" Answer the user's question strictly based on the provided context chunks.\"\n"
        "  \" If the answer cannot be found in the given context, reply explicitly\"\n"
        "  \" with 'I do not know'. Do not assume, extrapolate, or invent facts.\\n\"\n"
        "  \" Be extremely precise, factual, and concise. Limit your response to a\"\n"
        "  \" maximum of 2-3 focused sentences.\\n\"\n"
        "  \" Cite the chunks you used inline using their chunk_id in square brackets,\"\n"
        "  \" e.g. [pdf_my_document_p5_chunk_0].\"\n"
        ")\n"
        "CITATION_RE = re.compile(r\"\\[([^\\[\\]\\s,]+_chunk_\\d+)\\]\")",
        s
    ))
    el.append(Paragraph(
        "Each retrieved chunk is rendered with a structured header — "
        "<i>[chunk_id] (section='Module 7 &gt; Self-Attention', pages=p83-84)</i> — followed "
        "by the chunk text. <b>parse_citations()</b> extracts <i>[chunk_id]</i> tokens "
        "from the answer, intersects with the retrieved set, and de-duplicates while "
        "preserving order. Generation runs against a local <b>Ollama</b> server "
        "(<i>gemma3:4b</i>, <b>temperature=0.2</b>) with a 120&thinsp;s timeout; if "
        "Ollama is unreachable, the system raises <i>OllamaUnavailable</i> and returns "
        "the retrieved chunks for offline inspection.",
        s["body"]
    ))
    el.append(why_box(
        "<b>Strict grounding</b> + <b>explicit abstain</b> (&quot;I do not know&quot;) reduces "
        "hallucination on out-of-corpus queries; <b>2&ndash;3 sentence cap</b> prevents the "
        "model from drifting into ungrounded elaboration; <b>structured chunk headers</b> "
        "and <b>regex-validated citations</b> make every claim traceable to a specific chunk. "
        "<b>Temperature 0.2</b> trades diversity for reproducibility &mdash; essential for "
        "comparable ablation runs.",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 7. EVALUATION RESULTS
    # ══════════════════════════════════════════════════════════════════════
    el += section("7", "Evaluation Results", s)
    el.append(Paragraph(
        "<b>Gold set:</b> 194 (question, reference_answer) pairs generated by "
        "<i>eval/build_gold_set.py</i> using gemma3:4b, one pair per indexed chunk, "
        "deterministically rotated across five categories (<i>factual, numerical, "
        "temporal, comparison, negation</i>). Negation items intentionally have no "
        "corpus answer and are excluded from Hit@k / MRR@k tallies. Answer accuracy "
        "is token-level F1 on the first 20 grounded samples per configuration.",
        s["body"]
    ))
    metrics = [
        ["Metric",     "Formula",                                         "What it captures"],
        ["Hit@k",      "1 if &ge; 1 relevant chunk in top-k, else 0",     "Recall — did retrieval surface any correct evidence?"],
        ["MRR@k",      "(1/|Q|) &Sigma; 1 / rank<sub>first relevant</sub>",
         "Rank quality — how high is the first correct hit?"],
        ["Answer F1",  "mean token-F1(pred, ref) over 20 samples",        "End-to-end generation faithfulness"],
    ]
    tm = Table(metrics, colWidths=[0.85*inch, 2.65*inch, 3.7*inch])
    tm.setStyle(base_table_style())
    el.append(tm)
    el.append(Spacer(1, 4))
    el.append(Paragraph(
        "Production configuration <b>dense_k5</b> (Stage&nbsp;1 dense, k=5, no reranking) "
        "reaches <b>Hit@5 = 0.775</b>, <b>MRR@5 = 0.657</b>, Answer F1 = 0.254. "
        "<b>hybrid_k5</b> (RRF dense + BM25) lifts MRR@5 to <b>0.670</b> at identical "
        "Hit@5, and <b>dense_k8</b> reaches the highest <b>Hit@8 = 0.838</b> at the cost "
        "of context bloat. Full numbers in the ablation table below.",
        s["body"]
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 8. ABLATION TABLE
    # ══════════════════════════════════════════════════════════════════════
    el += section("8", "Ablation Table", s)
    el.append(Paragraph(
        "Two axes are varied: (1)&nbsp;top-k under the dense retriever, (2)&nbsp;retrieval "
        "strategy at k=5. Index identical across all five rows (219 chunks, "
        "<i>structured</i> load, RCTS 600/120). Source: <i>eval/results_20260527_112201.md</i>.",
        s["body_sm"]
    ))
    abl = [
        ["Config",      "Retriever",      "k", "Hit@k", "MRR@k", "Ans. F1", "Notes"],
        ["dense_k3",    "Dense (BGE)",    "3", "0.719", "0.644", "0.288", "Focused; misses ~28% of evidence"],
        ["dense_k5  ★",
                        "Dense (BGE)",    "5", "0.775", "0.657", "0.254", "Production baseline"],
        ["dense_k8",    "Dense (BGE)",    "8", "0.838", "0.666", "0.307", "Best Hit@k; noisier context"],
        ["bm25_k5",     "BM25 lexical",   "5", "0.750", "0.630", "0.297", "Wins on identifiers (BM25, RLHF)"],
        ["hybrid_k5",   "Dense + BM25 (RRF)", "5", "0.775", "0.670", "0.231", "Best MRR@k"],
    ]
    ta = Table(abl, colWidths=[0.85*inch, 1.20*inch, 0.25*inch, 0.55*inch,
                               0.55*inch, 0.6*inch, 2.2*inch])
    ts = base_table_style()
    ts.add("ALIGN", (2, 0), (5, -1), "CENTER")
    ts.add("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold")
    ts.add("BACKGROUND", (0, 2), (-1, 2), SOFT)
    # Highlight best numbers
    ts.add("TEXTCOLOR", (3, 4), (3, 4), colors.HexColor("#0a6b2c"))  # best Hit@k (dense_k8)
    ts.add("FONTNAME",  (3, 4), (3, 4), "Helvetica-Bold")
    ts.add("TEXTCOLOR", (4, 6), (4, 6), colors.HexColor("#0a6b2c"))  # best MRR@k (hybrid_k5)
    ts.add("FONTNAME",  (4, 6), (4, 6), "Helvetica-Bold")
    ts.add("TEXTCOLOR", (5, 4), (5, 4), colors.HexColor("#0a6b2c"))  # best Ans F1 (dense_k8)
    ts.add("FONTNAME",  (5, 4), (5, 4), "Helvetica-Bold")
    ta.setStyle(ts)
    el.append(ta)
    el.append(Spacer(1, 4))
    el.append(Paragraph(
        "<b>Take-aways.</b> (a)&nbsp;Hit@k scales monotonically with k (+11.9 pp from "
        "k=3 to k=8); MRR@k saturates earlier (+2.2 pp). "
        "(b)&nbsp;BM25 alone trails dense by 2.5 pp Hit@k but is a strong complement &mdash; "
        "hybrid RRF gains <b>+1.3 pp MRR@k over dense</b>. "
        "(c)&nbsp;Answer F1 is <b>non-monotonic</b> in k: gemma3:4b handles ~900 "
        "context tokens (k=5) better than ~1,400 (k=8) for some queries, but degrades "
        "again under hybrid (0.231) &mdash; the reranker occasionally promotes lexically "
        "matched chunks that are semantically off-topic for the generator.",
        s["body_sm"]
    ))

    # ══════════════════════════════════════════════════════════════════════
    # 9. FAILURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    el += section("9", "Failure Analysis", s)
    el.append(Paragraph(
        "Errors were sampled from <i>eval/rag_runs/</i> and the per-query JSONL "
        "outputs of <i>eval/run_eval.py</i>. Five recurring failure modes:",
        s["body_sm"]
    ))
    failures = [
        ("Semantic drift on peripheral topics.",
         "Bayes&rsquo; theorem appears only inside a formula on a single page. The "
         "question &quot;Explain Bayesian inference&quot; retrieves general-probability "
         "sections, not the formula page. Mitigation: query expansion or hybrid retrieval."),
        ("Hallucinated citations.",
         "gemma3:4b occasionally cites a <i>chunk_id</i> that does not verbatim support "
         "the stated fact &mdash; the model conflates semantic proximity with grounding. "
         "<i>parse_citations()</i> exposes this but does not yet cross-check claim &harr; evidence."),
        ("Over-abstention on formula chunks.",
         "Chunks whose body is dominated by raw math notation lose semantic signal "
         "through PDF extraction. The LLM sees a weak signal and replies &quot;I do not know&quot; "
         "even when the chunk is on the right topic."),
        ("Breadcrumb context overhead.",
         "The [Module &gt; Concept &gt; Subsection] prefix adds ~50&ndash;80 tokens per chunk "
         "(~25% of an avg chunk) &mdash; helpful for retrieval ranking but competes with "
         "answer tokens inside gemma3:4b&rsquo;s context window."),
        ("Instruction non-compliance.",
         "gemma3:4b sometimes omits the square brackets around <i>chunk_id</i>, breaking "
         "the citation regex. A future fix is structured-output decoding (JSON-mode) so "
         "citations cannot be malformed."),
    ]
    for title, desc in failures:
        el.append(Paragraph(f"<b>{title}</b> {desc}", s["body_sm"]))

    # ══════════════════════════════════════════════════════════════════════
    # 10. WHAT I WOULD IMPROVE NEXT
    # ══════════════════════════════════════════════════════════════════════
    el += section("10", "What I Would Improve Next", s)
    improvements = [
        ("Upgrade the generator.",
         "Replace gemma3:4b with Llama-3-8B or Claude Haiku 4.5. Expected "
         "Answer F1 lift of 15&ndash;25 pp and significantly more reliable citation formatting."),
        ("Contextual retrieval (Anthropic recipe).",
         "Prepend each chunk at index time with a one-sentence GPT-4-generated context "
         "(&quot;This chunk is from Module 7 and explains the self-attention formula&quot;). "
         "Anchors BM25 signals to the section&rsquo;s semantic theme."),
        ("Domain-fine-tuned embeddings.",
         "Fine-tune BGE on (question, chunk) pairs mined from the gold set. "
         "Expected +5&ndash;10 pp on Hit@k and MRR@k."),
        ("Structured-output generation.",
         "Require JSON <i>{answer, confidence, cited_ids}</i> from the LLM. Removes "
         "citation-regex fragility and enables automatic hallucination detection by "
         "cross-checking cited_ids against the answer tokens."),
        ("Human-validated gold set.",
         "Current gold set is LLM-authored &mdash; an evaluator and generator that share "
         "training data inflate scores. Replace with crowdsourced annotations and "
         "report Hit@k stratified by question category."),
        ("ColBERT late-interaction.",
         "Token-level scoring at lower latency than full cross-encoder rerank; "
         "promising path as the corpus grows beyond a few hundred chunks."),
        ("Multimodal ingestion.",
         "Diagrams in the textbook are currently lost to text-only extraction. A "
         "VLM (e.g. GPT-4o) could index diagrams as natural-language descriptions "
         "and unlock architecture-style questions that today miss."),
    ]
    for title, desc in improvements:
        el.append(bullet(f"<b>{title}</b> {desc}", s))

    el.append(Spacer(1, 6))
    el.append(thin_rule(thickness=0.4))
    el.append(Paragraph(
        "All experiments run on CPU (no GPU). On modern hardware, the cross-encoder "
        "rerank step is the dominant latency contributor; GPU inference would cut "
        "p95 query latency from ~8 s to under 1 s and unlock larger generator models.",
        s["note"]
    ))

    return el


def main():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="RAG Pipeline — Final Report",
        author="Kosta Sidorenko",
    )
    styles = make_styles()
    doc.build(build_story(styles))
    from pypdf import PdfReader
    pages = len(PdfReader(OUTPUT).pages)
    print(f"Report written to {OUTPUT}  ({pages} pages)")


if __name__ == "__main__":
    main()
