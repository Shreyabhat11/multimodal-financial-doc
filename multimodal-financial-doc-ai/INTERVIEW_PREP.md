# How to Explain This Project in an Interview

## The one-sentence version

"A multimodal document-understanding pipeline where a vision-language model does the
reading, but a deterministic Python layer — not an LLM — decides whether the numbers
are actually correct, and an agent crew provides a second, interpretive opinion on
top of that."

That sentence front-loads the single decision most worth defending: **separating
extraction (probabilistic, VLM-driven) from validation (deterministic where
possible, agentic where judgment is genuinely needed).** Almost every other design
choice in this project follows from that split.

## Talking through each major decision

**Why a multimodal VLM instead of OCR?**
Financial statements are tables with meaningful spatial structure — which column a
number is in determines whether it's a debit or a credit. OCR returns text in
reading order and discards that structure; a VLM reads the layout the way a human
analyst does, understanding headers, alignment, and grouping jointly with the text.
The trade-off: VLMs are slower and costlier per page, and can still fail on very
low-quality scans where OCR's raw character recognition sometimes wins — which is
exactly why this system keeps OCR as a **fallback**, not a competitor: vision-first
for the common case, OCR-based text-structuring specifically for pages where the VLM
signaled low confidence or failed outright.

**Why Qwen-VL and LLaVA specifically, and why both?**
Both are strong, open, actively maintained vision-language models with genuine table
and document understanding, and both are usable either locally (with a GPU) or via
a hosted inference API (with none). Supporting both — behind one `BaseVisionModel`
interface — isn't redundancy for its own sake: different VLM families have different
failure modes on dense tabular content, and having two backends lets you actually
A/B them on your own document distribution via the evaluation harness rather than
picking one on vibes.

**Why LangGraph?**
The pipeline has real branching logic — retry a failed extraction pass, route a
schema-invalid document straight to human review, force human review when
deterministic validation fails regardless of what an agent says — and it needs to
preserve full state through all of that so a document stuck in `needs_human_review`
carries its complete processing history, not just an error message. LangGraph's
typed `StateGraph`, conditional edges, and node-level retry policies are built
exactly for this; hand-rolling equivalent control flow in a linear script would mean
re-deriving state management, retry semantics, and routing logic that LangGraph
already gives you.

**Why CrewAI, and why does LangGraph call it (not the other way around)?**
CrewAI is good at *role-based collaboration* — a small team of agents with distinct
goals reasoning about a bounded task together. It is not a durable, resumable state
machine. So LangGraph owns orchestration (state, retries, routing) and calls CrewAI
as a single node — a specialist it delegates one bounded task to, the way you'd call
any other function. This mirrors how these frameworks are actually combined in
production.

**Why ReAct?**
The validation agents need to check specific facts (does the balance reconcile? are
there duplicates?) rather than reason in the abstract. ReAct-style tool calling gives
them a controlled way to fetch exact answers instead of guessing — and because every
tool is a thin wrapper over the deterministic validation layer, the agent's "actions"
always bottom out in real arithmetic, never in the agent inventing a number.

**Why deterministic financial validation instead of trusting the LLM/agents?**
LLMs are not reliable at arithmetic, and an agent's confidence in a number is not
evidence that the number is correct. Balance reconciliation
(`opening + credits - debits == closing`) is exact, cheap, pure-Python arithmetic —
there's no reason to introduce probabilistic uncertainty into a check that has one
correct answer. This is the single most defensible architectural claim in the
project, and it's not just asserted: `tests/integration/test_langgraph_pipeline.py::
TestFinancialValidationCannotBeOverriddenByCrew` constructs a document where the
mocked CrewAI validation reports everything as "passed," while the (real,
deterministic) financial validation correctly catches a wrong balance — and the
document still routes to human review. The test would fail if an agent's opinion
could override the arithmetic.

**How does confidence scoring work?**
Two signals, combined deliberately differently. Per-field confidence is the mean of
the VLM's own self-reported per-field confidence values, weighted by field (accepted
as a rough calibration signal, not a probability). Then a document-level
**multiplier** — not a blended average — is applied based on the worst deterministic
validation outcome. A `FAILED` reconciliation drags the whole score down even if
every individual field "looked confident," because balance mismatch is a cross-field
consistency signal that per-field self-report structurally cannot see on its own.

**How are hallucinations reduced?**
Three layers: (1) the extraction prompt explicitly instructs the model to return
`null` rather than guess when a field is illegible — an empty field is safe, a
confidently wrong number is not; (2) deterministic validation catches numeric
hallucinations regardless of how confident the model claims to be; (3) the CrewAI
crew provides an independent second read, and its own tools are grounded in the same
deterministic layer rather than free-form reasoning.

**How are multi-page documents handled?**
Each page is processed independently (its own VLM call, its own OCR-fallback
decision), then `merge_page_results` combines them: document-level fields
(account number, balances) take the first non-null value found across pages in
order; transactions concatenate in page order, each tagged with `source_page` for
traceability. A single bad page doesn't cost you the rest of the document — its
failure is recorded in per-page metadata, not treated as a document-wide failure.

**How are structured outputs guaranteed?**
Multiple layers, not one: the VLM prompt requests a specific JSON shape; a
multi-strategy parser (`response_parser.py`) recovers JSON from common
deviations (markdown fences, surrounding prose, trailing commas); Pydantic schemas
validate and coerce every field on the way into the pipeline; and the CrewAI Final
Reviewer's task uses `output_pydantic=CrewValidationOutput`, which makes CrewAI
itself enforce (and retry against) a structured schema for the agent's final output.

**How do agents communicate?**
`Process.sequential` — each task receives the prior tasks' outputs via CrewAI's
`context` parameter. No shared external state store, no agent-to-agent delegation
(`allow_delegation=False` on every agent, deliberately, for predictability and cost
control) — information flows one direction, task to task, ending at a single Final
Reviewer that synthesizes everything into one verdict.

**How are failures handled?**
Every EXPECTED failure (corrupted PDF, VLM call failure, malformed JSON, crew
execution failure) is caught inside its layer and turned into typed state/exception
objects — never allowed to crash the pipeline. A corrupted PDF fails fast in
milliseconds, before a single VLM call. A total extraction failure triggers exactly
one retry of the whole page-extraction pass, then routes to human review with a
precise list of what's missing. Only genuinely unexpected exceptions are allowed to
propagate, caught by a catch-all API handler that never leaks internal detail to
callers.

**How does the system scale?**
The FastAPI layer is stateless and background-task-based, horizontally scalable
behind a load balancer (with local file storage swapped for object storage — the
narrow `file_storage.py` interface makes that a contained change). Each page's VLM
call is independent, making page-level parallelism a natural next optimization. The
CrewAI validation step is the main latency/cost driver per document and is the
obvious first target for caching or batching in a high-volume deployment.

**How is security/privacy handled?**
Account numbers are encrypted at rest (Fernet, key derived from a real secret, not
stored anywhere in plaintext) and masked in every API response and UI view —
verified directly in the test suite, not just asserted in a docstring. No sensitive
data is ever logged. CrewAI's raw reasoning traces never reach application output.
Docker images run as non-root. (No auth layer yet — listed explicitly as a known gap
in the README, not glossed over.)

**How could this be deployed to production?**
Docker Compose is the local/demo deployment; production would move to Kubernetes (or
similar) with the API horizontally scaled, a managed Postgres instance, object
storage for uploads, migrations run as a separate release step rather than on every
container start (already noted in the entrypoint script's own comments), and an
authentication layer in front of every endpoint.

## 20 Likely Interview Questions and Strong Answers

**1. Walk me through what happens when someone uploads a bank statement.**
The file is validated (size, extension) and persisted; a background task starts the
LangGraph pipeline: PDF is rendered to per-page images with rotation auto-corrected,
each page is sent to the VLM independently with a page-scoped extraction prompt,
results are merged, normalized into typed `Transaction` objects, and structurally
validated. If structural validation passes, deterministic financial validation and
anomaly detection run, then a CrewAI crew reviews everything, then a confidence
score is computed. The document lands as `completed` or `needs_human_review`
depending on whether any validator failed or confidence is below threshold.

**2. Why not just ask the VLM to also tell you if the numbers add up?**
Because "does this reconcile" has one objectively correct answer computable in a few
lines of Python, and asking an LLM to do arithmetic introduces failure modes
(hallucinated confidence, inconsistent results across runs) that a deterministic
check simply doesn't have. Reserve the LLM for what it's actually good at —
understanding layout and semantics — and use exact arithmetic for arithmetic.

**3. What happens if the VLM returns malformed JSON?**
`response_parser.py` tries several recovery strategies in order — direct parse,
strip markdown fences, extract the outermost `{...}` object even with surrounding
prose, fix trailing commas — before giving up. If none succeed, that page is marked
failed (not silently dropped), which can trigger the OCR fallback or, if too many
pages fail, one retry of the whole extraction pass.

**4. How do you handle a 50-page statement without one bad page ruining everything?**
Pages are extracted independently; a failed page contributes nothing to the merge
but is recorded in metadata, not treated as a document-wide failure. Only if the
overall page failure rate crosses 50% does the graph retry the whole extraction
pass — an isolated single-page failure just shows up as a gap for a human reviewer
to notice, not a pipeline crash.

**5. Why separate the Pydantic schemas from the SQLAlchemy ORM models?**
They answer different questions. Pydantic schemas describe the shape of data
crossing a boundary — API request/response, VLM output — and are about validation
and parsing. ORM models describe physical storage — foreign keys, column types,
indexes. Conflating them means either your API leaks database constraints or your
database schema gets weakened to whatever's convenient for request validation.
Keeping them separate, with an explicit mapping layer, avoids both.

**6. Why is account_number encrypted at rest instead of just masked at the API layer?**
Masking at the API layer protects against a client seeing the number; it does
nothing against someone with direct database access (a stolen credential, a SQL
injection elsewhere in a larger system). Encrypting the column itself is defense in
depth — the data is genuinely unreadable without the application's secret key, not
just hidden by convention in one code path.

**7. Why Fernet specifically for that encryption?**
It's the standard library's own recommended choice for "encrypt now, decrypt later,
with one shared key" — symmetric, authenticated (a tampered ciphertext fails to
decrypt rather than silently producing garbage), and simple to reason about
correctly, which matters more for a security-relevant primitive than a marginally
more exotic algorithm would.

**8. How would you test this system?**
Layered, matching the architecture: pure functions (parsing, financial math,
anomaly detection) get direct unit tests with hand-constructed cases including
edge cases; the LangGraph pipeline gets integration tests using fake VLM backends
(no network) that exercise real routing decisions, including deliberately breaking
a balance to confirm human-review routing actually fires; the API gets tests via
`TestClient` against a real (if temporary) database; and an evaluation harness with
synthetic ground truth measures actual extraction quality, not just "does it run."

**9. What's the hardest bug you'd guess this project hit during development, and why?**
(Answer from real experience, not hypothetical.) A fresh `pip install` in a clean
environment — different from the incrementally-built dev environment — surfaced
five real transitive dependency conflicts between `langgraph`, `crewai`, and
`pydantic`/`pydantic-settings` pins, plus an entirely unused `crewai-tools`
dependency pulling in an incompatible `chromadb` version via `embedchain`. None of
these showed up until dependency resolution was tested in a genuinely clean
environment — which is exactly the argument for testing your Docker build's
`pip install` step in isolation, not just trusting your dev venv.

**10. How does the OCR fallback actually work?**
It's not a second vision model — it's Tesseract (pixels → raw text) feeding a
*text-only* completion call that structures that raw text into the same JSON shape
the vision-first path produces. Triggered when the VLM call fails outright or scores
below a confidence threshold. If both the vision-first and OCR-fallback attempts
succeed, the one with higher estimated confidence wins — OCR fallback isn't
automatically trusted more just because it ran second.

**11. Why weight `transactions` highest (0.35) in the confidence aggregation?**
Because a wrong transaction list is the most consequential possible extraction
error for this system's actual purpose — a wrong bank name is a cosmetic problem, a
wrong or missing transaction directly corrupts the financial picture the whole
pipeline exists to produce. The weights are a policy choice (configurable, not
hardcoded), and this is the reasoning behind the default.

**12. What would you change if this needed to process 10,000 documents a day?**
Page-level VLM calls are naturally parallelizable — currently sequential per
document, page-level or document-level concurrency would be the first lever.
Object storage instead of local disk for uploads (the codebase already isolates
that behind one narrow module). The CrewAI validation step is the highest per-document
latency/cost item; caching identical-document-hash results or batching would help.
And migrations would move out of the container entrypoint into a proper release
pipeline step, since every replica currently races to run them on startup — fine for
one instance, not for many.

**13. Why does `BaseVisionModel.extract()` distinguish "call failed" from "response failed to parse"?**
Because they warrant different recovery: a call failure (network timeout, API error)
is worth retrying the same call; a parsing failure means the model responded but not
usefully, which might warrant a different prompt strategy or the OCR fallback rather
than blindly retrying an approach that already produced unusable output.

**14. Walk me through the transaction-matching logic in the evaluation framework.**
Two predicted and ground-truth transactions are only match-eligible if they agree on
date and both debit/credit amounts within tolerance — that's the hard criterion.
Among eligible candidates, description-text similarity breaks ties, handling the
case of two same-day, same-amount transactions (two identical coffee purchases) that
date+amount alone can't disambiguate. It never lets description similarity alone
create a false match between transactions that don't actually agree on the hard
criteria.

**15. Why median, not mean, for the large-transaction outlier check?**
Mean is skewed by exactly the kind of large, legitimate transaction (a mortgage
payment on an otherwise low-value account) that would raise the threshold right when
sensitivity matters most. Median is robust to that skew.

**16. What's the Repository pattern doing for you here, concretely?**
It's the only place in the codebase that constructs SQLAlchemy queries — the
service layer calls `DocumentRepository.get_by_id(...)`, never raw ORM queries. If
storage ever changed, only the repository layer would need to change. For a project
this size, it's deliberately NOT a full abstract-interface-per-aggregate pattern —
that would be over-engineering; these are concrete, pragmatic classes.

**17. Why does `get_db` commit explicitly, and why did that matter?**
A generator-based FastAPI dependency that only closes the session (no `commit()`)
silently discards writes on close, since SQLAlchemy's default behavior without an
explicit commit is an implicit rollback. This was a real bug caught by actually
running the API's reprocess flow end-to-end, not by code review — the fix was
adding an explicit commit/rollback in the dependency, plus an explicit commit in the
one route where a `BackgroundTasks` job could otherwise start executing before the
dependency's own cleanup finished.

**18. Why is `crew_verbose=False` the production default?**
CrewAI's verbose mode prints the agents' internal reasoning traces. The brief (and
good practice) requires never exposing raw chain-of-thought in application output —
only the final structured `CrewValidationOutput`. Verbose mode is a local-debugging
tool, not something that should ever be on in a deployment whose output reaches
users or logs.

**19. What's a limitation of this system you'd tell an interviewer unprompted?**
VLM extraction quality has not actually been benchmarked against real (non-synthetic)
scanned statements in this development environment — no GPU or hosted-API access was
available while building it. The evaluation framework itself is real and verified
(deliberately-corrupted fake extractions are correctly penalized by the metrics), but
running it against real models on a real, larger, ideally partially-real document set
is the honest next step, not something this project claims to have already done.

**20. If you had one more week, what would you build next?**
An authentication layer (currently every endpoint is open — a real gap, not a
style choice) and a proper human-review correction UI — right now
`needs_human_review` documents are visible in Streamlit but there's no workflow for
a reviewer to actually correct and re-approve them, which is the natural next step
for the whole confidence-routing system to actually close the loop.
