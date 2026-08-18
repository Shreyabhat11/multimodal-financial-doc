"""
Streamlit dashboard for the multimodal financial document pipeline.

Run with: streamlit run frontend/streamlit_app.py

Architecture note: this app is a thin CLIENT of the FastAPI backend (via
api_client.ApiClient) — it does not import the LangGraph pipeline, VLM models, or
database layer directly. This mirrors real production separation (a UI service and
an API service are typically separate deployables) and means the Streamlit app has
zero GPU/heavy-ML dependencies of its own; it only needs `streamlit`, `requests`,
`pandas`, and (for the local page-preview feature below) `pymupdf`/`pillow`, which
this project already depends on for the backend anyway.

Page preview is the one place this app reaches into pipeline code directly
(app.document_processing.pdf_loader.render_pdf_pages) rather than going through the
API — deliberately: the API doesn't store or serve page images (only page
metadata), so rendering a quick low-DPI thumbnail locally from the bytes the user
just selected, before upload, is both simpler and faster than adding an
image-serving endpoint to the API for a preview-only feature.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for app.* imports

from api_client import ApiClient, ApiError  # noqa: E402
from formatting import (  # noqa: E402
    format_confidence,
    format_currency,
    is_terminal_status,
    severity_badge,
    status_badge,
    validation_status_badge,
)

st.set_page_config(page_title="Financial Document AI", page_icon="📄", layout="wide")

DEFAULT_API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 2.0
MAX_PREVIEW_PAGES = 5


def _get_client() -> ApiClient:
    base_url = st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
    return ApiClient(base_url=base_url)


def _render_page_previews(file_bytes: bytes) -> None:
    """Render low-DPI page thumbnails locally, reusing the same PDF-rendering code
    the backend uses (app.document_processing.pdf_loader) rather than a separate
    implementation — one rendering code path, even though this call site is local
    rather than server-side."""
    try:
        from app.document_processing.pdf_loader import render_pdf_pages
    except ImportError:
        st.info("Page preview requires the `app` package to be importable from this environment.")
        return

    try:
        pages = render_pdf_pages(file_bytes, dpi=80)
    except Exception as exc:
        st.warning(f"Could not render page previews: {exc}")
        return

    st.caption(f"{len(pages)} page(s) detected")
    preview_pages = pages[:MAX_PREVIEW_PAGES]
    cols = st.columns(len(preview_pages))
    for col, (page_number, image, _, _, _) in zip(cols, preview_pages):
        with col:
            st.image(image, caption=f"Page {page_number}", use_container_width=True)
    if len(pages) > MAX_PREVIEW_PAGES:
        st.caption(f"... and {len(pages) - MAX_PREVIEW_PAGES} more page(s)")


def _render_account_section(result: dict) -> None:
    account = result.get("account") or {}
    period = result.get("statement_period") or {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Account Holder", account.get("account_holder", "—"))
    col2.metric("Account Number", account.get("account_number", "—"))
    col3.metric("Bank", account.get("bank_name", "—") or "—")

    col4, col5, col6 = st.columns(3)
    col4.metric("Statement Period", f"{period.get('start_date', '—')} to {period.get('end_date', '—')}")
    col5.metric("Opening Balance", format_currency(result.get("opening_balance"), result.get("currency", "USD")))
    col6.metric("Closing Balance", format_currency(result.get("closing_balance"), result.get("currency", "USD")))


def _render_transactions_section(result: dict) -> None:
    transactions = result.get("transactions") or []
    if not transactions:
        st.info("No transactions extracted.")
        return

    df = pd.DataFrame(transactions)
    display_columns = [c for c in ("date", "description", "reference", "debit", "credit", "balance") if c in df.columns]
    df_display = df[display_columns].copy()
    for col in ("debit", "credit", "balance"):
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda v: format_currency(v, result.get("currency", "USD")) if v not in (None, "0", 0) else "—"
            )
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    total_debits = sum(float(t.get("debit") or 0) for t in transactions)
    total_credits = sum(float(t.get("credit") or 0) for t in transactions)
    col1, col2, col3 = st.columns(3)
    col1.metric("Transaction Count", len(transactions))
    col2.metric("Total Debits", format_currency(total_debits, result.get("currency", "USD")))
    col3.metric("Total Credits", format_currency(total_credits, result.get("currency", "USD")))


def _render_validation_section(result: dict) -> None:
    validation_results = result.get("validation_results") or []
    if not validation_results:
        st.info("No validation results available yet.")
        return

    for vr in validation_results:
        with st.expander(f"{vr['validator_name']} — {validation_status_badge(vr['status'])}", expanded=vr["status"] != "passed"):
            if vr.get("checks_performed"):
                st.caption("Checks performed: " + ", ".join(vr["checks_performed"]))
            issues = vr.get("issues") or []
            if issues:
                for issue in issues:
                    st.markdown(f"- {severity_badge(issue['severity'])} **{issue['field']}**: {issue['message']}")
            else:
                st.caption("No issues found.")
            st.caption(f"Recommendation: **{vr.get('recommendation', 'n/a')}**")


def _render_confidence_section(result: dict) -> None:
    overall = result.get("overall_confidence")
    st.metric("Overall Confidence", format_confidence(overall))
    if overall is not None:
        st.progress(min(max(overall, 0.0), 1.0))


def _upload_and_poll(client: ApiClient, file_bytes: bytes, filename: str) -> None:
    with st.spinner("Uploading document..."):
        try:
            upload_result = client.upload_document(file_bytes=file_bytes, filename=filename)
        except ApiError as exc:
            st.error(f"Upload failed: {exc.message} ({exc.error_code})")
            return

    document_id = upload_result["document_id"]
    st.session_state["document_id"] = document_id
    st.success(f"Document submitted. ID: `{document_id}`")

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    max_polls = 60
    for i in range(max_polls):
        try:
            status = client.get_status(document_id)
        except ApiError as exc:
            status_placeholder.error(f"Error checking status: {exc.message}")
            return

        status_placeholder.markdown(f"**Status:** {status_badge(status['status'])}")
        progress_bar.progress(min((i + 1) / max_polls, 0.95))

        if is_terminal_status(status["status"]):
            progress_bar.progress(1.0)
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        st.warning("Processing is taking longer than expected. Check back later or refresh.")


def main() -> None:
    st.title("📄 Multimodal Financial Document AI")
    st.caption("Upload a bank statement, credit card statement, invoice, or loan statement for extraction.")

    with st.sidebar:
        st.header("Settings")
        st.session_state["api_base_url"] = st.text_input(
            "API base URL", value=st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)
        )
        client = _get_client()
        if st.button("Check API health"):
            try:
                health = client.health_check()
                st.success(f"API is healthy (version {health.get('version', '?')})")
            except Exception as exc:  # noqa: BLE001 — surfacing any connectivity issue is the whole point here
                st.error(f"API unreachable: {exc}")

        st.divider()
        st.subheader("Existing document")
        lookup_id = st.text_input("Document ID", value=st.session_state.get("document_id", ""))
        if st.button("Load document") and lookup_id:
            st.session_state["document_id"] = lookup_id
            st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF statement", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        with st.expander("Page previews", expanded=True):
            _render_page_previews(file_bytes)

        if st.button("Start Processing", type="primary"):
            _upload_and_poll(_get_client(), file_bytes, uploaded_file.name)

    document_id = st.session_state.get("document_id")
    if not document_id:
        st.info("Upload a document above, or enter an existing Document ID in the sidebar, to see results here.")
        return

    st.divider()
    st.subheader(f"Results for `{document_id}`")

    client = _get_client()
    try:
        status = client.get_status(document_id)
    except ApiError as exc:
        st.error(f"Could not load document: {exc.message} ({exc.error_code})")
        return

    st.markdown(f"**Status:** {status_badge(status['status'])}")
    if status.get("error_message"):
        st.error(status["error_message"])

    col_a, col_b = st.columns(2)
    if col_a.button("🔄 Refresh"):
        st.rerun()
    if col_b.button("♻️ Reprocess document"):
        try:
            client.reprocess_document(document_id)
            st.success("Reprocessing queued.")
        except ApiError as exc:
            st.error(f"Reprocess failed: {exc.message}")

    if not is_terminal_status(status["status"]):
        st.info("Still processing — click Refresh to check again.")
        return

    try:
        result = client.get_result(document_id)
    except ApiError as exc:
        st.error(f"Could not load result: {exc.message}")
        return

    tabs = st.tabs(["Account & Totals", "Transactions", "Validation", "Confidence", "Raw JSON"])

    with tabs[0]:
        _render_account_section(result)

    with tabs[1]:
        _render_transactions_section(result)

    with tabs[2]:
        _render_validation_section(result)

    with tabs[3]:
        _render_confidence_section(result)

    with tabs[4]:
        raw_json = json.dumps(result, indent=2)
        st.download_button(
            "Download JSON",
            data=raw_json,
            file_name=f"{document_id}.json",
            mime="application/json",
        )
        st.json(result)


if __name__ == "__main__":
    main()
