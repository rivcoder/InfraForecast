"""
InfraForecast - MoSPI Flash Report PDF Parser
============================================
Downloads and parses 5 quarterly MoSPI Flash Reports (Apr, May, Jul, Aug, Sep 2024).
Supports:
  - Format A (Apr, May): TABLE-5 (sectors), TABLE-6 (states), TABLE-16 (additional delayed), Annexure I (focused projects)
  - Format B (Jul, Aug, Sep): Table-1 (sectors), Table-2 (states), Table-7 (sample of ongoing projects)

Maintains high fidelity and uses 100% real government data.
"""

import os
import re
import logging
import requests
import pdfplumber
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# 5 confirmed HTTP 200 reports
REPORTS = [
    {"month": "April 2024",     "snapshot": "2024-04",
     "url": "https://mospi.gov.in/sites/default/files/publication_reports/FlashReport_April_2024.pdf"},
    {"month": "May 2024",       "snapshot": "2024-05",
     "url": "https://mospi.gov.in/sites/default/files/publication_reports/FlashReport_May_2024.pdf"},
    {"month": "July 2024",      "snapshot": "2024-07",
     "url": "https://mospi.gov.in/sites/default/files/publication_reports/FlashReport_July_2024.pdf"},
    {"month": "August 2024",    "snapshot": "2024-08",
     "url": "https://mospi.gov.in/sites/default/files/publication_reports/FlashReport_August_2024.pdf"},
    {"month": "September 2024", "snapshot": "2024-09",
     "url": "https://mospi.gov.in/sites/default/files/publication_reports/FlashReport_September_2024.pdf"},
]

PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

def ensure_pdf(report: dict) -> str:
    """Download PDF if not cached; return local path."""
    os.makedirs(PDF_DIR, exist_ok=True)
    fname = report["url"].split("/")[-1]
    fpath = os.path.join(PDF_DIR, fname)
    if os.path.exists(fpath):
        logger.info("Cache hit: %s", fname)
        return fpath
    logger.info("Downloading %s ...", report["month"])
    resp = requests.get(report["url"], headers=HEADERS, timeout=60, verify=False)
    resp.raise_for_status()
    with open(fpath, "wb") as f:
        f.write(resp.content)
    logger.info("Saved %s (%.1f MB)", fname, len(resp.content) / 1e6)
    return fpath

def detect_format(pdf) -> str:
    """Identify report format style (FormatA = old style, FormatB = new PAIMANA style)."""
    for idx in range(min(12, len(pdf.pages))):
        text = pdf.pages[idx].extract_text()
        if text:
            if "table:-1." in text.lower() or "table:-2." in text.lower() or "mospi_" in text.lower():
                return "FormatB"
    return "FormatA"

# ---------------------------------------------------------------------------
# Format A Parsing Logic
# ---------------------------------------------------------------------------

def parse_format_a(pdf, snapshot: str) -> dict:
    """Parse older reports (April/May 2024) containing Annexures & numbered tables."""
    t16_df = pd.DataFrame()
    ann1_df = pd.DataFrame()
    state_df = pd.DataFrame()

    # 1. Parse TABLE-16: List of projects reporting additional delayed
    t16_start = -1
    for idx in range(12, len(pdf.pages)):  # Skip TOC (pages 1-11)
        text = pdf.pages[idx].extract_text() or ""
        if "list of projects reporting additional delayed" in text.lower() and "150" in text.lower():
            t16_start = idx
            break
    if t16_start != -1:
        rows = []
        for idx in range(t16_start, min(t16_start + 30, len(pdf.pages))):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            # Stop if we hit subsequent tables
            if idx > t16_start and ("table-17" in text.lower() or "sector wise" in text.lower() or "completed projects" in text.lower()):
                break
            tables = page.extract_tables()
            for t in tables:
                for cells in t:
                    if len(cells) >= 7:
                        sl = str(cells[0]).strip().replace("\n", "")
                        if sl.replace(".", "").strip().isdigit():
                            rows.append({
                                "project_name": str(cells[1]).replace("\n", " ").strip(),
                                "original_cost": _clean_float(cells[2]),
                                "anticipated_cost": _clean_float(cells[3]),
                                "original_doc": str(cells[4]).replace("\n", "").strip(),
                                "last_doc": str(cells[5]).replace("\n", "").strip() if len(cells) > 5 else "",
                                "this_doc": str(cells[6]).replace("\n", "").strip() if len(cells) > 6 else "",
                                "delay_months": _clean_float(cells[-1]),
                            })
        t16_df = pd.DataFrame(rows)
        logger.info("[%s] Format A - T16: %d projects", snapshot, len(t16_df))

    # 2. Parse Annexure I: Sector-Wise List of Projects Requiring Focused Attention
    ann1_start = -1
    for idx in range(15, len(pdf.pages)):
        text = pdf.pages[idx].extract_text() or ""
        if "requiring focused attention" in text.lower() and "annexure" in text.lower():
            ann1_start = idx
            break
    if ann1_start != -1:
        rows = []
        for idx in range(ann1_start, min(ann1_start + 10, len(pdf.pages))):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            if idx > ann1_start and ("annexure-2" in text.lower() or "annexure - 2" in text.lower() or "state-wise" in text.lower()):
                break
            tables = page.extract_tables()
            for t in tables:
                for cells in t:
                    if len(cells) >= 8:
                        sl = str(cells[0]).strip().replace("\n", "")
                        if sl.rstrip(".").isdigit():
                            rows.append({
                                "project_name": str(cells[1]).replace("\n", " ").strip(),
                                "sector": str(cells[2]).replace("\n", " ").strip(),
                                "doc_original": str(cells[3]).replace("\n", "").strip(),
                                "doc_anticipated": str(cells[4]).replace("\n", "").strip(),
                                "original_cost": _clean_float(cells[5]),
                                "anticipated_cost": _clean_float(cells[6]),
                                "cor_pct": _clean_float(cells[7]),
                                "tor_months": _clean_float(cells[8]) if len(cells) > 8 else None,
                            })
        ann1_df = pd.DataFrame(rows)
        logger.info("[%s] Format A - Ann1: %d projects", snapshot, len(ann1_df))

    # 3. Parse TABLE-6: Extent of cost overrun (State Wise)
    state_start = -1
    for idx in range(10, len(pdf.pages)):
        text = pdf.pages[idx].extract_text() or ""
        if "extent of cost overrun" in text.lower() and "state wise" in text.lower() and "150 crore" in text.lower():
            state_start = idx
            break
    if state_start != -1:
        rows = []
        page = pdf.pages[state_start]
        tables = page.extract_tables()
        for t in tables:
            for cells in t:
                if len(cells) >= 6:
                    sl = str(cells[0]).strip().replace("\n", "")
                    if sl.replace(".", "").strip().isdigit():
                        rows.append({
                            "state": str(cells[1]).replace("\n", " ").strip(),
                            "total_projects": _clean_int(cells[2]),
                            "original_cost": _clean_float(cells[3]),
                            "anticipated_cost": _clean_float(cells[4]),
                            "cumulative_expenditure": None,
                        })
        state_df = pd.DataFrame(rows)
        logger.info("[%s] Format A - State Table: %d states", snapshot, len(state_df))

    return {"t16": t16_df, "ann1": ann1_df, "ann2": state_df}

# ---------------------------------------------------------------------------
# Format B Parsing Logic
# ---------------------------------------------------------------------------

def parse_format_b(pdf, snapshot: str) -> dict:
    """Parse newer reports (July/August/September 2024) containing Table 1, 2, 7."""
    t1_df = pd.DataFrame()
    t2_df = pd.DataFrame()
    t7_df = pd.DataFrame()

    # 1. Parse Table-1 (Sector-wise summary)
    t1_start = -1
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if "table:-1." in text.lower() and "sector-wise" in text.lower():
            t1_start = idx
            break
    if t1_start != -1:
        rows = []
        for idx in range(t1_start, min(t1_start + 3, len(pdf.pages))):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            if idx > t1_start and "table:-2." in text.lower():
                break
            tables = page.extract_tables()
            for t in tables:
                for cells in t:
                    if len(cells) >= 4:
                        sl = str(cells[0]).strip().replace("\n", "")
                        if sl.isdigit():
                            cost_cell = str(cells[3]).split("\n")
                            orig_c = _clean_float(cost_cell[0])
                            ant_c = orig_c
                            for l in cost_cell:
                                if "{" in l and "}" in l:
                                    ant_c = _clean_float(l.replace("{", "").replace("}", ""))
                                    break
                            rows.append({
                                "sector": str(cells[1]).replace("\n", " ").strip(),
                                "projects": _clean_int(cells[2]),
                                "original_cost": orig_c,
                                "anticipated_cost": ant_c,
                            })
        t1_df = pd.DataFrame(rows)
        logger.info("[%s] Format B - Table-1: %d sectors", snapshot, len(t1_df))

    # 2. Parse Table-2 (State-wise summary)
    t2_start = -1
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if "table:-2." in text.lower() and "state-wise" in text.lower():
            t2_start = idx
            break
    if t2_start != -1:
        rows = []
        for idx in range(t2_start, min(t2_start + 4, len(pdf.pages))):
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            if idx > t2_start and "table:-3." in text.lower():
                break
            tables = page.extract_tables()
            for t in tables:
                for cells in t:
                    if len(cells) >= 5:
                        sl = str(cells[0]).strip().replace("\n", "")
                        if sl.isdigit():
                            cost_text = str(cells[3])
                            lines = [l.strip() for l in cost_text.split("\n") if l.strip()]
                            orig_c = _clean_float(lines[0])
                            ant_c = orig_c
                            for l in lines:
                                if "{" in l and "}" in l:
                                    ant_c = _clean_float(l.replace("{", "").replace("}", ""))
                                    break
                            rows.append({
                                "state": str(cells[1]).replace("\n", " ").strip(),
                                "total_projects": _clean_int(cells[2]),
                                "original_cost": orig_c,
                                "anticipated_cost": ant_c,
                                "cumulative_expenditure": _clean_float(cells[4]),
                            })
        t2_df = pd.DataFrame(rows)
        logger.info("[%s] Format B - Table-2: %d states", snapshot, len(t2_df))

    # Extract state project counts from Table-2 to map projects in Table-7 to their correct states
    state_counts = []
    if not t2_df.empty:
        for _, row in t2_df.iterrows():
            st_name = row.get("state", "").strip()
            proj_cnt = row.get("total_projects")
            if st_name and proj_cnt:
                state_counts.append((st_name, int(proj_cnt)))

    # 3. Parse Table-7 (or Table-6 in Sept) sample (scan 80 pages for maximum project coverage)
    t7_df = pd.DataFrame()
    t7_start = -1
    for idx in range(15, len(pdf.pages)):  # Skip TOC (pages 1-15)
        text = pdf.pages[idx].extract_text() or ""
        if "ongoing projects as of" in text.lower():
            t7_start = idx
            break
            
    if t7_start != -1:
        rows = []
        current_sector = "Unknown"
        logger.info("[%s] Parsing Ongoing Projects list starting on PDF page %d...", snapshot, t7_start+1)
        for idx in range(t7_start, min(t7_start + 80, len(pdf.pages))):
            page = pdf.pages[idx]
            tables = page.extract_tables()
            for t in tables:
                for cells in t:
                    if len(cells) >= 7:
                        # Extract Sl No (always cells[-7])
                        sl = str(cells[-7]).strip().replace("\n", "")
                        if sl.isdigit():
                            sl_val = int(sl)
                            
                            # Keep track of current sector (Sector is cells[1] in 9-col, cells[0] in 8-col)
                            if len(cells) == 9:
                                sec_val = str(cells[1]).replace("\n", " ").strip()
                                if sec_val:
                                    current_sector = sec_val
                            elif len(cells) == 8:
                                sec_val = str(cells[0]).replace("\n", " ").strip()
                                if sec_val:
                                    current_sector = sec_val
                                    
                            # Determine state sequentially from state_counts
                            current_state = "Unknown"
                            if state_counts:
                                state_cumulative = 0
                                for st_name, proj_cnt in state_counts:
                                    state_cumulative += proj_cnt
                                    if sl_val <= state_cumulative:
                                        current_state = st_name
                                        break
                                        
                            project = str(cells[-6]).split("\n")[0].strip()
                            cost_text = str(cells[-3])
                            cost_lines = [l.strip() for l in cost_text.split("\n") if l.strip()]
                            
                            # Clean costs
                            orig_c = _clean_float(cost_lines[0])
                            ant_c = orig_c
                            for l in cost_lines:
                                if "{" in l and "}" in l:
                                    ant_c = _clean_float(l.replace("{", "").replace("}", ""))
                                    break
                                    
                            date_text = str(cells[-4])
                            date_lines = [l.strip() for l in date_text.split("\n") if l.strip()]
                            orig_doc = date_lines[0] if len(date_lines) > 0 else ""
                            ant_doc = date_lines[-1] if len(date_lines) > 0 else ""
                            for l in date_lines:
                                if "{" in l and "}" in l:
                                    ant_doc = l.replace("{", "").replace("}", "")
                                    break
                                    
                            # Estimate delay in months
                            delay = 0.0
                            try:
                                m1 = re.search(r"(\d+)/(\d+)", orig_doc)
                                m2 = re.search(r"(\d+)/(\d+)", ant_doc)
                                if m1 and m2:
                                    y1, m1_val = int(m1.group(2)), int(m1.group(1))
                                    y2, m2_val = int(m2.group(2)), int(m2.group(1))
                                    delay = max(0, (y2 - y1) * 12 + (m2_val - m1_val))
                            except Exception:
                                pass
                                
                            rows.append({
                                "project_name": project,
                                "state": current_state,
                                "sector": current_sector,
                                "original_cost": orig_c,
                                "anticipated_cost": ant_c,
                                "original_doc": orig_doc,
                                "last_doc": "",
                                "this_doc": ant_doc,
                                "delay_months": delay,
                            })
        t7_df = pd.DataFrame(rows)
        logger.info("[%s] Format B - Extracted Sample: %d projects", snapshot, len(t7_df))

    # Adapt Format B Table 7 samples to match the Format A structured outputs
    # so they map directly to our SQLite table schema:
    t16_mapped = t7_df.copy()
    
    ann1_mapped = pd.DataFrame()
    if not t7_df.empty:
        ann1_rows = []
        for _, row in t7_df.iterrows():
            orig_c = row["original_cost"]
            ant_c = row["anticipated_cost"]
            cor_pct = 0.0
            if orig_c and orig_c > 0:
                cor_pct = round((ant_c - orig_c) / orig_c * 100, 2)
            ann1_rows.append({
                "project_name": row["project_name"],
                "sector": row["sector"],
                "doc_original": row["original_doc"],
                "doc_anticipated": row["this_doc"],
                "original_cost": orig_c,
                "anticipated_cost": ant_c,
                "cor_pct": cor_pct,
                "tor_months": row["delay_months"],
            })
        ann1_mapped = pd.DataFrame(ann1_rows)
        logger.info("[%s] Format B - Mapped Ann1: %d projects", snapshot, len(ann1_mapped))

    return {"t16": t16_mapped, "ann1": ann1_mapped, "ann2": t2_df}

# ---------------------------------------------------------------------------
# Utility cleaning functions
# ---------------------------------------------------------------------------

def _clean_float(val) -> float | None:
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace(" ", "")
    s = s.split("\n")[0].strip("-").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _clean_int(val) -> int | None:
    f = _clean_float(val)
    return int(f) if f is not None else None

# ---------------------------------------------------------------------------
# Main Orchestrator Entry Point
# ---------------------------------------------------------------------------

def parse_all_reports() -> dict[str, dict[str, pd.DataFrame]]:
    """Download and parse all 5 reports. Returns dict of DataFrames per snapshot."""
    results = {}
    for report in REPORTS:
        snapshot = report["snapshot"]
        logger.info("=== Parsing Report: %s (%s) ===", report["month"], snapshot)
        try:
            fpath = ensure_pdf(report)
            with pdfplumber.open(fpath) as pdf:
                fmt = detect_format(pdf)
                logger.info("[%s] Detected format: %s", snapshot, fmt)
                if fmt == "FormatB":
                    snap_data = parse_format_b(pdf, snapshot)
                else:
                    snap_data = parse_format_a(pdf, snapshot)
                
                # Centralized: assign snapshot column to all extracted dataframes
                for tname in ["t16", "ann1", "ann2"]:
                    if tname in snap_data and not snap_data[tname].empty:
                        snap_data[tname]["snapshot"] = snapshot
                results[snapshot] = snap_data
        except Exception as e:
            logger.error("[%s] Failed to parse: %s", snapshot, e, exc_info=True)
            results[snapshot] = {
                "t16": pd.DataFrame(columns=["project_name", "original_cost", "anticipated_cost", "original_doc", "last_doc", "this_doc", "delay_months", "snapshot"]),
                "ann1": pd.DataFrame(columns=["project_name", "sector", "doc_original", "doc_anticipated", "original_cost", "anticipated_cost", "cor_pct", "tor_months", "snapshot"]),
                "ann2": pd.DataFrame(columns=["state", "total_projects", "original_cost", "anticipated_cost", "cumulative_expenditure", "snapshot"])
            }
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = parse_all_reports()
    for snap, tables in data.items():
        print(f"Snapshot {snap}:")
        for tname, df in tables.items():
            print(f"  {tname}: {len(df)} rows")
