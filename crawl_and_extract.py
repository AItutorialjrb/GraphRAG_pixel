import os
import re
import json
import time
import pathlib
import requests
from typing import Dict, List, Optional, Tuple

import pymupdf4llm
from tqdm import tqdm


# =========================================================
# 0. 配置
# =========================================================

BASE_DIR = "/eos/home-r/rjiang/RAG_pixel"

PDF_DIR = os.path.join(BASE_DIR, "main_pdf")
MD_DIR = os.path.join(BASE_DIR, "main_markdown")
META_DIR = os.path.join(BASE_DIR, "meta")
LOG_DIR = os.path.join(BASE_DIR, "logs")

INSPIRE_API = "https://inspirehep.net/api/literature"

MAX_RECORDS = 5000
PAGE_SIZE = 100

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.4

SEARCH_QUERY = (
    '(abstract:"silicon pixel" OR titles.title:"silicon pixel" OR '
    'texkeys:"silicon pixel" OR keywords.value:"silicon") '
    'AND (collection:Published OR publication_info.journal_title:*) '
    'AND (date > 2004)'
   
  
 



)

MIN_MARKDOWN_LEN = 200


# =========================================================
# 1. 工具函数
# =========================================================

def ensure_dirs() -> None:
    for d in [PDF_DIR, MD_DIR, META_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def sanitize_filename(text: str, max_len: int = 140) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-\.]+", "", text)
    return text[:max_len] if len(text) > max_len else text


def safe_get(dct: Dict, path: List[str], default=None):
    cur = dct
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def pick_best_title(metadata: Dict) -> str:
    titles = metadata.get("titles", [])
    if titles and isinstance(titles, list):
        first = titles[0]
        if isinstance(first, dict) and "title" in first:
            return first["title"]
    return "untitled"


def pick_arxiv_id(metadata: Dict) -> Optional[str]:
    eprints = metadata.get("arxiv_eprints", [])
    if eprints and isinstance(eprints, list):
        first = eprints[0]
        if isinstance(first, dict):
            return first.get("value")
    return None


def pick_doi(metadata: Dict) -> Optional[str]:
    dois = metadata.get("dois", [])
    if dois and isinstance(dois, list):
        first = dois[0]
        if isinstance(first, dict):
            return first.get("value")
    return None


def pick_year(metadata: Dict) -> Optional[int]:
    pub_info = metadata.get("publication_info", [])
    if pub_info and isinstance(pub_info, list):
        for item in pub_info:
            if isinstance(item, dict) and "year" in item:
                return item["year"]
    if "preprint_date" in metadata:
        m = re.match(r"(\d{4})", str(metadata["preprint_date"]))
        if m:
            return int(m.group(1))
    return None


def build_pdf_url_from_arxiv(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_text_len(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return len(f.read().strip())


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =========================================================
# 2. INSPIRE 检索
# =========================================================

def search_inspire(query: str, size: int = 100, page: int = 1) -> Dict:
    params = {
        "q": query,
        "size": size,
        "page": page,
        "sort": "mostrecent",
        "fields": "titles,abstracts,authors,arxiv_eprints,dois,publication_info,keywords,control_number"
    }
    resp = requests.get(INSPIRE_API, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def collect_records(query: str, max_records: int) -> List[Dict]:
    results = []
    page = 1

    while len(results) < max_records:
        data = search_inspire(query=query, size=PAGE_SIZE, page=page)
        hits = safe_get(data, ["hits", "hits"], [])

        if not hits:
            break

        for hit in hits:
            results.append(hit)
            if len(results) >= max_records:
                break

        print(f"[INFO] collected {len(results)} records...")
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return results[:max_records]


# =========================================================
# 3. PDF 下载
# =========================================================

def download_pdf(url: str, out_path: str) -> bool:
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type and "application/octet-stream" not in content_type:
                print(f"[WARN] not a PDF content-type: {content_type} | {url}")

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"[ERROR] download failed: {url} | {e}")
        return False


# =========================================================
# 4. PDF -> Markdown
# =========================================================

def extract_markdown(pdf_path: str, md_path: str) -> Tuple[bool, str]:
    """
    只使用 pymupdf4llm，不依赖 fitz。
    """
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        md_text = clean_text(md_text)

        if len(md_text) < MIN_MARKDOWN_LEN:
            return False, f"markdown_too_short: {len(md_text)}"

        pathlib.Path(md_path).write_text(md_text, encoding="utf-8")
        return True, "ok:pymupdf4llm"
    except Exception as e:
        return False, f"pymupdf4llm_failed: {e}"


# =========================================================
# 5. 主流程
# =========================================================

def process_one_record(hit: Dict) -> Dict:
    metadata = hit.get("metadata", {})
    recid = hit.get("id", None) or metadata.get("control_number", None)

    title = pick_best_title(metadata)
    arxiv_id = pick_arxiv_id(metadata)
    doi = pick_doi(metadata)
    year = pick_year(metadata)

    base_name_parts = []
    if year:
        base_name_parts.append(str(year))
    if recid:
        base_name_parts.append(f"recid_{recid}")
    if arxiv_id:
        base_name_parts.append(arxiv_id.replace("/", "_"))
    base_name_parts.append(sanitize_filename(title, max_len=90))

    base_name = "__".join(base_name_parts)

    pdf_path = os.path.join(PDF_DIR, f"{base_name}.pdf")
    md_path = os.path.join(MD_DIR, f"{base_name}.md")
    meta_path = os.path.join(META_DIR, f"{base_name}.json")

    out = {
        "recid": recid,
        "title": title,
        "year": year,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "pdf_path": pdf_path,
        "md_path": md_path,
        "meta_path": meta_path,
        "download_ok": False,
        "extract_ok": False,
        "extract_msg": "",
    }

    if not os.path.exists(pdf_path):
        if arxiv_id:
            pdf_url = build_pdf_url_from_arxiv(arxiv_id)
            ok = download_pdf(pdf_url, pdf_path)
            out["download_ok"] = ok
        else:
            out["download_ok"] = False
            out["extract_msg"] = "no arxiv id, skipped"
    else:
        out["download_ok"] = True

    if out["download_ok"] and os.path.exists(pdf_path):
        if not os.path.exists(md_path) or read_text_len(md_path) < MIN_MARKDOWN_LEN:
            ok, msg = extract_markdown(pdf_path, md_path)
            out["extract_ok"] = ok
            out["extract_msg"] = msg
        else:
            out["extract_ok"] = True
            out["extract_msg"] = "markdown already exists"

    full_meta = {
        "local_info": out,
        "inspire_hit": hit
    }
    save_json(full_meta, meta_path)

    return out


def main():
    ensure_dirs()

    print("[INFO] Searching INSPIRE...")
    hits = collect_records(SEARCH_QUERY, MAX_RECORDS)

    raw_hits_path = os.path.join(LOG_DIR, "inspire_hits_raw.json")
    save_json({"query": SEARCH_QUERY, "hits": hits}, raw_hits_path)

    print(f"[INFO] total hits collected for processing: {len(hits)}")

    processed = []
    failed = []

    for hit in tqdm(hits, desc="Processing papers"):
        try:
            result = process_one_record(hit)
            processed.append(result)

            if not result["download_ok"] or not result["extract_ok"]:
                failed.append(result)

            time.sleep(SLEEP_BETWEEN_REQUESTS)
        except Exception as e:
            bad = {
                "error": str(e),
                "hit_id": hit.get("id"),
                "title": pick_best_title(hit.get("metadata", {}))
            }
            failed.append(bad)

    save_json(processed, os.path.join(LOG_DIR, "processed_summary.json"))
    save_json(failed, os.path.join(LOG_DIR, "failed_summary.json"))

    n_pdf = len([x for x in processed if x.get("download_ok")])
    n_md = len([x for x in processed if x.get("extract_ok")])

    print("\n===== DONE =====")
    print(f"Processed records : {len(processed)}")
    print(f"Downloaded PDFs   : {n_pdf}")
    print(f"Extracted MD      : {n_md}")
    print(f"Failed / partial  : {len(failed)}")
    print(f"PDF dir           : {PDF_DIR}")
    print(f"Markdown dir      : {MD_DIR}")
    print(f"Metadata dir      : {META_DIR}")


if __name__ == "__main__":
    main()
