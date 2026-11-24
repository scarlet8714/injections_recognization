"""Look up OCR label evidence in the local medicine SQLite database."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime
from pathlib import Path

DEFAULT_DATABASE = Path(__file__).resolve().parent / "sql" / "medicine.sqlite3"
MIN_CONFIDENCE = 0.8
MEDICINES = "SELECT id, name_zh, name_en, strength_text, company_text, license_no, barcodes FROM medicines"
LICENSE = re.compile(r"(衛署|衛部)?藥([製輸])字第?([0-9]{6})(?![0-9])號?")


def compact(text: str | None) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text or "") if char.isalnum())


def english_words(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text or "").casefold()))


def medicine(row: sqlite3.Row) -> dict:
    return {**row, "barcodes": json.loads(row["barcodes"])}


def lookup_licenses(db: sqlite3.Connection, lines: list[dict]) -> tuple[list[dict], list[dict]]:
    evidence = []
    candidates = {}
    for line in lines:
        for match in LICENSE.finditer(compact(line["text"])):
            issuer, kind, number = match.groups()
            suffix = f"藥{kind}字第{number}號"
            license_numbers = [issuer + suffix] if issuer else ["衛署" + suffix, "衛部" + suffix]
            placeholders = ", ".join("?" for _ in license_numbers)
            rows = db.execute(f"{MEDICINES} WHERE license_no IN ({placeholders}) ORDER BY id", license_numbers)
            ids = []
            for row in rows:
                ids.append(row["id"])
                candidates[row["id"]] = medicine(row)
            evidence.append({**line, "parsed_license": match.group(), "queried_licenses": license_numbers,
                             "candidate_ids": ids})
    return evidence, list(candidates.values())


def lookup_names(db: sqlite3.Connection, lines: list[dict]) -> list[dict]:
    candidates = {}
    for line in lines:
        if line["confidence"] < MIN_CONFIDENCE:
            continue
        zh = re.match(r"[\u3400-\u9fff]{3,}", compact(line["text"]))
        en = english_words(line["text"])
        # A name fragment should begin with a distinctive word, such as "Dynastat".
        if zh is not None:
            language, term = "name_zh", zh.group()
            condition = "instr(ocr_compact(name_zh), ?) = 1"
        elif re.match(r"[a-z]{6,}\b", en):
            language, term = "name_en", en + " "
            condition = "instr(ocr_english(name_en) || ' ', ?) = 1"
        else:
            continue
        for row in db.execute(f"{MEDICINES} WHERE {condition} ORDER BY id", (term,)):
            if row["id"] not in candidates:
                candidates[row["id"]] = {**medicine(row), "name_evidence": []}
            candidates[row["id"]]["name_evidence"].append({**line, "field": language})

    return sorted(candidates.values(), key=lambda row: (-len({e["field"] for e in row["name_evidence"]}), row["id"]))


def lookup_bottle(db: sqlite3.Connection, roi: dict) -> dict:
    result = {"id": roi["id"], "image": roi.get("image"), "ocr_status": roi["status"],
              "focus_status": roi.get("focus", {}).get("status"), "method": None,
              "license_evidence": [], "candidates": []}
    if roi["status"] != "recognized":
        return {**result, "status": "skipped", "reason": "ocr_not_recognized"}

    lines = [{"line": index, "text": row["text"], "confidence": row["confidence"]}
             for index, row in enumerate(roi["lines"], 1)]
    evidence, licensed = lookup_licenses(db, lines)
    result["license_evidence"] = evidence
    if licensed:
        result.update(method="license", candidates=licensed)
        if len(licensed) > 1:
            return {**result, "status": "review", "reason": "multiple_license_rows"}
        if any(not entry["candidate_ids"] for entry in evidence):
            return {**result, "status": "review", "reason": "unresolved_license_evidence"}
        if max(entry["confidence"] for entry in evidence) < MIN_CONFIDENCE:
            return {**result, "status": "review", "reason": "low_ocr_confidence"}
        if result["focus_status"] == "review":
            return {**result, "status": "review", "reason": "focus_needs_review"}
        return {**result, "status": "matched", "reason": "unique_license_row"}

    names = lookup_names(db, lines)
    if names:
        reason = "license_not_found_name_candidates" if evidence else "name_candidates"
        return {**result, "status": "review", "reason": reason, "method": "name", "candidates": names}
    reason = "license_not_found" if evidence else "no_name_match"
    return {**result, "status": "not_found", "reason": reason}


def run_lookup(ocr_path: Path, database: Path = DEFAULT_DATABASE) -> Path:
    source = json.loads(ocr_path.read_text(encoding="utf-8-sig"))
    if source["status"] not in ("completed", "needs_review", "ocr_error"):
        raise ValueError(f"Expected a finished OCR report: {ocr_path}")
    database = database.resolve(strict=True)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.create_function("ocr_compact", 1, compact, deterministic=True)
        db.create_function("ocr_english", 1, english_words, deterministic=True)
        rows = [lookup_bottle(db, roi) for roi in source["rois"]]

    report = {"source_ocr": str(ocr_path.resolve()), "database": str(database),
              "min_confidence": MIN_CONFIDENCE,
              "counts": {status: sum(row["status"] == status for row in rows)
                         for status in ("matched", "review", "not_found", "skipped")},
              "rois": rows}
    name = datetime.now().strftime("lookup_%Y%m%d_%H%M%S_%f.json")
    output = ocr_path.parent / name
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    for row in rows:
        print(f"Bottle {row['id']}: {row['status']} ({row['reason']}); {len(row['candidates'])} database row(s)")
    print(f"SQLite lookup: {output}")
    return output
