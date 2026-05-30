#!/usr/bin/env python3
"""
TinyTU Test Report Generator
============================
Parses test output logs and generates a structured JSON/HTML report.

Usage:
  python3 tools/test_report.py [--json] [--html] [LOGFILE...]

If no LOGFILE is given, scans build/ci_reports/logs/ for test logs.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_test_log(filepath: str) -> dict:
    """Parse a test log and extract pass/fail/total counts."""
    result = {
        "file": os.path.basename(filepath),
        "path": filepath,
        "passed": 0,
        "failed": 0,
        "total": 0,
        "tests": [],
        "exit_code": -1,
        "max_error": None,
        "mean_error": None,
    }

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        result["error"] = "unreadable"
        return result

    # Count individual PASS/FAIL lines
    for line in content.split("\n"):
        # Match lines like "  test name                    PASS"
        m = re.match(r"^\s+(.+?)\s{2,}(PASS|FAIL)", line)
        if m:
            name, status = m.group(1).strip(), m.group(2)
            result["tests"].append({"name": name, "status": status})
            if status == "PASS":
                result["passed"] += 1
            else:
                result["failed"] += 1

    result["total"] = result["passed"] + result["failed"]

    # Fallback: look for summary line like "N/M tests passed"
    if result["total"] == 0:
        m = re.search(r"(\d+)/(\d+)\s+tests\s+passed", content)
        if m:
            result["passed"] = int(m.group(1))
            result["total"] = int(m.group(2))
            result["failed"] = result["total"] - result["passed"]

    # Extract max_observed_error if present
    m = re.search(r"Max observed error:\s*([\d.e+\-]+)", content)
    if m:
        result["max_error"] = float(m.group(1))

    # Extract mean error
    m = re.search(r"avg_mean=([\d.e+\-]+)", content)
    if m:
        result["mean_error"] = float(m.group(1))

    # Check for overall pass/fail
    if "PASS" in content[-200:] or "passed" in content[-200:].lower():
        result["status"] = "PASS"
    elif "FAIL" in content:
        result["status"] = "FAIL"
    else:
        result["status"] = "UNKNOWN"

    return result


def scan_logs(log_dir: str) -> list:
    """Scan a directory for test log files."""
    logs = []
    if not os.path.isdir(log_dir):
        return logs
    for f in sorted(os.listdir(log_dir)):
        if f.endswith(".log"):
            logs.append(os.path.join(log_dir, f))
    return logs


def generate_html(report: dict) -> str:
    """Generate an HTML report page."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>TinyTU Test Report</title>",
        "<style>",
        "  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }",
        "  h1 { color: #7c83ff; border-bottom: 2px solid #333; padding-bottom: 10px; }",
        "  h2 { color: #66ccff; margin-top: 30px; }",
        "  table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
        "  th { background: #333; text-align: left; padding: 8px; }",
        "  td { padding: 6px 8px; border-bottom: 1px solid #333; }",
        "  .pass { color: #4caf50; font-weight: bold; }",
        "  .fail { color: #f44336; font-weight: bold; }",
        "  .summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 20px 0; }",
        "  .bar { height: 20px; border-radius: 4px; margin: 10px 0; }",
        "  .bar-pass { background: #4caf50; display: inline-block; height: 100%; border-radius: 4px; }",
        "  .bar-fail { background: #f44336; display: inline-block; height: 100%; border-radius: 4px; }",
        "</style>",
        "</head><body>",
        f"<h1>🔬 TinyTU Test Report</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # Summary
    total_p = sum(r["passed"] for r in report["results"])
    total_f = sum(r["failed"] for r in report["results"])
    total_t = total_p + total_f
    pct = (total_p / total_t * 100) if total_t > 0 else 0

    lines.append('<div class="summary">')
    lines.append(f'<h2>Summary: {total_p}/{total_t} passed ({pct:.1f}%)</h2>')
    lines.append(f'<div class="bar"><span class="bar-pass" style="width:{pct}%;"></span><span class="bar-fail" style="width:{100-pct}%;"></span></div>')
    lines.append("</div>")

    # Results table
    lines.append("<table><tr><th>Test</th><th>Status</th><th>Passed</th><th>Failed</th><th>Max Error</th></tr>")
    for r in report["results"]:
        status_class = "pass" if r["status"] == "PASS" else "fail"
        max_err = f"{r['max_error']:.6f}" if r.get("max_error") else "—"
        lines.append(
            f"<tr><td>{r['file']}</td>"
            f"<td class='{status_class}'>{r['status']}</td>"
            f"<td>{r['passed']}</td><td>{r['failed']}</td>"
            f"<td>{max_err}</td></tr>"
        )
    lines.append("</table>")

    # Detailed failures
    failures = [r for r in report["results"] if r["status"] == "FAIL"]
    if failures:
        lines.append("<h2>❌ Failures</h2>")
        for r in failures:
            lines.append(f"<h3>{r['file']}</h3>")
            for t in r.get("tests", []):
                if t["status"] == "FAIL":
                    lines.append(f"<p class='fail'>  ✗ {t['name']}</p>")

    lines.append("</body></html>")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    json_mode = "--json" in args
    html_mode = "--html" in args
    logs = [a for a in args if not a.startswith("--")]

    if not logs:
        # Scan default log directory
        log_dir = "build/ci_reports/logs"
        logs = scan_logs(log_dir)
        if not logs:
            print(f"No log files found in {log_dir}")
            sys.exit(1)

    results = []
    for logpath in logs:
        results.append(parse_test_log(logpath))

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": sum(r["total"] for r in results),
        "total_passed": sum(r["passed"] for r in results),
        "total_failed": sum(r["failed"] for r in results),
        "results": results,
    }

    if html_mode:
        print(generate_html(report))
        return

    # Default: text summary
    print(f"TinyTU Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    total_p, total_f = report["total_passed"], report["total_failed"]
    total_t = total_p + total_f
    pct = (total_p / total_t * 100) if total_t > 0 else 0
    print(f"  Total:  {total_t:4d} tests")
    print(f"  Passed: {total_p:4d}")
    print(f"  Failed: {total_f:4d}")
    print(f"  Rate:   {pct:.1f}%")
    print()

    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        print(f"  {icon} {r['file']:40s}  {r['passed']:3d}/{r['total']:3d}  {r['status']}")
        if r.get("max_error"):
            print(f"     max_err={r['max_error']:.6f}")

    if html_mode:
        print(generate_html(report))


if __name__ == "__main__":
    main()
