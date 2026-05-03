"""
E7 — Eval harness for SROP routing accuracy.

Usage:
    python eval/run_eval.py [--base-url http://127.0.0.1:8000]

Creates a session, sends each eval query, compares routed_to against expected.
Prints a formatted table with pass/fail and overall accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import httpx


def load_eval_cases() -> list[dict]:
    """Load eval cases from JSON file."""
    cases_path = Path(__file__).parent / "eval_cases.json"
    with open(cases_path) as f:
        return json.load(f)


def run_eval(base_url: str) -> None:
    """Run all eval cases against the running server."""
    cases = load_eval_cases()

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        # Create a session for all eval queries
        resp = client.post("/v1/sessions", json={"user_id": "eval_user", "plan_tier": "pro"})
        if resp.status_code != 200:
            print(f"❌ Failed to create session: {resp.status_code} {resp.text}")
            sys.exit(1)
        session_id = resp.json()["session_id"]
        print(f"Created session: {session_id}\n")

        results: list[dict] = []
        correct = 0
        total = len(cases)

        print(f"{'#':<4} {'Query':<50} {'Expected':<14} {'Actual':<14} {'Result':<8}")
        print("─" * 94)

        for i, case in enumerate(cases, 1):
            query = case["query"]
            expected = case["expected_route"]

            try:
                resp = client.post(
                    f"/v1/chat/{session_id}",
                    json={"content": query},
                )
                if resp.status_code != 200:
                    actual = f"ERROR({resp.status_code})"
                    passed = False
                else:
                    data = resp.json()
                    actual = data.get("routed_to", "unknown")
                    passed = actual == expected
            except Exception as e:
                actual = f"ERROR({e})"
                passed = False

            if passed:
                correct += 1

            status = "✅ PASS" if passed else "❌ FAIL"
            display_query = query[:47] + "..." if len(query) > 50 else query
            print(f"{i:<4} {display_query:<50} {expected:<14} {actual:<14} {status}")

            results.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "passed": passed,
            })

        accuracy = (correct / total * 100) if total > 0 else 0
        print("─" * 94)
        print(f"\nRouting Accuracy: {correct}/{total} ({accuracy:.1f}%)")

        if accuracy >= 90:
            print("🎯 Excellent routing accuracy!")
        elif accuracy >= 70:
            print("👍 Good routing accuracy.")
        else:
            print("⚠️  Routing accuracy needs improvement.")

        # Save results to file
        results_path = Path(__file__).parent / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SROP Routing Eval Harness")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running SROP server",
    )
    args = parser.parse_args()
    run_eval(args.base_url)


if __name__ == "__main__":
    main()
