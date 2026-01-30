"""
Extract and aggregate scores from judged recipe prediction results.

Usage:
    python score.py judged_results.jsonl
    python score.py judged_results.jsonl --output_file scores.json
"""

import json
import re
import fire
import jsonlines


# ============================================
# Score Keys (14 evaluation criteria)
# ============================================

SCORE_KEYS = [
    "materials_appropriateness_score",
    "materials_completeness_score",
    "device_structure_completeness_score",
    "device_structure_similarity_score",
    "device_structure_feasibility_score",
    "deposition_parameter_completeness_score",
    "deposition_parameter_accuracy_score",
    "deposition_procedure_feasibility_score",
    "post_processing_completeness_score",
    "post_processing_similarity_score",
    "post_processing_feasibility_score",
    "performance_appropriateness_score",
    "performance_similarity_score",
    "overall_score",
]


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text containing ```json ... ``` blocks."""
    pattern = r"```json\s*(\{[^`]+\})\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def compute_average_scores(filename: str, output_file: str = None) -> dict:
    """
    Compute average scores from judged results.

    Args:
        filename: Path to judged JSONL file
        output_file: Path to save JSON output (default: {input}_scores.json)

    Returns:
        Dictionary with average scores per criterion
    """
    if output_file is None:
        output_file = filename.replace(".jsonl", "_scores.json")

    scores = {key: [] for key in SCORE_KEYS}
    failed_count = 0
    total_count = 0
    corrupted_lines = 0

    # Read file line by line to handle corrupted JSONL
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Try to parse line as JSON
            items = []
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                if e.msg == "Extra data":
                    # Multiple JSON objects on same line
                    pos = 0
                    while pos < len(line):
                        try:
                            obj = json.loads(line[pos:])
                            items.append(obj)
                            break
                        except json.JSONDecodeError as e2:
                            if e2.msg == "Extra data" and e2.pos:
                                obj = json.loads(line[pos : pos + e2.pos])
                                items.append(obj)
                                pos = pos + e2.pos
                            else:
                                corrupted_lines += 1
                                print(f"Warning: Could not parse line {line_num}")
                                break
                else:
                    corrupted_lines += 1
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue

            # Process all items extracted from this line
            for item in items:
                total_count += 1
                judge_result = item.get("judge_result", "")
                extracted = extract_json_from_text(judge_result)

                if extracted is None:
                    failed_count += 1
                    continue

                for key in SCORE_KEYS:
                    if key in extracted and extracted[key] is not None:
                        scores[key].append(float(extracted[key]))

    # Compute averages
    avg_scores = {}
    for key in SCORE_KEYS:
        if scores[key]:
            avg_scores[key] = round(sum(scores[key]) / len(scores[key]), 3)
        else:
            avg_scores[key] = None

    result = {
        "total_samples": total_count,
        "parsed_samples": total_count - failed_count,
        "failed_samples": failed_count,
        "corrupted_lines": corrupted_lines,
        "average_scores": avg_scores,
    }

    if corrupted_lines > 0:
        print(f"\nWarning: {corrupted_lines} corrupted lines were skipped")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scores saved to {output_file}")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    fire.Fire(compute_average_scores)
