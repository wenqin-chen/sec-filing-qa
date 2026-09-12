"""secqa.eval: the reproducible FinanceBench harness.

Public surface (see each submodule's docstring):

* :func:`load_financebench` / :func:`download_pdfs` -- the dataset (ids only ever persisted).
* :class:`EvalConfig` / :func:`run_eval` -- one row of the matrix -> ``results/<name>/<run_id>/``.
* :func:`page_recall_at_k`, :func:`evidence_overlap_recall`, :func:`gold_page_mrr`,
  :func:`numeric_match`, :func:`bootstrap_ci`, :func:`classify_failure`, :func:`summarize`.
* :func:`judge_correctness` / :func:`judge_faithfulness` (LLM), :class:`RuleJudge` (offline),
  :func:`judge_swap` / :func:`human_agreement` (Cohen's kappa).
* :func:`rescore` -- replay cassettes with no keys; :func:`render_results_md` -- RESULTS.md.
"""

from secqa.eval.financebench import (
    DownloadReport,
    download_pdfs,
    load_financebench,
    load_questions_jsonl,
    question_from_row,
    questions_from_rows,
    save_questions_jsonl,
)
from secqa.eval.judge import (
    JUDGE_VERSION,
    AgreementReport,
    JudgeParseError,
    LLMJudge,
    RuleJudge,
    cohen_kappa,
    human_agreement,
    judge_correctness,
    judge_faithfulness,
    judge_prompt_hashes,
    judge_swap,
    make_judge,
)
from secqa.eval.metrics import (
    bootstrap_ci,
    classify_failure,
    effective_label,
    evidence_overlap_recall,
    gold_page_mrr,
    numeric_match,
    page_recall_at_k,
    read_records,
    summarize,
)
from secqa.eval.report import render_results_md, write_results_md
from secqa.eval.rescore import rescore
from secqa.eval.runner import EvalConfig, build_fixture_index, load_config, run_eval

__all__ = [
    "JUDGE_VERSION",
    "AgreementReport",
    "DownloadReport",
    "EvalConfig",
    "JudgeParseError",
    "LLMJudge",
    "RuleJudge",
    "bootstrap_ci",
    "build_fixture_index",
    "classify_failure",
    "cohen_kappa",
    "download_pdfs",
    "effective_label",
    "evidence_overlap_recall",
    "gold_page_mrr",
    "human_agreement",
    "judge_correctness",
    "judge_faithfulness",
    "judge_prompt_hashes",
    "judge_swap",
    "load_config",
    "load_financebench",
    "load_questions_jsonl",
    "make_judge",
    "numeric_match",
    "page_recall_at_k",
    "question_from_row",
    "questions_from_rows",
    "read_records",
    "render_results_md",
    "rescore",
    "run_eval",
    "save_questions_jsonl",
    "summarize",
    "write_results_md",
]
