"""Judges: the LLM correctness / faithfulness judges, the offline rule judge, and agreement.

Design (SPEC section 7):

* One fixed judge for every row (``anthropic:claude-sonnet-5``, effort ``low``) so rows are
  comparable; the judge prompts are frozen files whose SHA-256 is recorded in every record next
  to :data:`JUDGE_VERSION`. Editing a prompt is a new judge version and every row is re-judged
  from cassettes.
* :func:`judge_correctness` sees the question, gold answer, justification and the prediction and
  returns a tri-state label. Its free-text rationale restates that CC-BY-NC gold text, so the
  verdict carries only ``sha256:<digest>`` of it (:func:`redact_rationale`); the verbatim reply
  lives in the run's cassette, never in ``predictions.jsonl`` (CONTRACTS rule 5).
  :func:`judge_faithfulness` sees the prediction and its cited passages *only* (gold hidden),
  extracts atomic claims and counts the supported ones.
* :class:`RuleJudge` (``judge: rule`` in a config) is the key-free judge for CI and mock rows:
  abstention detection plus :func:`~secqa.eval.metrics.numeric_match`; a free-text answer it
  cannot decide is left unscored (``None``), never guessed.
* :func:`judge_swap` re-judges a run with a second model and :func:`human_agreement` compares the
  effective labels with ``human_labels.csv``; both report Cohen's kappa, and kappa below
  :data:`PROVISIONAL_KAPPA` marks accuracy cells "provisional" in the report.

Judge output parsing prefers the provider's ``parsed`` JSON, then JSON in the text (code fences
stripped). Anything else raises :class:`JudgeParseError`; the runner records the failure on the
record instead of inventing a verdict.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

from secqa.core.contracts import (
    Answer,
    Citation,
    EvalRecord,
    FaithVerdict,
    FBQuestion,
    Frozen,
    JudgeVerdict,
    LLMProvider,
    Message,
    Usage,
)
from secqa.core.errors import ConfigError
from secqa.core.ids import sha256_hex
from secqa.core.logging import get_logger
from secqa.core.settings import provider_vendor
from secqa.eval.metrics import effective_label, numeric_match, numeric_match_scale, read_records
from secqa.rag import ABSTAIN_TEXT
from secqa.rag.prompts import JUDGE_CORRECTNESS_SYSTEM, JUDGE_FAITHFULNESS_SYSTEM
from secqa.rag.prompts import PROMPTS_DIR as _RAG_PROMPTS_DIR

log = get_logger(__name__)

JUDGE_VERSION = "v1"
"""Bumped whenever a judge prompt or schema changes; recorded on every verdict."""

PROMPTS_DIR = _RAG_PROMPTS_DIR
"""``src/secqa/prompts/``: the one prompt directory (CONTRACTS rule 11); the judge prompts are
listed in :data:`secqa.rag.prompts.PROMPT_NAMES` and pinned with the answering prompts."""
JUDGE_CORRECTNESS = JUDGE_CORRECTNESS_SYSTEM
JUDGE_FAITHFULNESS = JUDGE_FAITHFULNESS_SYSTEM
JUDGE_PROMPT_NAMES: tuple[str, ...] = (JUDGE_CORRECTNESS, JUDGE_FAITHFULNESS)

RULE_JUDGE = "rule"
JUDGE_EFFORT: Literal["low"] = "low"
CORRECTNESS_MAX_TOKENS = 512
FAITHFULNESS_MAX_TOKENS = 1024
PROVISIONAL_KAPPA = 0.6
LABELS: tuple[str, ...] = ("correct", "incorrect", "abstain")
PASSAGE_MAX_CHARS = 2000
RATIONALE_DIGEST_PREFIX = "sha256:"
"""Prefix of a persisted LLM-judge rationale: the digest of the text, never the text."""

JUDGE_SCHEMA: dict[str, Any] = {
    "title": "judge_correctness",
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "rationale"],
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "rationale": {"type": "string"},
    },
}

FAITH_SCHEMA: dict[str, Any] = {
    "title": "judge_faithfulness",
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim", "supported"],
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
            },
        }
    },
}

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class JudgeParseError(ValueError):
    """The judge's reply could not be turned into a verdict (kept as ``None`` by the runner)."""


# ---------------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------------


@cache
def load_judge_prompt(name: str) -> str:
    """Text of ``prompts/<name>`` (cached); ``ConfigError`` if missing or empty."""
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise ConfigError(f"judge prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"judge prompt is empty: {path}")
    return text


def judge_prompt_hashes() -> dict[str, str]:
    """``{file name: sha256}`` of both judge prompts, read from disk on every call.

    Byte-for-byte the same digests :func:`secqa.rag.prompts.prompt_hashes` reports for these
    two files; the runner merges both so a record is comparable across modes.
    """
    return {name: sha256_hex((PROMPTS_DIR / name).read_bytes()) for name in JUDGE_PROMPT_NAMES}


def _render_prediction(pred: Answer) -> str:
    value = "null" if pred.value is None else repr(pred.value)
    unit = pred.unit or "null"
    return (
        f"Prediction text: {' '.join(pred.text.split()) or '(empty)'}\n"
        f"Prediction value: {value}\n"
        f"Prediction unit: {unit}\n"
        f"Prediction abstain: {'true' if pred.abstained else 'false'}"
    )


def build_correctness_prompt(q: FBQuestion, pred: Answer) -> str:
    """User message of the correctness judge: question, gold answer, justification, prediction."""
    return (
        f"Question: {' '.join(q.question.split())}\n"
        f"Reference answer: {' '.join(q.answer.split())}\n"
        f"Reference justification: {' '.join(q.justification.split()) or '(none)'}\n\n"
        f"{_render_prediction(pred)}"
    )


def cited_passages(citations: Sequence[Citation]) -> list[str]:
    """Store-sourced text the faithfulness judge may see: quote and snippet of each valid citation.

    The ``snippet`` is always store text (CONTRACTS rule 2) and the ``quote`` is included only
    when the verifier confirmed it, so the judge never sees a fabricated passage.
    """
    passages: list[str] = []
    for citation in citations:
        if not citation.valid:
            continue
        parts: list[str] = []
        if citation.verified and citation.quote.strip():
            parts.append(citation.quote.strip())
        if citation.snippet.strip():
            parts.append(citation.snippet.strip())
        if not parts:
            continue
        label = citation.ref
        if citation.doc_name and citation.page_num is not None:
            label += f" {citation.doc_name} p.{citation.page_num}"
        text = " ... ".join(dict.fromkeys(parts))
        passages.append(f"({label}) {text[:PASSAGE_MAX_CHARS]}")
    return passages


def build_faithfulness_prompt(pred: Answer) -> str:
    """User message of the faithfulness judge: the answer and its cited passages (no gold)."""
    passages = cited_passages(pred.citations)
    rendered = "\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    if not rendered:
        rendered = "(no valid cited passages)"
    return f"Cited passages:\n{rendered}\n\nAnswer: {' '.join(pred.text.split()) or '(empty)'}"


# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------


def _decode_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    fenced = _CODE_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def parse_correctness(parsed: dict[str, Any] | None, text: str) -> tuple[str, str]:
    """``(label, rationale)`` from a judge reply; raises :class:`JudgeParseError`."""
    data = parsed if parsed is not None else _decode_object(text)
    if data is None:
        raise JudgeParseError("judge reply is not a JSON object")
    label = str(data.get("label") or "").strip().lower()
    if label not in LABELS:
        raise JudgeParseError(f"judge label {label!r} is not one of {LABELS}")
    rationale = data.get("rationale")
    return label, (" ".join(str(rationale).split()) if rationale else "")


def parse_faithfulness(parsed: dict[str, Any] | None, text: str) -> list[tuple[str, bool]]:
    """``[(claim, supported), ...]`` from a judge reply; raises :class:`JudgeParseError`."""
    data = parsed if parsed is not None else _decode_object(text)
    if data is None:
        raise JudgeParseError("judge reply is not a JSON object")
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise JudgeParseError("judge reply has no 'claims' list")
    out: list[tuple[str, bool]] = []
    for index, item in enumerate(claims):
        if not isinstance(item, dict) or not isinstance(item.get("supported"), bool):
            raise JudgeParseError(f"claims[{index}] needs a boolean 'supported'")
        out.append((" ".join(str(item.get("claim") or "").split()), item["supported"]))
    return out


# ---------------------------------------------------------------------------------------------
# LLM judges
# ---------------------------------------------------------------------------------------------


def redact_rationale(rationale: str) -> str:
    """Persistable form of an LLM judge rationale: ``sha256:<hex>``, or ``''`` when empty.

    The correctness judge is shown the gold answer and justification and asked to name the
    decisive difference, so its rationale routinely quotes CC-BY-NC dataset text. Every
    :class:`JudgeVerdict` is written to ``predictions.jsonl`` (committed under ``results/``),
    which must carry no question / answer / evidence text (CONTRACTS rule 5), so only the
    digest is kept; the verbatim reply stays in the run's cassette (a Release asset) and the
    digest identifies that reply. The rule judge's rationale is built from our own prediction
    only and is stored as is.
    """
    if not rationale:
        return ""
    return f"{RATIONALE_DIGEST_PREFIX}{sha256_hex(rationale)}"


def judge_correctness(q: FBQuestion, pred: Answer, judge: LLMProvider) -> JudgeVerdict:
    """Tri-state correctness verdict from ``judge`` (effort low, JSON-schema output).

    A provider refusal is a parse failure (no label was produced), never a silent "incorrect".
    The verdict's ``rationale`` is the digest of the judge's rationale (:func:`redact_rationale`),
    not its text.

    Raises:
        JudgeParseError: when the reply carries no valid label.
        ProviderError: propagated from the provider.
    """
    response = judge.complete(
        [Message(role="user", content=build_correctness_prompt(q, pred))],
        system=load_judge_prompt(JUDGE_CORRECTNESS),
        json_schema=JUDGE_SCHEMA,
        max_tokens=CORRECTNESS_MAX_TOKENS,
        effort=JUDGE_EFFORT,
    )
    if response.stop_reason == "refusal":
        raise JudgeParseError("judge refused the request")
    label, rationale = parse_correctness(response.parsed, response.text)
    verdict = JudgeVerdict(
        label=label,  # type: ignore[arg-type]  # validated against LABELS in parse_correctness
        rationale=redact_rationale(rationale),
        judge_model=f"{response.provider}:{response.model}",
        judge_version=JUDGE_VERSION,
        usage=response.usage,
    )
    log.info(
        "judge_correctness",
        financebench_id=q.id,
        label=verdict.label,
        judge_model=verdict.judge_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached=response.cached,
    )
    return verdict


def judge_faithfulness(pred: Answer, judge: LLMProvider) -> FaithVerdict:
    """Claim-level faithfulness against the cited passages only (gold hidden).

    ``score`` is ``supported / claims`` and ``None`` when the judge found no factual claim.

    Raises:
        JudgeParseError: when the reply carries no valid claims list.
        ProviderError: propagated from the provider.
    """
    response = judge.complete(
        [Message(role="user", content=build_faithfulness_prompt(pred))],
        system=load_judge_prompt(JUDGE_FAITHFULNESS),
        json_schema=FAITH_SCHEMA,
        max_tokens=FAITHFULNESS_MAX_TOKENS,
        effort=JUDGE_EFFORT,
    )
    if response.stop_reason == "refusal":
        raise JudgeParseError("judge refused the request")
    claims = parse_faithfulness(response.parsed, response.text)
    supported = sum(1 for _, ok in claims if ok)
    verdict = FaithVerdict(
        claims=len(claims),
        supported=supported,
        score=(supported / len(claims)) if claims else None,
        judge_model=f"{response.provider}:{response.model}",
        judge_version=JUDGE_VERSION,
        usage=response.usage,
    )
    log.info(
        "judge_faithfulness",
        claims=verdict.claims,
        supported=verdict.supported,
        judge_model=verdict.judge_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached=response.cached,
    )
    return verdict


# ---------------------------------------------------------------------------------------------
# judge objects used by the runner
# ---------------------------------------------------------------------------------------------


class RuleJudge:
    """Key-free judge: abstention detection plus strict numeric match; undecidable -> ``None``."""

    name = RULE_JUDGE
    model = RULE_JUDGE

    def correctness(self, q: FBQuestion, pred: Answer) -> JudgeVerdict | None:
        """``abstain`` / ``correct`` / ``incorrect`` when decidable by rule, else ``None``."""
        if pred.abstained:
            label, why = "abstain", "prediction abstained"
        else:
            match = numeric_match(pred.value, q.answer)
            if match is None:
                return None
            label = "correct" if match else "incorrect"
            scale = numeric_match_scale(pred.value, q.answer) if match else None
            if scale is not None and scale != 1.0:
                # Auditable: a scaled acceptance names the factor the gold was understated by.
                outcome = f"match at x{int(scale):,}, gold understated by a unit scale"
            else:
                outcome = "match" if match else "mismatch"
            why = f"structured value {pred.value!r} vs gold number ({outcome})"
        return JudgeVerdict(
            label=label,  # type: ignore[arg-type]  # one of LABELS by construction
            rationale=why,
            judge_model=RULE_JUDGE,
            judge_version=JUDGE_VERSION,
            usage=Usage(),
        )

    def faithfulness(self, pred: Answer) -> FaithVerdict | None:
        """The rule judge cannot read claims; faithfulness stays unscored."""
        return None


class LLMJudge:
    """Wrap an :class:`LLMProvider` as the runner's judge (correctness + faithfulness)."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.name = f"{provider.provider}:{provider.model}"
        self.model = self.name

    def correctness(self, q: FBQuestion, pred: Answer) -> JudgeVerdict | None:
        """See :func:`judge_correctness` (errors propagate to the runner)."""
        return judge_correctness(q, pred, self.provider)

    def faithfulness(self, pred: Answer) -> FaithVerdict | None:
        """Faithfulness only for answered predictions with at least one valid citation."""
        if pred.abstained or not cited_passages(pred.citations):
            return None
        return judge_faithfulness(pred, self.provider)


Judge = RuleJudge | LLMJudge


def make_judge(spec: str, provider: LLMProvider | None = None) -> Judge:
    """``'rule'`` -> :class:`RuleJudge`; otherwise wrap ``provider`` (must be given).

    A ``mock`` judge is refused: the mock provider fills the enum with its first member and
    would fabricate verdicts. Use ``rule`` for offline rows.
    """
    if spec.strip().lower() == RULE_JUDGE:
        return RuleJudge()
    if provider_vendor(spec) == "mock":
        raise ConfigError("judge 'mock' would fabricate verdicts; use judge: rule for offline runs")
    if provider is None:
        raise ConfigError(f"judge {spec!r} needs a provider instance")
    return LLMJudge(provider)


# ---------------------------------------------------------------------------------------------
# agreement: judge swap and human labels
# ---------------------------------------------------------------------------------------------


class AgreementReport(Frozen):
    """Agreement between two labelings of the same records (Cohen's kappa)."""

    name_a: str
    name_b: str
    n: int
    agreement: float | None
    kappa: float | None
    confusion: dict[str, dict[str, int]]  # confusion[a_label][b_label]
    disagreements: list[str]  # financebench_ids
    provisional: bool  # kappa is undefined or below PROVISIONAL_KAPPA
    judge_cost_usd: float = 0.0
    run_id: str = ""


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two label sequences; ``None`` when undefined (empty, or no variation)."""
    if len(a) != len(b):
        raise ValueError("label sequences must have the same length")
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for x, y in pairs:
        table.setdefault(x, {}).setdefault(y, 0)
        table[x][y] += 1
    return {x: dict(sorted(row.items())) for x, row in sorted(table.items())}


def _agreement_report(
    name_a: str,
    name_b: str,
    labelled: list[tuple[str, str, str]],
    *,
    judge_cost_usd: float = 0.0,
    run_id: str = "",
) -> AgreementReport:
    ids = [rid for rid, _, _ in labelled]
    a = [x for _, x, _ in labelled]
    b = [y for _, _, y in labelled]
    kappa = cohen_kappa(a, b)
    return AgreementReport(
        name_a=name_a,
        name_b=name_b,
        n=len(labelled),
        agreement=(sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)) if a else None,
        kappa=kappa,
        confusion=_confusion(zip(a, b, strict=True)),
        disagreements=sorted(rid for rid, x, y in zip(ids, a, b, strict=True) if x != y),
        provisional=kappa is None or kappa < PROVISIONAL_KAPPA,
        judge_cost_usd=round(judge_cost_usd, 6),
        run_id=run_id,
    )


def answer_from_record(rec: EvalRecord, question: str) -> Answer:
    """Rebuild the parts of an :class:`Answer` a judge needs from a persisted record."""
    return Answer(
        request_id=rec.financebench_id,
        question=question,
        text=rec.answer_text or (ABSTAIN_TEXT if rec.abstained else ""),
        value=rec.value,
        unit=rec.unit,
        abstained=rec.abstained,
        citations=list(rec.citations),
        grounded=rec.grounded,
        retrieved=[],
        trace=[],
        usage=rec.usage,
        cost_usd=rec.cost_usd,
        latency_ms=rec.latency_ms,
        retrieval_ms=rec.retrieval_ms,
        llm_ms=rec.llm_ms,
        provider=rec.provider,
        model=rec.model,
        mode=rec.mode,
        steps=rec.steps,
        tool_calls=rec.tool_calls,
        terminated_by=rec.terminated_by,
        prompt_hashes=dict(rec.prompt_hashes),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def judge_swap(
    pred_path: Path,
    judge: LLMProvider,
    questions: Sequence[FBQuestion] | None = None,
    *,
    prices: Any | None = None,
) -> AgreementReport:
    """Re-judge every completed record with a second judge and report Cohen's kappa.

    Predictions carry no dataset text (SPEC 9), so ``questions`` (the locally loaded
    FinanceBench rows) are required to rebuild the judge prompt; records without a matching
    question are skipped with a warning. Compares the run's *effective* label (numeric override
    included) against the swap judge's label. The report is also written as
    ``judge_swap_<provider>_<model>.json`` next to the predictions.

    Raises:
        ConfigError: when ``questions`` is missing or no record could be judged.
    """
    if not questions:
        raise ConfigError("judge_swap needs the FinanceBench questions to rebuild judge prompts")
    pred_path = Path(pred_path)
    records = [rec for rec in read_records(pred_path) if not rec.error]
    by_id = {q.id: q for q in questions}
    labelled: list[tuple[str, str, str]] = []
    cost = 0.0
    for rec in records:
        original = effective_label(rec)
        question = by_id.get(rec.financebench_id)
        if original is None or question is None:
            log.warning(
                "judge_swap_skipped",
                financebench_id=rec.financebench_id,
                reason="unscored record" if original is None else "question not found",
            )
            continue
        verdict = judge_correctness(question, answer_from_record(rec, question.question), judge)
        if prices is not None:
            cost += float(prices.cost_usd(judge.provider, judge.model, verdict.usage))
        labelled.append((rec.financebench_id, original, verdict.label))
    if not labelled:
        raise ConfigError("judge_swap: no record could be re-judged")
    name_a = next((rec.judge.judge_model for rec in records if rec.judge), "original")
    report = _agreement_report(
        f"{name_a}+numeric",
        f"{judge.provider}:{judge.model}",
        labelled,
        judge_cost_usd=cost,
        run_id=records[0].run_id if records else "",
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{judge.provider}_{judge.model}")
    _write_json(pred_path.parent / f"judge_swap_{safe}.json", report.model_dump(mode="json"))
    log.info("judge_swap", n=report.n, kappa=report.kappa, agreement=report.agreement)
    return report


HUMAN_LABEL_COLUMNS: tuple[str, ...] = ("financebench_id", "label")
HUMAN_AGREEMENT_NAME = "human_agreement.json"


def read_human_labels(labels_csv: Path) -> list[dict[str, str]]:
    """Rows of ``human_labels.csv`` (``#`` comment lines skipped, labels validated)."""
    labels_csv = Path(labels_csv)
    if not labels_csv.is_file():
        raise ConfigError(f"human labels file not found: {labels_csv}")
    lines = [
        line
        for line in labels_csv.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ConfigError(f"{labels_csv} has no header row")
    reader = csv.DictReader(lines)
    missing = [col for col in HUMAN_LABEL_COLUMNS if col not in (reader.fieldnames or [])]
    if missing:
        raise ConfigError(f"{labels_csv} is missing columns {missing}")
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=2):
        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            raise ConfigError(f"{labels_csv}: row {index}: label {label!r} is not one of {LABELS}")
        rows.append({k: (v or "").strip() for k, v in row.items() if k})
    return rows


def human_agreement(pred_path: Path, labels_csv: Path) -> AgreementReport:
    """Cohen's kappa between the run's effective labels and the human labels for the same ids.

    Label rows whose ``run_id`` is set and differs from the predictions' run are ignored (they
    grade another run's answers). The report is written as ``human_agreement.json`` next to the
    predictions, which is where the report renderer looks to clear the "provisional" mark.

    Raises:
        ConfigError: when no label row matches a scored record.
    """
    pred_path = Path(pred_path)
    records = {rec.financebench_id: rec for rec in read_records(pred_path) if not rec.error}
    run_id = next((rec.run_id for rec in records.values()), "")
    labelled: list[tuple[str, str, str]] = []
    for row in read_human_labels(labels_csv):
        row_run = row.get("run_id", "")
        if row_run and run_id and row_run != run_id:
            continue
        rec = records.get(row["financebench_id"])
        if rec is None:
            continue
        judged = effective_label(rec)
        if judged is None:
            continue
        labelled.append((rec.financebench_id, judged, row["label"]))
    if not labelled:
        raise ConfigError(f"no human label in {labels_csv} matches a scored record of {pred_path}")
    name_a = next((rec.judge.judge_model for rec in records.values() if rec.judge), "judge")
    report = _agreement_report(f"{name_a}+numeric", "human", labelled, run_id=run_id)
    _write_json(
        pred_path.parent / HUMAN_AGREEMENT_NAME,
        {**report.model_dump(mode="json"), "computed_at": datetime.now(UTC).isoformat()},
    )
    log.info("human_agreement", n=report.n, kappa=report.kappa, agreement=report.agreement)
    return report


__all__ = [
    "FAITH_SCHEMA",
    "HUMAN_AGREEMENT_NAME",
    "JUDGE_CORRECTNESS",
    "JUDGE_EFFORT",
    "JUDGE_FAITHFULNESS",
    "JUDGE_PROMPT_NAMES",
    "JUDGE_SCHEMA",
    "JUDGE_VERSION",
    "LABELS",
    "PROMPTS_DIR",
    "PROVISIONAL_KAPPA",
    "RULE_JUDGE",
    "AgreementReport",
    "Judge",
    "JudgeParseError",
    "LLMJudge",
    "RuleJudge",
    "answer_from_record",
    "build_correctness_prompt",
    "build_faithfulness_prompt",
    "cited_passages",
    "cohen_kappa",
    "human_agreement",
    "judge_correctness",
    "judge_faithfulness",
    "judge_prompt_hashes",
    "judge_swap",
    "load_judge_prompt",
    "make_judge",
    "parse_correctness",
    "parse_faithfulness",
    "read_human_labels",
]
