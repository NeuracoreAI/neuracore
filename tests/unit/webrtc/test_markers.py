"""Unit tests for the xfail-marker mechanism (PR1 ``shared/markers.py``).

Pins two PR1 self-checks: ``greened_by`` returns a strict xfail marker for every
valid PR slice (and rejects an unknown one), and the suite's live ``greened_by``
tags form a clean test -> PR map — every tag is a known ``PR_SLICES`` key and no
test is tagged more than once. (A PR greens its slice by deleting its marker, so
PR2-PR4 keys are no longer used as live tags; the invariant tested here is that
whatever tags remain are valid and unambiguous.)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.integration.webrtc.shared.markers import PR_SLICES, greened_by

_SUITE_DIR = Path(__file__).resolve().parents[2] / "integration" / "webrtc"
_TEST_MODULES = sorted(_SUITE_DIR.glob("test_*.py"))


def test_pr_slices_descriptions_are_non_empty_and_unique() -> None:
    assert PR_SLICES, "the PR -> slice map must not be empty"
    assert all(desc.strip() for desc in PR_SLICES.values())
    assert len(set(PR_SLICES.values())) == len(PR_SLICES), "duplicate slice text"


@pytest.mark.parametrize("pr", sorted(PR_SLICES))
def test_greened_by_yields_a_strict_xfail_for_every_slice(pr: str) -> None:
    detail = "a specific assertion this test pins"
    marker = greened_by(pr, detail)

    assert marker.mark.name == "xfail"
    assert marker.mark.kwargs["strict"] is True
    reason = marker.mark.kwargs["reason"]
    assert pr in reason
    assert PR_SLICES[pr] in reason
    assert detail in reason


def test_greened_by_rejects_an_unknown_pr() -> None:
    with pytest.raises(KeyError):
        greened_by("PR999", "no such slice")


def _greened_tags_by_test() -> dict[str, list[str]]:
    """Map each integration test function -> the PR tags it is greened_by.

    Parses the suite statically (no import, no native module) so the check runs
    in the unit layer.
    """
    tags: dict[str, list[str]] = {}
    for module in _TEST_MODULES:
        tree = ast.parse(module.read_text(), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            found: list[str] = []
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "greened_by"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    found.append(decorator.args[0].value)
            tags[node.name] = found
    return tags


def test_suite_is_discovered() -> None:
    # Guard against a path typo silently passing the static checks below.
    assert _TEST_MODULES, f"no integration test modules under {_SUITE_DIR}"
    assert _greened_tags_by_test(), "no test functions discovered in the suite"


def test_every_live_tag_is_a_known_slice() -> None:
    for test_name, found in _greened_tags_by_test().items():
        for tag in found:
            assert tag in PR_SLICES, f"{test_name} tagged with unknown slice {tag!r}"


def test_no_test_is_greened_by_more_than_once() -> None:
    # "covers every test exactly once": a marked test maps to a single PR; an
    # unmarked (already-greened) test maps to none.
    for test_name, found in _greened_tags_by_test().items():
        assert len(found) <= 1, f"{test_name} carries multiple greened_by markers"
