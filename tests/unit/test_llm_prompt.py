"""Tests for CommonMark fence derivation in trelix.llm.prompt."""

from __future__ import annotations

from trelix.llm.prompt import MIN_FENCE_LENGTH, fence_for, fenced_block

# A verbatim excerpt from this repo's own .github/SECURITY.md — the worst real
# payload in the tracked tree (72 of 79 tracked .md files contain a run of
# exactly three backticks; none contain a longer one). trelix indexes markdown,
# so this is what a real symbol body looks like, not a crafted attack string.
REAL_MARKDOWN_PAYLOAD = (
    "on the public Sigstore transparency log.\n"
    "\n"
    "Verify a release:\n"
    "```bash\n"
    "pip install pypi-attestations\n"
    "python -m pypi_attestations verify pypi trelix\n"
    "```\n"
)


class TestFenceFor:
    def test_no_backticks_uses_minimum_fence(self) -> None:
        assert fence_for("def f():\n    return 1\n") == "```"

    def test_short_runs_still_use_minimum_fence(self) -> None:
        # Runs of 1 and 2 cannot close a three-backtick fence, so they must not
        # inflate it — this is what keeps output byte-identical for source code.
        assert fence_for("use `x` and ``y`` inline") == "```"

    def test_run_of_exactly_three_grows_to_four(self) -> None:
        assert fence_for("```") == "````"

    def test_run_longer_than_three(self) -> None:
        assert fence_for("`````") == "``````"

    def test_multiple_runs_uses_longest(self) -> None:
        assert fence_for("```\na\n`````\nb\n````\n") == "``````"

    def test_run_at_start_of_payload(self) -> None:
        assert fence_for("````python\nx = 1\n") == "`````"

    def test_run_at_end_of_payload(self) -> None:
        assert fence_for("x = 1\n`````") == "``````"

    def test_empty_payload(self) -> None:
        assert fence_for("") == "```"

    def test_whitespace_only_payload(self) -> None:
        assert fence_for("   \n\t\n") == "```"

    def test_minimum_is_three(self) -> None:
        assert MIN_FENCE_LENGTH == 3
        assert len(fence_for("")) == 3

    def test_real_markdown_payload_grows_the_fence(self) -> None:
        assert fence_for(REAL_MARKDOWN_PAYLOAD) == "````"


class TestFencedBlock:
    def test_plain_payload_is_byte_identical_to_legacy_format(self) -> None:
        """The exact string the three replaced call sites used to build."""
        payload = "def login(user):\n    return check(user)\n"
        assert fenced_block(payload) == f"```\n{payload}\n```"

    def test_empty_payload_is_byte_identical_to_legacy_format(self) -> None:
        assert fenced_block("") == "```\n\n```"

    def test_backtick_payload_is_well_formed(self) -> None:
        block = fenced_block("```\nrm -rf /\n```")
        assert block.startswith("````\n")
        assert block.endswith("\n````")
        # The opening fence must be strictly longer than anything inside, or
        # CommonMark closes the block early and leaks the payload as prose.
        assert "`````" not in block

    def test_real_markdown_payload_is_well_formed(self) -> None:
        block = fenced_block(REAL_MARKDOWN_PAYLOAD)
        opening, _, rest = block.partition("\n")
        assert opening == "````"
        assert rest.endswith("\n````")
        body = rest[: -len("\n````")]
        assert body == REAL_MARKDOWN_PAYLOAD
        # No line inside the body may open with a fence >= the opening fence.
        for line in body.split("\n"):
            assert not line.lstrip().startswith("````")

    def test_fence_matches_on_both_sides(self) -> None:
        for payload in ["", "x", "```", "`````", "a```b````c"]:
            block = fenced_block(payload)
            lines = block.split("\n")
            assert lines[0] == lines[-1], payload
            assert lines[0] == fence_for(payload), payload
