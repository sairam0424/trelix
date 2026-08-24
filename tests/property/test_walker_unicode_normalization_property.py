"""DEFECT (pinned deliberately): `FileWalker` does not normalize filenames to a
canonical Unicode form, so a file whose on-disk name is stored in NFD
(decomposed) form is walked with that RAW, unnormalized `rel_path` -- the
"re-embedded every run forever" risk the round's own plan measured: an index
built from a checkout that hands back one normalization form and later
re-walked on a checkout (or filesystem) that hands back the other sees a
`rel_path` string change for a file whose content never did, which the
indexer's change-detection (keyed on `rel_path`) reads as delete+add.

VERIFIED STILL TRUE (per this round's instructions, before writing this test):
grepped `normalize|unicodedata|NFC|NFD` across src/trelix/indexing/walker.py --
zero hits. `walker.py`'s only path-string handling
(`_iter_files`/`walk()`) passes `path.name/rel_path` straight through from
`pathlib`, which itself passes through whatever `os.scandir` returns with no
normalization. Confirmed empirically as well as by reading: this machine's
APFS temp filesystem is normalization-PRESERVING, not normalization-forcing
(unlike historical HFS+) -- creating a file with an NFD-encoded name and
listing the directory returns that exact NFD form back, unchanged. So the
defect is real on the platform this suite actually runs on, not a hypothetical
cross-platform-only concern.

FALSIFYING INPUT CONFIRMED BY HAND (see PROOF PROTOCOL below): a file literally
named "cafe" + NFD("é") + ".py" (i.e. "cafeé.py", the
decomposed e+combining-acute-accent form of "café.py"). Walking it yields
`IndexedFile.rel_path == "cafeé.py"` verbatim --
`unicodedata.normalize("NFC", rel_path) != rel_path`. Confirmed by actually
walking a real temp directory before writing the xfail (pasted in the round
report); no source file was touched to get this result -- it is the shipped
behavior today.
"""

from __future__ import annotations

import shutil
import string
import tempfile
import unicodedata
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.config import IndexConfig
from trelix.indexing.walker import FileWalker

# Every one of these has a composed (NFC) form that differs from its canonical
# decomposition (NFC) form -- i.e. genuine precomposed Latin-1 letters with
# diacritics, each independently confirmed to satisfy
# `unicodedata.normalize("NFD", c) != c` (checked as this file's own
# precondition below, not assumed).
_NFC_NFD_DIVERGENT_CHARS = ("é", "ü", "ñ", "å", "ö", "ç", "à", "î")

_STEM_ALPHABET = string.ascii_lowercase + string.digits


class TestFixtureCharsActuallyDiverge:
    """Precondition control: if any of these chars stopped having a distinct NFD
    form, the property below would build a filename indistinguishable from its
    own NFC form and could not discriminate normalizing from not normalizing.
    """

    def test_every_fixture_char_has_a_distinct_nfd_form(self) -> None:
        for char in _NFC_NFD_DIVERGENT_CHARS:
            nfd = unicodedata.normalize("NFD", char)
            assert nfd != char, (
                f"{char!r} has no distinct NFD decomposition ({nfd!r} == {char!r}); "
                "it cannot discriminate normalization behaviour and must be replaced"
            )


class TestWalkerPassesThroughUnnormalizedFilenames:
    """Fails under the CORRECT (currently unimplemented) behaviour: a walker that
    canonicalizes filenames would make `unicodedata.normalize("NFC", rel_path)
    == rel_path` hold. `raises=AssertionError` because today's failure is a
    string mismatch, not a crash.
    """

    @pytest.mark.xfail(
        reason=(
            "DEFECT: FileWalker.walk() does not Unicode-normalize rel_path, so an "
            "NFD-encoded filename is yielded verbatim instead of canonicalized to NFC."
        ),
        raises=AssertionError,
        strict=True,
    )
    @settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @example(stem="cafe", char="é")  # the hand-verified case
    @given(
        stem=st.text(alphabet=_STEM_ALPHABET, min_size=1, max_size=8),
        char=st.sampled_from(_NFC_NFD_DIVERGENT_CHARS),
    )
    def test_nfd_encoded_filename_is_normalized_to_nfc(self, stem: str, char: str) -> None:
        nfd_char = unicodedata.normalize("NFD", char)
        filename = f"{stem}{nfd_char}.py"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            repo = tmp_dir / "repo"
            repo.mkdir()
            (repo / filename).write_text("x = 1\n", encoding="utf-8")

            config = IndexConfig(repo_path=str(repo))
            files = list(FileWalker(config).walk())

            assert len(files) == 1, (
                f"expected exactly one walked file for {filename!r}, got {len(files)}"
            )
            rel_path = files[0].rel_path

            # DESIRED (currently failing) property: rel_path is canonical NFC, so
            # cross-platform/cross-filesystem re-walks of the same logical file
            # produce an identical string.
            assert unicodedata.normalize("NFC", rel_path) == rel_path
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestWalkerContentHashIsUnaffectedControl:
    """Discriminating control: the file's content hash (SHA-256 of bytes, not of
    the name) must NOT be affected by the filename's normalization form. If this
    ever failed, the risk description above (identical content, changing
    identity) would be wrong about WHICH field drifts.
    """

    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(char=st.sampled_from(_NFC_NFD_DIVERGENT_CHARS))
    def test_hash_is_identical_across_normalization_forms_of_the_same_name(self, char: str) -> None:
        nfc_char = unicodedata.normalize("NFC", char)
        nfd_char = unicodedata.normalize("NFD", char)
        content = "x = 1\n"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            repo_nfc = tmp_dir / "repo_nfc"
            repo_nfc.mkdir()
            (repo_nfc / f"a{nfc_char}.py").write_text(content, encoding="utf-8")
            repo_nfd = tmp_dir / "repo_nfd"
            repo_nfd.mkdir()
            (repo_nfd / f"a{nfd_char}.py").write_text(content, encoding="utf-8")

            hash_nfc = list(FileWalker(IndexConfig(repo_path=str(repo_nfc))).walk())[0].hash
            hash_nfd = list(FileWalker(IndexConfig(repo_path=str(repo_nfd))).walk())[0].hash

            assert hash_nfc == hash_nfd
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
