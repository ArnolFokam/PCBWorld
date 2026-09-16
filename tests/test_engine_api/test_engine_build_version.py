"""Build provenance: the loaded ``kicad_rl_router.so`` must match the source engine
version, so a stale build (source patched + bumped, but not rebuilt) is caught early
instead of silently producing wrong results in every downstream engine test.

Second marker, same idea one level finer: ``ENGINE_CPP_HASH`` is the content hash of
``engine/kicad-patches/`` (``engine/tools/cpp_content_hash.sh``, the one implementation)
written next to the ``.so`` at build time. The version cannot see a C++ EDIT — editing a
patch leaves ``ENGINE_VERSION`` untouched until someone bumps — so without this marker a
tree happily runs every engine test against a router built from other sources.

Mechanism:
  - ``engine/kicad-patches/ENGINE_VERSION`` holds the engine version (MAJOR.MINOR),
    written on every version bump.
    String equality is the contract: any difference means the built
    router no longer matches the source engine identity.
  - ``engine/build_rl_router.sh`` copies that file next to the built ``.so`` as a
    stamp (a marker only — no runtime code reads it).
  - This test compares the two. A mismatch (usually source > stamp: bumped but not yet
    rebuilt) fails with a rebuild instruction.
  - The C++ hash goes through the runtime's own ``provenance.source_hash`` (the code path
    the load-time guard uses), and a router WITHOUT the stamp fails here: the runtime
    tolerates one with a warning (old builds in the field), the suite is the rebuild gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pcb_world.engine import engine_home, provenance

# The engine is a separate repository, pinned as the engine/ submodule.
SRC_VERSION_FILE = Path(engine_home()) / "kicad-patches" / "ENGINE_VERSION"
HASH_TOOL = Path(engine_home()) / "tools" / "cpp_content_hash.sh"
REBUILD_HINT = "rebuild the router: `bash engine/build_rl_router.sh`"


def _read_version(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _module_dir() -> Path:
    """Directory of the actually-loaded router .so (conftest puts the resolved lib dir on the path)."""
    import kicad_rl_router

    return Path(kicad_rl_router.__file__).resolve().parent


def test_engine_build_matches_source_version() -> None:
    assert SRC_VERSION_FILE.exists(), (
        f"missing source engine-version file {SRC_VERSION_FILE} — "
        f"recreate it with the current MAJOR.MINOR."
    )
    source = _read_version(SRC_VERSION_FILE)

    stamp_file = _module_dir() / "ENGINE_VERSION"
    if not stamp_file.exists():
        pytest.fail(
            f"the built router at {stamp_file.parent} carries no ENGINE_VERSION stamp — "
            f"it predates the build-versioning mechanism (source is {source!r}). {REBUILD_HINT}."
        )
    stamp = _read_version(stamp_file)

    assert stamp == source, (
        f"engine build is stale: source ENGINE_VERSION is {source!r} but the built .so "
        f"was stamped {stamp!r}"
        + (" (source is ahead — a minor/major bump landed without a rebuild)"
           if source > stamp else " (unexpected: build is ahead of source)")
        + f". {REBUILD_HINT}."
    )


def test_engine_build_matches_source_cpp_content() -> None:
    """A C++ edit without a rebuild is caught here, whatever the version says."""
    assert HASH_TOOL.is_file(), f"missing {HASH_TOOL} — the engine bundle is incomplete"
    stamp_file = _module_dir() / provenance.STAMP_NAME
    if not stamp_file.exists():
        pytest.fail(f"the router at {stamp_file.parent} carries no {provenance.STAMP_NAME} — it predates "
                    f"the C++ content stamp and cannot be verified against the sources. {REBUILD_HINT}.")
    stamp = stamp_file.read_text(encoding="utf-8").strip()
    source = provenance.source_hash(engine_home())
    assert stamp == source, (
        f"engine build is stale: engine/kicad-patches/ hashes to {source!r} but the built .so "
        f"was stamped {stamp!r} — C++ sources changed since it was built. {REBUILD_HINT}."
    )


BUILD_SCRIPT = Path(engine_home()) / "build_rl_router.sh"


def test_build_script_hashes_the_sources_before_copying_them():
    """The stamp must name the sources the build was made from: ``build_rl_router.sh`` takes
    the hash BEFORE the source tree is synced into the build and writes that value at the end.
    Hashing at stamp time instead would stamp an edit made during the multi-minute compile as
    built, and the load-time guard would then pass a router made from other sources."""
    script = BUILD_SCRIPT.read_text(encoding="utf-8")
    hash_call = 'CPP_HASH="$(bash "$BUNDLE_DIR/tools/cpp_content_hash.sh"'
    first_copy = 'rsync -a "$KICAD_SRC_ORIG/"'
    stamp_write = '"$CPP_HASH" > "$STAMP_DIR/ENGINE_CPP_HASH"'
    for needle in (hash_call, first_copy, stamp_write):
        assert needle in script, f"build_rl_router.sh no longer contains {needle!r}"
    assert script.index(hash_call) < script.index(first_copy) < script.index(stamp_write), (
        "build_rl_router.sh must compute CPP_HASH before the first source copy and write that "
        "value as the ENGINE_CPP_HASH stamp at the end")
    assert script.count('bash "$BUNDLE_DIR/tools/cpp_content_hash.sh"') == 1, (
        "the hash is taken once, up front — a second call at stamp time would re-hash the tree "
        "after the compile")
