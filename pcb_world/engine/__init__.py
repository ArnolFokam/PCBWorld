"""KiCad API layer: the sole interface to C++ bindings.

Modules outside this package should never import kicad_rl_router directly.

Where the router comes from is decided in ONE place, :func:`router_lib_dir` (precedence:
``PCBWORLD_KICAD_RL_MODULE_DIR`` → ``PCBWORLD_KICAD_RL_BUILD_DIR`` → this tree's ``build_rl`` →
discovery of a build of this tree's C++, ``provenance.resolve_build_dir``), and whether that
router may be loaded is decided in ONE place, :func:`ensure_router_provenance`, which both load
sites (the IPC server spawn and the in-process import) call and nothing else. An explicit
module or build dir is used as given, and guarded all the same.

The cheap cases (a variable set, or a ``build_rl`` of our own) resolve at import, as before.
Discovery runs ``git`` and the hash script, so it is deferred to first use: a plain
``import pcb_world.engine`` spawns no subprocess. A discovery that cannot even hash this
tree's C++ (the bundle's hash script fails) leaves the probe ``engine_available()`` False
with a warning and makes ``ensure_router_provenance`` raise — loud where an engine is used,
not at import.
"""

import os as _os
import sys as _sys
import warnings as _warnings

_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_OWN_BUILD_DIR = _os.path.join(_PROJECT_ROOT, "build_rl")

# (lib_dir, build_dir, problem) once resolved. problem is set only when discovery ran and could
# not deliver a usable build although builds exist (every one from other C++, or the hash script
# failed): the guard raises it. None = nothing to refuse (found, or nothing built at all).
_resolved: "tuple[str, str, str | None] | None" = None
_guarded = False


def _needs_discovery() -> bool:
    """No explicit module/build dir and no ``build_rl`` of our own (a git worktree)."""
    return (not _os.environ.get("PCBWORLD_KICAD_RL_MODULE_DIR")
            and "PCBWORLD_KICAD_RL_BUILD_DIR" not in _os.environ
            and not _os.path.isdir(_OWN_BUILD_DIR))


def _resolve() -> "tuple[str, str, str | None]":
    global _resolved
    if _resolved is not None:
        return _resolved
    module_dir = _os.environ.get("PCBWORLD_KICAD_RL_MODULE_DIR")
    build_dir = _os.environ.get("PCBWORLD_KICAD_RL_BUILD_DIR") or _OWN_BUILD_DIR
    problem = None
    if _needs_discovery():
        # Find a build of THIS tree's C++ among the known places — the main clone's build_rl,
        # var/builds/* snapshots (provenance.resolve_build_dir). Paid only on this path.
        from pcb_world.engine.provenance import RouterProvenanceError, resolve_build_dir
        from pcb_world.engine.router_client import engine_home
        try:
            found, problem = resolve_build_dir(_PROJECT_ROOT, engine_home())
        except RouterProvenanceError as e:            # the hash script failed: nothing can be verified
            found, problem = None, str(e)
            _warnings.warn(f"router discovery skipped — {e}", RuntimeWarning, stacklevel=2)
        if found:
            build_dir = found
    if module_dir:
        lib_dir = module_dir
    else:
        from pcb_world.engine.provenance import lib_dir_of
        lib_dir = lib_dir_of(build_dir)
    if _os.path.isdir(lib_dir) and lib_dir not in _sys.path:
        _sys.path.insert(0, lib_dir)
    _resolved = (lib_dir, build_dir, problem)
    return _resolved


def router_lib_dir() -> str:
    """Directory the ``kicad_rl_router`` extension is loaded from (memoised per process; put on
    ``sys.path`` when it exists). Precedence: see the module docstring."""
    return _resolve()[0]


def router_build_dir() -> str:
    """Build dir the IPC server is pointed at (its ``PCBWORLD_KICAD_RL_BUILD_DIR``): the explicit
    variable, else this tree's ``build_rl``, else the discovered build."""
    return _resolve()[1]


def ensure_router_provenance() -> None:
    """Refuse a router built from other C++ than this tree's — the ONE guard call, made by both
    load sites right before a router is loaded. Once per process: a verified router stays
    verified; a refusal re-raises on every retry without re-hashing (``provenance.source_hash``
    is memoised). Applies to an explicit ``PCBWORLD_KICAD_RL_MODULE_DIR`` too. When discovery
    found builds but every one was made from other C++, that reason is the error, instead of
    the generic "no module named kicad_rl_router" the import would give."""
    global _guarded
    if _guarded:
        return
    from pcb_world.engine import provenance
    from pcb_world.engine.router_client import engine_home
    lib_dir, _, problem = _resolve()
    if problem is not None:
        raise provenance.RouterProvenanceError(problem)
    provenance.check_router_matches_sources(lib_dir, engine_home())
    _guarded = True


if not _needs_discovery():
    _resolve()      # cheap (no discovery on this path): resolve and extend sys.path at import


def engine_available() -> bool:
    """True when the C++ router build is present on this host.

    Checks for the ``kicad_rl_router`` extension file WITHOUT importing it —
    the import would load the GPL shared library into this (BSD-3, environment-side) process,
    which engine-IPC mode exists to avoid. This is the availability probe
    tests must use instead of ``import kicad_rl_router``.
    """
    import glob as _glob
    return bool(_glob.glob(_os.path.join(router_lib_dir(), "kicad_rl_router*")))


from pcb_world.engine.kicad_engine import (
    KiCadEngine,
    CLEANUP_FINALIZE,
    CLEANUP_TOPOLOGY_PRESERVING,
)
from pcb_world.engine.containers import (
    BoardMeta,
    BoardSnapshot,
    CleanupResult,
    DRCResult,
    RewardSnapshot,
    RoutingSessionState,
)
from pcb_world.engine.drc import DRCUtils
from pcb_world.engine.router_client import engine_home


__all__ = [
    "KiCadEngine",
    "engine_available",
    "engine_home",
    "router_lib_dir",
    "router_build_dir",
    "ensure_router_provenance",
    "BoardMeta",
    "BoardSnapshot",
    "RoutingSessionState",
    "CleanupResult",
    "CLEANUP_TOPOLOGY_PRESERVING",
    "CLEANUP_FINALIZE",
    "DRCResult",
    "RewardSnapshot",
    "DRCUtils"
]
