"""Determinism gate for the CP2-V-71101 geometry build.

Two separate things are tested here.

The cheap tests exercise the fingerprint itself, including a negative control
built in a temporary directory. Without a negative control a fingerprint that
returns None for everything would pass every comparison, which is exactly how a
name/bounds/triangle-count fingerprint previously reported two different builds
as equivalent.

The expensive test builds the real model twice - once in source order, once with
the proxy visit order shuffled - and requires the emitted artifacts to be
identical, not merely equivalent. It is the gate for the canonical model, so it
runs by default when the build input and gltfpack are present; set
TWIN_SKIP_BUILD_REGRESSION=1 to skip it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts" / "build_twin_pipeline.py"
IFC = REPO_ROOT / "geometry" / "build_input.ifc"


def _load_fingerprint_module():
    path = REPO_ROOT / "scripts" / "glb_fingerprint.py"
    spec = importlib.util.spec_from_file_location("glb_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gf = _load_fingerprint_module()


# --------------------------------------------------------------------------
# helpers: build a minimal, fully readable glTF so the fingerprint can be
# tested without the 27 MB production intermediate
# --------------------------------------------------------------------------

def _write_minimal_gltf(dirpath: Path, positions, indices, name="probe") -> Path:
    """Write a one-mesh glTF with an external .bin and return the .gltf path."""
    dirpath.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    pos_off = 0
    for p in positions:
        blob += struct.pack("<3f", *p)
    idx_off = len(blob)
    for i in indices:
        blob += struct.pack("<I", i)
    (dirpath / "probe.bin").write_bytes(bytes(blob))

    lo = [min(p[k] for p in positions) for k in range(3)]
    hi = [max(p[k] for p in positions) for k in range(3)]
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "probe.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(positions) * 12},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(indices) * 4},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions),
             "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": 1, "componentType": 5125, "count": len(indices),
             "type": "SCALAR"},
        ],
        "materials": [{"name": "mat_probe"}],
        "meshes": [{"name": f"mesh_{name}", "primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "nodes": [{"name": name, "mesh": 0, "extras": {"elementCount": 1}}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    out = dirpath / "probe.gltf"
    out.write_text(json.dumps(gltf), encoding="utf-8")
    return out


_TRI = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
_TRI_IDX = [0, 1, 2]


# --------------------------------------------------------------------------
# fingerprint unit tests
# --------------------------------------------------------------------------

def test_fingerprint_reads_vertices_and_indices(tmp_path):
    src = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    fp = gf.fingerprint(str(src))
    node = next(r for r in fp["nodes"] if r["name"] == "probe")
    assert fp["geometryReadable"] is True
    assert node["triCount"] == 1
    assert node["vertexCount"] == 3
    assert node["indexCount"] == 3
    for key in gf.GEOMETRY_KEYS:
        assert node[key], f"{key} must be computed for readable geometry"
    for key in gf.GEOMETRY_KEYS:
        v = node[key]
        assert len(v) == 64 and all(c in "0123456789abcdef" for c in v), \
            f"{key} must be a sha256 hex digest"
        assert v != hashlib.sha256(b"").hexdigest(), \
            f"{key} must hash real data, not an empty buffer"


def test_identical_builds_compare_exact(tmp_path):
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    b = _write_minimal_gltf(tmp_path / "b", _TRI, _TRI_IDX)
    diffs, geo = gf.compare(gf.fingerprint(str(a)), gf.fingerprint(str(b)),
                            require_vertex_exact=True)
    assert diffs == []
    assert geo == []


def test_vertex_perturbation_is_detected(tmp_path):
    """The negative control.

    Move one vertex by a small amount and the comparison must fail. If this
    passes, the fingerprint is not actually reading the geometry and every
    other assertion in this file is meaningless.
    """
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    moved = [(0.5, 0.0, 0.0)] + _TRI[1:]
    b = _write_minimal_gltf(tmp_path / "b", moved, _TRI_IDX)
    diffs, geo = gf.compare(gf.fingerprint(str(a)), gf.fingerprint(str(b)),
                            require_vertex_exact=True)
    assert diffs or geo, "a moved vertex must not compare equal"
    assert any("vertex data" in g for g in geo)


def test_index_perturbation_is_detected(tmp_path):
    """Reordering the index buffer without moving a vertex must still fail.

    Guards against a fingerprint that only hashes positions.
    """
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    b = _write_minimal_gltf(tmp_path / "b", _TRI, [0, 2, 1])
    diffs, geo = gf.compare(gf.fingerprint(str(a)), gf.fingerprint(str(b)),
                            require_vertex_exact=True)
    assert diffs or geo, "a re-wired index buffer must not compare equal"
    assert any("index data" in g for g in geo)


def test_reordered_vertex_numbering_still_matches(tmp_path):
    """Renumbering the same solid must give the same topology hash.

    Vertex order and triangle order carry no engineering meaning, so the
    order-invariant `topologySha256` must be unchanged by a pure renumbering.
    `positionSha256`/`indexSha256` are exact buffer hashes and are expected to
    differ here - that is the point of the determinism gate, which requires the
    emitted buffers to be byte-identical, not merely equivalent.
    """
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    perm = [_TRI[2], _TRI[0], _TRI[1]]
    b = _write_minimal_gltf(tmp_path / "b", perm, [2, 1, 0])
    na = next(r for r in gf.fingerprint(str(a))["nodes"] if r["name"] == "probe")
    nb = next(r for r in gf.fingerprint(str(b))["nodes"] if r["name"] == "probe")
    assert na["topologySha256"] == nb["topologySha256"], (
        "the same solid must produce the same order-invariant topology hash")


def test_empty_group_nodes_are_not_a_difference(tmp_path):
    """Mesh-free group nodes must not be reported as missing geometry."""
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    b = _write_minimal_gltf(tmp_path / "b", _TRI, _TRI_IDX)
    fa, fb = gf.fingerprint(str(a)), gf.fingerprint(str(b))
    fa["nodes"].append({"name": "05_WATER_OUTLET", "positionSha256": None,
                        "indexSha256": None, "topologySha256": None,
                        "triCount": 0, "vertexCount": 0, "indexCount": 0})
    fb["nodes"].append({"name": "05_WATER_OUTLET", "positionSha256": None,
                        "indexSha256": None, "topologySha256": None,
                        "triCount": 0, "vertexCount": 0, "indexCount": 0})
    diffs, geo = gf.compare(fa, fb, require_vertex_exact=True)
    assert geo == [], f"an empty group on both sides is not a difference: {geo}"


def test_compressed_input_fails_closed(tmp_path):
    """A meshopt-compressed file must not be able to certify vertex-exactness."""
    a = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    b = _write_minimal_gltf(tmp_path / "b", _TRI, _TRI_IDX)
    fa = gf.fingerprint(str(a))
    # Simulate what the builder produces: meshopt means no readable geometry.
    fa["geometryReadable"] = False
    fb = gf.fingerprint(str(b))
    diffs, geo = gf.compare(fa, fb, require_vertex_exact=True)
    assert any("cannot be certified" in g for g in geo), (
        "compressed input must block a vertex-exact claim")


def test_load_fingerprint_does_not_treat_gltf_as_fingerprint_json(tmp_path):
    """A .gltf parses as JSON; it must be fingerprinted, not read back as one.

    Returning the glTF document itself made every field None and silently
    turned the whole comparison into a no-op.
    """
    src = _write_minimal_gltf(tmp_path / "a", _TRI, _TRI_IDX)
    doc = gf.load_fingerprint(str(src))
    assert "geometrySha256" in doc
    node = next(r for r in doc["nodes"] if r["name"] == "probe")
    assert node["positionSha256"]


# --------------------------------------------------------------------------
# the real gate
# --------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_build(outdir: Path, seed=None):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(BUILDER),
           "--ifc", str(IFC),
           "--hierarchy", str(outdir / "hierarchy.json"),
           "--inventory", str(outdir / "inventory.json"),
           "--glb", str(outdir / "out.glb"),
           "--intermediates", str(outdir / "raw_extracted.gltf")]
    if seed is not None:
        cmd += ["--shuffle-proxies", str(seed)]
    env = dict(os.environ)
    env["TWIN_DUMP_ELEMENTS"] = str(outdir / "elements.json")
    env["TWIN_DUMP_DUPGROUPS"] = str(outdir / "dupgroups.json")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                         capture_output=True, text=True, timeout=7200)
    assert res.returncode == 0, f"build failed:\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return outdir


def _normalized_hierarchy(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["gltf"] = "<OUTPUT_GLB>"  # the two builds write to different paths
    return doc


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("TWIN_SKIP_BUILD_REGRESSION") == "1",
                    reason="TWIN_SKIP_BUILD_REGRESSION=1")
@pytest.mark.skipif(not IFC.exists(), reason="build_input.ifc not present")
@pytest.mark.skipif(shutil.which("gltfpack") is None, reason="gltfpack not on PATH")
def test_shuffled_build_is_byte_identical_to_normal(tmp_path_factory):
    base = tmp_path_factory.mktemp("determinism")
    normal = _run_build(base / "normal")
    shuffled = _run_build(base / "shuffled", seed=777)

    # 1. emitted geometry, byte for byte
    for rel in ("raw_extracted.gltf/raw_extracted.gltf",
                "raw_extracted.gltf/raw_extracted.bin",
                "out.glb"):
        a, b = normal / rel, shuffled / rel
        assert a.exists() and b.exists(), f"missing artifact {rel}"
        assert _sha(a) == _sha(b), f"{rel} differs between normal and shuffled"

    # 2. the inventory must be byte identical, not merely equal in totals
    assert _sha(normal / "inventory.json") == _sha(shuffled / "inventory.json")

    # 2a. Byte identity is only meaningful if the row order is actually a
    # function of content. If the ordering key left ties, a stable sort would
    # fall back to visit order and the byte comparison above would be comparing
    # two orderings that happen to coincide. Assert the key is total instead.
    inv = json.loads((normal / "inventory.json").read_text(encoding="utf-8"))
    rows = inv["unresolved"] + inv["excluded"]
    keyfields = ("source_entity_name", "source_object_type", "source_branch",
                 "center", "reason", "geometry_key",
                 "source_global_id", "source_entity_id")
    keys = [tuple(json.dumps(r.get(f), sort_keys=True) for f in keyfields)
            for r in rows]
    assert len(set(keys)) == len(keys), (
        f"inventory ordering key is not total: "
        f"{len(keys) - len(set(keys))} tied row(s); a shuffled run could "
        f"number those rows differently")

    # 2b. Temporary ids are assigned after ordering, so they must be unique and
    # must follow the same order in both runs.
    tids = [r["temporary_id"] for r in inv["unresolved"]]
    assert len(set(tids)) == len(tids), "duplicate temporary_id in unresolved rows"
    inv_b = json.loads((shuffled / "inventory.json").read_text(encoding="utf-8"))
    assert tids == [r["temporary_id"] for r in inv_b["unresolved"]]

    # 2c. Every row is accountable: nothing was silently dropped.
    assert inv["accounting"]["unaccounted"] == 0

    # 2d. Statuses must come from the declared vocabulary, so an excluded item
    # can never be reported as a resolved one.
    #
    # The vocabulary is asserted against the builder's own declarations rather
    # than a copy pasted here, so the test cannot drift away from the
    # implementation it is meant to police.
    src = BUILDER.read_text(encoding="utf-8")
    # The builder declares two vocabularies: a `status` for an individual
    # deferred row (is its identity known?), and a `geometry_state` for an
    # emitted leaf (is it confirmed core?). Both are read from the source of
    # truth so this test cannot silently drift from the implementation.
    def _values(prefix):
        return set(re.findall(
            r"^" + prefix + r"[A-Z_]+\s*=\s*[\"']([^\"']+)[\"']", src, re.M))

    declared = _values("DEFERRED_STATUS_")
    states = _values("GEOM_STATE_")
    assert declared and states, (
        "could not read the status/geometry_state vocabularies from the "
        "builder; both must be declared in one place")

    used = {r["status"] for r in rows}
    assert used <= declared, (
        f"non-conforming statuses: {used - declared}; declared vocabulary is "
        f"{sorted(declared)}")

    # Exclusion must be its own value, never a synonym for resolved.
    assert "UNRESOLVED" in declared
    # The lifecycle states must still cover retirement, so a future item can be
    # marked LEGACY/SCRATCH explicitly instead of silently disappearing.
    assert {"LEGACY", "SCRATCH"} <= states, (
        f"the retirement states must remain declared; got {sorted(states)}")
    assert {"CORE_CONFIRMED", "DEFERRED_REVIEW"} <= states
    assert all(r.get("source_global_id") is not None for r in rows), \
        "every accounted row must carry its source GlobalId"

    # 3. the hierarchy must match apart from the recorded output path
    assert _normalized_hierarchy(normal / "hierarchy.json") == \
           _normalized_hierarchy(shuffled / "hierarchy.json")

    # 4. element identity, one record per emitted mesh part
    ea = json.loads((normal / "elements.json").read_text(encoding="utf-8"))
    eb = json.loads((shuffled / "elements.json").read_text(encoding="utf-8"))
    assert ea == eb, "per-element identity, ordering or index hash differs"
    assert ea, "element dump is empty; the gate would be vacuous"

    # 5. explicit vertex/index-exact comparison of the emitted geometry
    fa = gf.fingerprint(str(normal / "raw_extracted.gltf" / "raw_extracted.gltf"))
    fb = gf.fingerprint(str(shuffled / "raw_extracted.gltf" / "raw_extracted.gltf"))
    diffs, geo = gf.compare(fa, fb, require_vertex_exact=True)
    assert diffs == [], f"metadata differs: {diffs}"
    assert geo == [], f"geometry differs: {geo}"
    assert fa["geometryReadable"], "geometry must be readable to certify exactness"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("TWIN_SKIP_BUILD_REGRESSION") == "1",
                    reason="TWIN_SKIP_BUILD_REGRESSION=1")
@pytest.mark.skipif(not IFC.exists(), reason="build_input.ifc not present")
@pytest.mark.skipif(shutil.which("gltfpack") is None, reason="gltfpack not on PATH")
def test_every_collapsed_duplicate_is_vertex_and_index_identical(tmp_path_factory):
    """A collapse is only legitimate if the geometry is genuinely identical.

    Collapsing on a vertex-only hash would silently delete real geometry that
    happens to reuse a vertex array with a different wiring.

    Two kinds of collapse are audited, matched against the inventory accounting
    so the audit can never be vacuous:
      * TRUE_GEOMETRIC_DUPLICATE  - distinct proxies with byte-identical
        geometry. `dupgroups.json` holds independent vertex+index evidence for
        each group, computed from the geometry directly rather than from the
        builder's own content hash.
      * REPEATED_SUB_ITEM         - one proxy that repeats the same emitted
        mesh. Each collapse carries `identical_sub_item_key`, the exact
        vertex+index identity key that justified it.
    On the SEP export there are no true geometric duplicates (only repeated
    sub-items), so the true-duplicate group must be empty and the accounting
    must agree - a mismatch means geometry was collapsed without evidence.
    """
    base = tmp_path_factory.mktemp("dupaudit")
    out = _run_build(base / "build")
    groups = json.loads((out / "dupgroups.json").read_text(encoding="utf-8"))
    inv = json.loads((out / "inventory.json").read_text(encoding="utf-8"))
    acct = inv["accounting"]
    dup_rows = inv.get("duplicates_collapsed", [])

    # Independent byte-identity evidence for the true-duplicate groups.
    for g in groups:
        verts = {m["vertexSha256"] for m in g["members"]}
        idxs = {m["indexSha256"] for m in g["members"]}
        assert len(verts) == 1, (
            f"group {g['content_hash']} members disagree on vertices")
        assert len(idxs) == 1, (
            f"group {g['content_hash']} members disagree on indices")
        # survivor must be the smallest GlobalId, decided by source identity
        expected = sorted((m["source_global_id"] or "") for m in g["members"])[0]
        assert (g["survivor"] or "") == expected, (
            f"survivor for {g['content_hash']} is not min(GlobalId)")
        # each group of N members accounts for exactly N-1 true duplicates
        assert g["memberCount"] >= 2

    # The accounting must reconcile with the evidence: the number of collapsed
    # true duplicates equals the total surplus across the audited groups, and
    # no collapse may exist that the evidence dump does not explain.
    true_dup_expected = sum(g["memberCount"] - 1 for g in groups)
    assert acct["collapsed_true_duplicates"] == true_dup_expected, (
        f"accounting says {acct['collapsed_true_duplicates']} true duplicates "
        f"but the evidence dump explains {true_dup_expected}")

    # Every REPEATED_SUB_ITEM collapse must carry its exact identity key, and
    # the count must reconcile with the accounting too.
    repeated = [r for r in dup_rows if r["reason"] == "REPEATED_SUB_ITEM"]
    true_rows = [r for r in dup_rows if r["reason"] == "TRUE_GEOMETRIC_DUPLICATE"]
    assert len(repeated) == acct["collapsed_duplicate_sub_items"], (
        f"{len(repeated)} repeated-sub-item collapses out of "
        f"{acct['collapsed_duplicate_sub_items']} accounting rows")
    for r in repeated:
        assert r.get("identical_sub_item_key"), (
            f"{r['source_entity_name']} was collapsed without identity evidence")
    assert len(true_rows) == acct["collapsed_true_duplicates"]

    # The audit must not be vacuous and must never outrun the accounting.
    assert (groups or repeated), "no collapse of either kind; the audit is vacuous"
    assert len(dup_rows) == acct["collapsed_true_duplicates"] + \
        acct["collapsed_duplicate_sub_items"]

    assert acct["unaccounted"] == 0
    assert acct["emitted_source_elements"] <= acct["emitted_mesh_parts"]
