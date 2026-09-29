"""N1 correction regression suite.

Locks the outcome of removing the radial deformation from engineering element
`/CP2-V-71101/N1` (and N2/N3) in `scripts/build_twin_pipeline.py`, and promotes
N1 to its own addressable, cutaway-clipped GLB node.

The engineering source dimensions for N1 are 1.495 m x 1.090 m x 1.090 m. The
GLB is written with `-cc` (meshopt compression) after KHR_mesh_quantization, so
extents are compared with a small tolerance rather than exactly.
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GLB_PATH = os.path.join(REPO, "simulator", "stage1", "assets", "final_twin.glb")
TEMPLATE_PATH = os.path.join(
    REPO, "simulator", "stage1", "assets", "twin3d_template.html")
BUILDER_PATH = os.path.join(REPO, "scripts", "build_twin_pipeline.py")
SELECTION_MANIFEST_PATH = os.path.join(REPO, "geometry", "selection_manifest.json")

# Selector kinds the battery-limit manifest is allowed to use. Every one of them
# is a source attribute; none of them is an IFC entity id.
ALLOWED_SELECTOR_KINDS = frozenset({
    "name_token",      # substring of the source Name attribute
    "tag_token",       # engineering tag token inside the source Name
    "branch_line",     # engineering line designator, e.g. P-711101
    "object_type",     # IFC ObjectType attribute
    # Exact PDMS line designator, e.g. D-711001. Same engineering identity as
    # `branch_line`, but matched on the full normalized designator so the
    # service letter is preserved: `D-711001` (drain) must never match
    # `P-711001` (gas outlet). Both forms occur in the IFC: lines that are
    # ATTACHMENT branches carry `/P-711101-...`, while the vessel drain line
    # carries bare `/D711001002`-style products. `branch_line` alone silently
    # dropped the drain line, which is why `pdms_line_id` exists.
    "pdms_line_id",
    # EXACT Navisworks SelectionID, matched against IfcRoot.GlobalId. This is
    # NOT IFC entity numbering, which is what the guard above exists to stop:
    # renumbering the input cannot move geometry, because a SelectionID is a
    # stable Navisworks identity that BIMCamel copies verbatim. It exists solely
    # for source proxies that carry no name at all - their manifest DisplayName
    # is literally "Cylinder" - so every name-based selector misses them and they
    # would otherwise be unrecoverable. Matching is exact string equality, never
    # a pattern, because '$' is a legal InstanceGuid character.
    "selection_id",
})
BUILD_INPUT_PATH = os.path.join(REPO, "geometry", "build_input.ifc")
MANIFEST_PATH = os.path.join(REPO, "geometry", "build_input.manifest.json")

# Authentic engineering source extents for the N1 nozzle stub.
N1_SOURCE_DIMS_M = (1.495, 1.090, 1.090)
N1_ELEMENT_COUNT = 1
N1_TRI_COUNT = 240

# M1A/M1B, the vessel's two manholes, ARE modelled. Engineering inspection
# 2026-09-28 confirmed the shared source name /CP2-V-71101/M1A/B under
# /CP2-V-71101/NOZZLES is real access hardware that belongs in the model. In
# the SEP export each manhole is a SINGLE instance (the old whole-field export
# carried two rows per manhole), so modelling them puts two elements and 3,584
# triangles back into the container holding the vessel nozzles and saddles.
#
# These same numbers previously described an EXCLUSION, when the manholes were
# wrongly taken for access-envelope proxies and removed. The regression that
# matters now is the opposite assertion: they must be present, not absent.
M1A_M1B_ELEMENT_COUNT = 2
M1A_M1B_TRI_COUNT = 3584

# The M1B manhole is modelled with its cover swung open. Four handrail sections
# of the access platform at that deck level physically intersect the open cover,
# and those four - and only those four - are removed by declared rule.
#
# Recounted 2026-09-28 (SEP-export authority) from
# geometry/emitted_entity_provenance.json: 671 emitted sub-items carry HANDRAIL
# in their name in total, of which 273 are under /PRIMARY-P1-P1B-TS01-HANDRAILS
# and 111 under /PLATFORM-P1-P1B-EM01-HANDRAILS; the rest belong to other
# handrail assemblies such as /PR-PR2-GM01-HANDRAILS. An earlier version of this
# file said "686 sections of /PRIMARY-P1-P1B-TS01-HANDRAILS and all 496 of
# /PLATFORM-P1-P1B-EM01-HANDRAILS", which conflated the all-assembly total with
# the per-assembly count and overstated the platform count by 4.5x. The numbers
# above are the measured ones. The access platform is permanent structure and
# the removal is the clashing run, not the railing.
CLASHING_HANDRAIL_SECTIONS = (55, 269, 275, 276)
CLASHING_HANDRAIL_ELEMENT_COUNT = 4
CLASHING_HANDRAIL_TRI_COUNT = 11888
EMITTED_HANDRAIL_SUBITEMS_TOTAL = 671
EMITTED_HANDRAIL_SUBITEMS_PRIMARY_P1B_TS01 = 273
EMITTED_HANDRAIL_SUBITEMS_PLATFORM_P1B_EM01 = 111

# Post-correction container figures, locked as the regression baseline on the
# SEP-export authority.
#
# The container holds the vessel nozzles, saddles and the M1A/M1B manholes. On
# the SEP export it measured 61 elements / 10,943 triangles (including 2
# manhole elements / 3,584 triangles). The pre-correction figures are defined
# by the transfer itself, not chosen: they are the same container with N1 still
# inside it, so they are exactly the post-correction figures plus
# N1_ELEMENT_COUNT / N1_TRI_COUNT. Keeping the relationship (rather than
# hardcoding unrelated numbers) is what makes the delta below meaningful.
CONTAINER_ELEMENT_COUNT = 61
CONTAINER_TRI_COUNT = 10943
PRE_CORRECTION_CONTAINER_ELEMENT_COUNT = CONTAINER_ELEMENT_COUNT + N1_ELEMENT_COUNT
PRE_CORRECTION_CONTAINER_TRI_COUNT = CONTAINER_TRI_COUNT + N1_TRI_COUNT

# The six level-instrument interface nozzles, placed in their own leaf.
LEVEL_INTERFACE_ELEMENT_COUNT = 6
LEVEL_INTERFACE_TRI_COUNT = 1808

# Validated scene triangle total for this extraction. Asserting the absolute
# total is what catches geometry arriving without being inventoried, or being
# inventoried without being emitted.
#
# The total is the SEP-export reset value: the build input switched from the
# 4,645,005,194-byte whole-field export to the scoped first-separator export
# (docs/Mj-Real-Data/navisworks/New_model/1st_Sperator_area.ifc), so the total
# dropped from 1,570,091 to 1,368,613 because out-of-scope structure left. The
# inventory's own accounting closes to unaccounted: 0.
#
# 2026-09-28: raised to 1,380,925 by the inlet valve-station completion. The
# spatial_crop tx upper bound went 21.5 -> 28.5, admitting +366 proxies, which
# brought in UZV-051, UZV-052, RO-051 and PDZT-051 (+12,312 triangles) plus their
# cables and platform steel. This is a deliberate scope increase, not drift: see
# test_the_inlet_valve_station_is_complete_and_addressable for the item-level
# proof and test_no_declared_leaf_is_starved for the guard that caught the crop
# being hardcoded in the builder.
VALIDATED_SCENE_TRIANGLE_TOTAL = 1381181

# The scene total above is the sum over ALL emitted leaves, so it legitimately
# grows when a previously deferred line is promoted to proven core on new
# evidence. It is NOT a check that core classification is conservative: this
# test would pass just as happily if the builder emitted 1,583,143 triangles of
# unproven geometry. That is the job of the two tests below, which are the
# actual guard for the CP2-V-71101 core work.
#
# Promotion history, each step traceable to a source in docs/index/extracts:
#   1,556,171  pre-core baseline (naming-reconciled, deduped, pruned)
#   1,583,143  P&ID-proven service piping promoted: 02/03/04/05/06
#              +26,972 triangles, 99_REVIEW_UNCLASSIFIED correspondingly
#              reduced. Net core gain, not new invented geometry.
PROVEN_PIPING_TRIANGLE_DELTA = 1583143 - 1556171  # = 26,972

# Non-core geometry must stay bounded and must not be mistaken for core. This
# is the population that still lacks a placement decision.
DEFERRED_REVIEW_LEAF = "service_piping:99_REVIEW_UNCLASSIFIED"


@pytest.fixture(scope="module")
def glb():
    import pygltflib
    if not os.path.exists(GLB_PATH):
        pytest.skip(f"canonical GLB not present at {GLB_PATH}")
    return pygltflib.GLTF2().load(GLB_PATH)


def _trs(node):
    if node.matrix is not None:
        return np.array(node.matrix, dtype=float).reshape(4, 4).T
    T = np.eye(4)
    if node.translation:
        T[:3, 3] = node.translation
    if node.scale:
        T[:3, :3] = np.diag(node.scale)
    if node.rotation:
        x, y, z, w = node.rotation
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T[:3, :3] = R @ T[:3, :3]
    return T


def _subtree_aabb(gltf, name):
    """World-space AABB and triangle count for a node and its mesh descendants.

    Uses accessor min/max rather than decoding vertex buffers: the asset is
    emitted with per-node scale/translation only, so accessor extents are exact
    and avoid depending on quantised component types.
    """
    matches = [i for i, n in enumerate(gltf.nodes) if n.name == name]
    assert len(matches) == 1, f"expected exactly one node named {name!r}, got {len(matches)}"
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    tris = 0
    stack = [(matches[0], np.eye(4))]
    while stack:
        idx, parent = stack.pop()
        node = gltf.nodes[idx]
        world = parent @ _trs(node)
        if node.mesh is not None:
            for prim in gltf.meshes[node.mesh].primitives:
                acc = gltf.accessors[prim.attributes.POSITION]
                if acc.min is None or acc.max is None:
                    continue
                corners = np.array([acc.min, acc.max], dtype=float)
                moved = (world[:3, :3] @ corners.T).T + world[:3, 3]
                lo = np.minimum(lo, moved.min(axis=0))
                hi = np.maximum(hi, moved.max(axis=0))
                if prim.indices is not None:
                    tris += gltf.accessors[prim.indices].count // 3
        for child in (node.children or []):
            stack.append((child, world))
    return lo, hi, hi - lo, tris


# --- Node presence and identity ---

def test_n1_is_its_own_unique_addressable_node(glb):
    """N1 must be promoted out of vessel_nozzles_saddles into its own node."""
    names = [n.name for n in glb.nodes if n.name == "nozzle:N1"]
    assert len(names) == 1, "nozzle:N1 must exist exactly once in final_twin.glb"

    node = [n for n in glb.nodes if n.name == "nozzle:N1"][0]
    assert node.extras.get("category") == "nozzle:N1"
    assert node.extras.get("elementCount") == N1_ELEMENT_COUNT
    assert node.extras.get("triCount") == N1_TRI_COUNT
    assert "center" in node.extras


def test_n1_geometry_matches_engineering_source(glb):
    """N1 extents must match the authentic source, i.e. no radial deformation."""
    lo, hi, dim, _ = _subtree_aabb(glb, "nozzle:N1")
    for axis, (actual, expected) in enumerate(zip(dim, N1_SOURCE_DIMS_M)):
        assert actual == pytest.approx(expected, abs=0.01), (
            f"nozzle:N1 {['X', 'Y', 'Z'][axis]} extent {actual:.4f} m deviates "
            f"from engineering source {expected:.4f} m")


def test_n1_geometry_was_exactly_transferred_out_of_the_container(glb):
    """The correction must move, not duplicate or drop, N1's geometry."""
    container = [n for n in glb.nodes if n.name == "vessel_nozzles_saddles"][0]
    assert container.extras.get("elementCount") == CONTAINER_ELEMENT_COUNT
    assert container.extras.get("triCount") == CONTAINER_TRI_COUNT

    # One element and 240 triangles left the container, which is exactly N1.
    assert (PRE_CORRECTION_CONTAINER_ELEMENT_COUNT - container.extras["elementCount"]
            ) == N1_ELEMENT_COUNT
    assert (PRE_CORRECTION_CONTAINER_TRI_COUNT - container.extras["triCount"]
            ) == N1_TRI_COUNT


def test_nozzle_clamp_did_not_distort_scene_totals(glb):
    """Scene triangle total must equal the validated total (redistributed)."""
    total = sum(n.extras.get("triCount", 0) for n in glb.nodes if n.extras)
    assert total == VALIDATED_SCENE_TRIANGLE_TOTAL, (
        f"scene triangle total {total} differs from the validated "
        f"{VALIDATED_SCENE_TRIANGLE_TOTAL:,}. Geometry must be emitted exactly "
        f"once and fully inventoried; a drift here means something was dropped, "
        f"duplicated, or counted without being emitted.")


# --- Core / non-core boundary (CP2-V-71101) ---
#
# The scene-total test above cannot tell core from non-core: it only proves the
# emitted population is fully accounted for. These tests assert that the
# boundary itself is evidence-backed, which is the property the core work is
# actually about.

HIERARCHY_PATH = os.path.join(REPO, "geometry", "canonical_hierarchy.json")
MANIFEST_CSV = os.path.join(REPO, "docs", "index", "extracts",
                            "cad_pid_extraction.json")
CORE_GROUPS = ("01_VESSEL", "02_INLET", "03_LIQUID_OUTLET", "04_GAS_OUTLET",
               "05_WATER_OUTLET", "06_RELIEF", "07_INSTRUMENTATION",
               "08_STRUCTURE")
SELECTION_MANIFEST_PATH = os.path.join(REPO, "geometry", "selection_manifest.json")
STATE_CORE = "CORE_CONFIRMED"
STATE_DEFERRED = "DEFERRED_REVIEW"


def _hierarchy():
    with open(HIERARCHY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Full, uncapped emitted provenance and the unknown-geometry inventory.
#
# canonical_hierarchy.json truncates `source_entities` to 50 names per leaf, so
# it cannot answer "is this name in the model at all?" for a large leaf. The
# builder also writes a complete sidecar next to the inventory; these helpers
# read that, and skip rather than fail when a scratch build is not present.

INVENTORY_PATH = os.path.join(REPO, "geometry", "unknown_geometry_inventory.json")
PROVENANCE_PATH = os.path.join(REPO, "geometry", "emitted_entity_provenance.json")


def _inventory():
    if not os.path.exists(INVENTORY_PATH):
        pytest.skip(f"inventory not present at {INVENTORY_PATH}")
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _provenance():
    if not os.path.exists(PROVENANCE_PATH):
        pytest.skip(f"emitted provenance not present at {PROVENANCE_PATH}")
    with open(PROVENANCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _emitted_source_entities():
    names = set()
    for entities in _provenance()["leaves"].values():
        names.update(entities)
    return names


def test_every_required_core_group_is_populated_and_proven():
    """No required core group may be empty; each must cite its proving source."""
    h = _hierarchy()
    groups = {g["id"]: g for g in h["groups"]}
    for gid in CORE_GROUPS:
        assert gid in groups, f"required core group {gid} is missing"
        g = groups[gid]
        assert not g["empty"], (
            f"required core group {gid} is empty. An empty required group means "
            f"the source evidence for it has not been ingested yet.")
        assert g["children"], f"core group {gid} has no children"
        assert g.get("battery_limit_evidence"), (
            f"core group {gid} has no evidence string; every placement must be "
            f"traceable to a source")
        if g.get("proved_line"):
            assert re.match(r"^[DPV]-\d{6}$", g["proved_line"]), (
                f"core group {gid} has a malformed proved_line "
                f"{g['proved_line']!r}; expected a PDMS designator such as "
                f"D-711001")


def test_deferred_geometry_is_marked_and_never_called_core():
    """Non-core leaves must be labelled DEFERRED_REVIEW, not silently included."""
    h = _hierarchy()
    leaves = [l for l in h["leaves"] if l.get("elementCount")]
    for l in leaves:
        state = l.get("geometry_state")
        assert state in (STATE_CORE, STATE_DEFERRED), (
            f"leaf {l['name']!r} has geometry_state {state!r}; every emitted "
            f"leaf must be explicitly core-confirmed or deferred")
        assert l.get("evidence"), f"leaf {l['name']!r} has no evidence string"
    deferred = [l for l in leaves if l["geometry_state"] == STATE_DEFERRED]
    assert deferred, (
        "no leaf is DEFERRED_REVIEW. The unproven population is large and real; "
        "if it vanished from the inventory, review work would be silently lost.")
    for l in deferred:
        assert l["name"] == DEFERRED_REVIEW_LEAF, (
            f"unexpected deferred leaf {l['name']!r}; deferred geometry is "
            f"expected to be quarantined in {DEFERRED_REVIEW_LEAF!r}")


def test_core_promotion_is_evidence_backed_by_the_pid_extract():
    """Every proved_line must appear in the zero-OCR P&ID extract."""
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        blob = json.dumps(json.load(f))
    h = _hierarchy()
    proved = [g for g in h["groups"] if g.get("proved_line")]
    assert len(proved) >= 5, (
        f"expected the P&ID-proven service lines to be recorded, found "
        f"{len(proved)}: {[g['id'] for g in proved]}")
    for g in proved:
        # Designators are stored with a service-letter hyphen (D-711001); the
        # P&ID prints them without one (D711001). Both spellings must be found.
        bare = g["proved_line"].replace("-", "", 1)
        assert bare in blob, (
            f"group {g['id']} claims proved_line {g['proved_line']} but that "
            f"designator does not appear in the P&ID extract; a core placement "
            f"with no source evidence is exactly the failure this blocks")


def test_proven_piping_actually_left_the_review_bucket():
    """The promotion must reduce review, not merely relabel geometry."""
    h = _hierarchy()
    leaves = {l["name"]: l for l in h["leaves"] if l.get("elementCount")}
    review = leaves[DEFERRED_REVIEW_LEAF]
    assert review["triCount"] < 165316, (
        f"{DEFERRED_REVIEW_LEAF} still holds {review['triCount']:,} triangles; "
        f"expected the review bucket to shrink once P&ID-proven piping was "
        f"promoted to core")
    water = leaves.get("service_piping:05_WATER_OUTLET")
    assert water and water["elementCount"] > 0, (
        "05_WATER_OUTLET is empty; the P&ID proves the vessel drain line "
        "CP2-3\"-D711001-3C6N-NN and it must be placed, not force-fitted "
        "into 06_RELIEF")
    assert water["geometry_state"] == STATE_CORE


# --- Runtime cutaway wiring ---

def test_n1_is_shell_clipped_by_the_runtime_cutaway():
    """N1 internal-stub visibility is a view concern, handled by clipping."""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    match = re.search(r"const GLB_SHELL_NODES = \[(.*?)\]", template, re.S)
    assert match, "GLB_SHELL_NODES declaration not found in the runtime template"
    members = [m.strip().strip("'\"") for m in match.group(1).split(",")]
    assert "nozzle:N1" in members, (
        "nozzle:N1 must be in GLB_SHELL_NODES so the cutaway clips the "
        "PDMS internal stub instead of the builder deforming it")
    assert "vessel_shell" in members
    assert "vessel_nozzles_saddles" in members


# --- Builder source guards ---

def test_builder_no_longer_contains_the_radial_clamp():
    """Guard against the radial deformation being reintroduced."""
    with open(BUILDER_PATH, "r", encoding="utf-8") as f:
        builder = f.read()
    assert "2.12 / r_yz" not in builder, (
        "the removed radial clamp expression is back in the builder")
    assert not re.search(r"ty_v\s*\*=\s*2\.12", builder)
    assert not re.search(r"tz_v\s*\*=\s*2\.12", builder)


def test_builder_promotes_n1_before_the_generic_vessel_rule():
    """N1 must be evaluated before the generic CP2-V-71101 vessel rule.

    The rule used to live inline in the builder as a string test. Classification
    is now driven by geometry/selection_manifest.json, so the invariant is
    asserted against the manifest: `nozzle:N1` must outrank
    `vessel_nozzles_saddles`, which matches the bare `/CP2-V-71101` token and
    would otherwise swallow N1.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    leaves = {lf["name"]: lf for g in manifest["groups"] for lf in g.get("leaves", [])}

    assert "nozzle:N1" in leaves, "N1 leaf missing from the selection manifest"
    assert "vessel_nozzles_saddles" in leaves

    n1_tokens = {s["value"].upper() for s in leaves["nozzle:N1"].get("selectors", [])}
    assert "/CP2-V-71101/N1" in n1_tokens, (
        "the N1 leaf must match the full /CP2-V-71101/N1 token so unrelated "
        "nozzles are untouched")

    vessel_tokens = {s["value"].upper()
                     for s in leaves["vessel_nozzles_saddles"].get("selectors", [])}
    assert "/CP2-V-71101" in vessel_tokens, (
        "the generic vessel rule is missing from the selection manifest")

    n1_prio = leaves["nozzle:N1"].get("priority", 0)
    vessel_prio = leaves["vessel_nozzles_saddles"].get("priority", 0)
    assert n1_prio > vessel_prio, (
        "nozzle:N1 must outrank vessel_nozzles_saddles, otherwise N1 is "
        "swallowed by the generic CP2-V-71101 rule")


def test_selection_manifest_does_not_classify_by_entity_id():
    """No battery-limit selector may depend on an IFC entity id or ID range.

    Classification must be a function of source attributes (name, object type,
    branch/line identity, placement) so that renumbering the input cannot move
    geometry between groups.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for g in manifest["groups"]:
        for lf in g.get("leaves", []):
            for sel in lf.get("selectors", []):
                assert sel["kind"] in ALLOWED_SELECTOR_KINDS, (
                    f"{lf['name']} uses unsupported selector kind "
                    f"{sel['kind']!r}")
                assert "eid" not in sel["kind"].lower()
                assert "id_range" not in sel["kind"].lower()

    # The legacy ID-range mechanism must not reappear in anything the builder
    # actually reads. Prose in `known_limitations` is allowed to name the
    # removed mechanism, so only the machine-readable parts are scanned.
    machine_readable = json.dumps({
        "groups": manifest["groups"],
        "excluded_tokens": manifest.get("excluded_tokens", []),
        "tag_service_rules": manifest.get("tag_service_rules", []),
    })
    for banned in ("INLET_EID_RANGES", "eid_range", "entity_id_range",
                   "id_range"):
        assert banned not in machine_readable, (
            f"{banned} reappeared in the selection manifest's selectors; "
            "classification must not depend on IFC entity numbering")

    # A selector value that is a bare integer would be an entity id in disguise.
    for g in manifest["groups"]:
        for lf in g.get("leaves", []):
            for sel in lf.get("selectors", []):
                value = str(sel.get("value", ""))
                assert not value.isdigit(), (
                    f"{lf['name']} selector {value!r} looks like an entity id")


def test_m1a_m1b_manholes_are_modelled_and_never_excluded():
    """M1A/M1B are real access hardware, so they belong in the model.

    Engineering inspection 2026-09-28: the two manholes share the source name
    /CP2-V-71101/M1A/B under /CP2-V-71101/NOZZLES. They are real access
    hardware. An earlier pass excluded them by conflating them with the
    clearance/access-envelope proxies that the ACCESS token already covers, and
    that was wrong: it removed the manholes instead of the railing that clashes
    with the open manhole cover. This test pins the corrected intent - present in
    the model, absent from the exclusion list, absent from the excluded rows.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tokens = [t.upper() for t in manifest["excluded_tokens"]["tokens"]]
    for manhole in ("/CP2-V-71101/M1A", "/CP2-V-71101/M1B"):
        assert not any(manhole.upper() in t for t in tokens), (
            f"{manhole} is real access hardware and must not be an exclusion "
            "token; only the clashing handrail is removed")

    emitted = _emitted_source_entities()
    assert "/CP2-V-71101/M1A" in emitted
    assert "/CP2-V-71101/M1B" in emitted

    inv = _inventory()
    excluded = {r.get("source_entity_name") for r in inv["excluded"]}
    assert "/CP2-V-71101/M1A" not in excluded
    assert "/CP2-V-71101/M1B" not in excluded

    # They must be in the vessel container, and the container must carry the
    # measured manhole element/triangle counts.
    container = next(lf for lf in _hierarchy()["leaves"]
                     if lf["name"] == "vessel_nozzles_saddles")
    assert container["elementCount"] == CONTAINER_ELEMENT_COUNT
    assert container["triCount"] == CONTAINER_TRI_COUNT
    names = {s for s in container.get("source_entities") or []}
    assert "/CP2-V-71101/M1A" in names
    assert "/CP2-V-71101/M1B" in names


def test_only_the_four_clashing_handrail_sections_are_removed():
    """The M1B cover is modelled open, so the railing across it must go.

    Measured in local build coordinates, the open M1B cover swings through
    tx -3.57..-2.18, ty -0.46..0.63, tz 2.14..2.66. Four handrail sections of
    /PRIMARY-P1-P1B-TS01-HANDRAILS at that deck level (tz 2.34..2.39) occupy
    tx -2.85..-1.73 and physically intersect it. Those four are excluded by
    declared rule. M1A has no handrail within 0.8 m and needs no such rule.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tokens = [t.upper() for t in manifest["excluded_tokens"]["tokens"]]
    for section in CLASHING_HANDRAIL_SECTIONS:
        token = f"SCTN {section} OF FRMWORK /PRIMARY-P1-P1B-TS01-HANDRAILS"
        assert token in tokens, f"clashing handrail section {section} is unguarded"

    inv = _inventory()
    removed = [r for r in inv["excluded"]
               if "HANDRAIL" in str(r.get("source_entity_name", "")).upper()]
    assert len(removed) == CLASHING_HANDRAIL_ELEMENT_COUNT, (
        "only the four measured clashing handrail sections may be removed, got "
        f"{len(removed)}")
    for r in removed:
        name = r["source_entity_name"].upper()
        assert "PRIMARY-P1-P1B-TS01-HANDRAILS" in name
        assert any(f"OF SCTN {s} OF" in name for s in CLASHING_HANDRAIL_SECTIONS)
        assert r["status"] == "OUTSIDE_LOCAL_SCOPE", (
            "a removed handrail must be recorded, not silently dropped")

    # The rest of the railing must survive: the access platform is permanent
    # structure, so this is the clashing run, not the handrail family. The
    # emitted names are sub-item names ("... OF SCTN n OF FRMWORK /..."), and
    # the fence is family-scoped so a same-numbered section of another family
    # cannot mask or fake a leak.
    emitted = {s.upper() for s in _emitted_source_entities()}
    fence = "OF FRMWORK /PRIMARY-P1-P1B-TS01-HANDRAILS"
    kept = {s for s in emitted if fence in s}
    assert len(kept) >= 200, (
        f"only {len(kept)} handrail sub-items remain; the platform railing must "
        "survive except for the four clashing sections")
    for section in CLASHING_HANDRAIL_SECTIONS:
        leaked = sorted(s for s in kept if f"OF SCTN {section} " + fence in s)
        assert not leaked, (
            f"handrail SCTN {section} is excluded by rule but {len(leaked)} of "
            f"its sub-items are still emitted, e.g. {leaked[:1]}")

    structure = next(lf for lf in _hierarchy()["leaves"]
                     if lf["name"] == "structure")
    assert structure["triCount"] > 0
    assert structure["elementCount"] > 0


def test_pmjp1a_b_c_are_absent_from_the_export_and_the_token_cannot_enforce_it():
    """`/PMJP1A`/`PMJP1B`/`PMJP1C` must not be in the model - and no rule here can stop that.

    docs/DECISIONS.md P13 and docs/NAVIS-MODULE-STRUCTURE.md record the
    engineer-confirmed layout: `/PMJP1A` = access road, `/PMJP1B` = first
    separator, `/PMJP1C` = second separator, all three **tree nodes** under
    `Majnoon_NAVIS_OBS.rvm`. They are NOT layers; an earlier claim to that
    effect is withdrawn.

    The governing measurement is that all three are 0 occurrences in the crop,
    in the full 4,645,005,194-byte source AND in the 926,472-row BIMCamel
    manifest, while sibling `/PMJP1F` has 406 and `/MJP1A` 62. So OBS content
    that reached the export is the crane and civil scopes - not these nodes.

    Two consequences, and the test exists to keep both visible:

    * Today there is **nothing to remove**. The model is already clean, and the
      `PMJP1B` name token never had anything to match. Reporting this as an
      unremoved clash layer would be wrong.
    * The risk is forward-looking. A re-export with OBS fully appended puts
      that content in scope, and then exact identities (not a name token) are
      the only durable defence. The control case is the manholes below, which
      a broad `/PMJP1B` rule must never be able to delete.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tokens = [t.upper() for t in manifest["excluded_tokens"]["tokens"]]
    assert "PMJP1B" in tokens
    assert "CMJP1B" not in tokens, (
        "/CMJP1B was separately confirmed as permanent structure; it must not "
        "be swept up by a /PMJP1B rule")

    # Whatever the token can and cannot catch, nothing may be emitted carrying
    # any of the three node names.
    emitted = {s.upper() for s in _emitted_source_entities()}
    for node in ("PMJP1A", "PMJP1B", "PMJP1C"):
        leaked = sorted(s for s in emitted if node in s)
        assert not leaked, f"{node} geometry is in the model: {leaked[:3]}"

    # The declared limitation must stay visible in the artifact, so a future
    # reader cannot mistake the token for enforcement.
    note = manifest["excluded_tokens"].get("note", "")
    assert "INERT" in note, (
        "the manifest must record that the PMJP1B token does not enforce "
        "anything, otherwise it reads as a guarantee it cannot keep")

def test_the_m1a_axial_clamp_that_corrupted_geometry_is_gone():
    """The x=7.72 clamp moved vertices and was order-dependent.

    It flattened the vertices of whichever element happened to follow an /M1A
    proxy in document order - /STIFFENE-R-1 was emitted with all eight vertices
    at x=7.72 instead of its true 0.911..1.100. It was removed as dead, silently
    corrupting code. That judgement no longer depends on M1A being excluded: the
    clamp's own defect (mutating vertices of an unrelated element based on
    document order) is sufficient reason to keep it out of the builder, and
    M1A/M1B are now modelled and must render from authentic source geometry.
    """
    with open(BUILDER_PATH, "r", encoding="utf-8") as f:
        builder = f.read()
    assert "is_m1a" not in builder
    # Checked against code lines only: the constant is named in the comment
    # that records why the clamp was removed, and that is the point of it.
    code = [ln for ln in builder.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    assert not any("7.72" in ln for ln in code), (
        "the x=7.72 clamp is back in executable code")


def test_registry_nozzle_tags_are_bound_to_their_qualified_path():
    """A nozzle tag must not match as a bare substring.

    Bound bare, tag 'N1' also fired on /CP2-A-00002/N1, CHEMICAL INLET-N1 and
    N1-VENT, and tag 'S1' fired inside /P1-P1B-HS101-FD001-GR01 and
    /CP2-V-71101/LS1 - 20 foreign elements asserted into the separator's own
    nozzle leaves. The binding must use the path-qualified identity.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import geometry_selection as G

    for tag in ("N1", "N2", "N3", "S1", "D1"):
        assert G._binding_token(tag, "nozzles") == "/CP2-V-71101/" + tag
    # Longer tags are already unambiguous and keep the bare token.
    assert G._binding_token("FT-001", "instruments") == "FT-001"
    assert G._binding_token("PSV-001A", "valves") == "PSV-001A"


@pytest.mark.parametrize("foreign", [
    "/CP2-A-00002/N1",
    "/CP2-SP-SC15-1A/N1",
    "/CP2-SP-SC16-1A/N3",
    "CHEMICAL INLET-N1",
    "N1-VENT",
    "N2-VENT",
    "TANK OUTLET-N2",
    "CALIBRATION RETURN-N3",
    "N3A-LT",
    "N3B-LT",
    "PSV RETURN-S1",
    "S1-PSV RETURN",
    "/P1-P1B-HS101-FD001-GR01",
    "/P1-P1B-HS101-FD001-PD01",
])
def test_no_other_units_geometry_is_asserted_into_the_model(foreign):
    """Other equipment inside the crop must not enter the model as ours."""
    assert foreign not in _emitted_source_entities()


def test_no_bare_token_nozzle_leaves_are_emitted():
    """The four junk leaves that carried only foreign equipment must be gone."""
    leaves = set(_provenance()["leaves"])
    for gone in ("unregistered_tag:N1", "unregistered_tag:N2",
                 "unregistered_tag:N3", "unregistered_tag:S1",
                 "unregistered_tag:D1"):
        assert gone not in leaves


def test_the_six_level_instrument_nozzles_are_placed():
    """They are level-instrument nozzles, and were classified nowhere.

    tag:LT-002 listed 'LEVEL GAUGE' and 'LEVEL TRANSMITTER' in extra_names but
    its spatial_guard rejected all six, so they fell through to unclassified.
    """
    leaves = _provenance()["leaves"]
    lf = "level_interfaces:L3A_L3B_L1_L2"
    assert lf in leaves
    placed = set(leaves[lf])
    for name in ("L3A-LEVEL GAUGE", "L3B-LEVEL GAUGE", "LEVEL GAUGE-L3A",
                 "LEVEL GAUGE-L3B", "LEVEL TRANSMITTER-L1",
                 "LEVEL TRANSMITTER-L2"):
        assert name in placed

    h = _hierarchy()
    node = {lf["name"]: lf for lf in h["leaves"]}[lf]
    assert node["elementCount"] == LEVEL_INTERFACE_ELEMENT_COUNT
    assert node["triCount"] == LEVEL_INTERFACE_TRI_COUNT
    assert node["geometry_state"] == STATE_CORE


def test_the_level_nozzle_leaf_survives_the_connectivity_prune():
    """These are small isolated shells, so the prune needs an explicit opt-in.

    The connectivity prune keeps geometry something seeds or reaches. The six
    nozzles stand alone, so their leaf declares prune_exempt rather than relying
    on a name prefix to spare them.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    found = False
    for g in manifest["groups"]:
        for lf in g["leaves"]:
            if lf["name"] == "level_interfaces:L3A_L3B_L1_L2":
                assert lf.get("prune_exempt") is True
                found = True
    assert found


def test_lt002_no_longer_claims_the_level_interface_nozzles():
    """The names moved to a dedicated leaf, so they cannot be claimed twice."""
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for g in manifest["groups"]:
        for lf in g["leaves"]:
            if lf["name"] == "tag:LT-002":
                extras = [t.upper() for t in lf.get("extra_names", [])]
                assert "LEVEL GAUGE" not in extras
                assert "LEVEL TRANSMITTER" not in extras


def test_drain_line_is_connected_from_the_vessel_to_the_main_run():
    """The vessel drain boundary is contiguous on source geometry only."""
    emitted = _emitted_source_entities()
    assert "/CP2-V-71101/D1" in emitted
    for seg in range(2, 8):
        assert "/D711001%03d" % seg in emitted
    branch = [e for e in emitted if "D-711001-P1B-01" in e]
    assert any("VALVE" in e for e in branch)
    branch2 = [e for e in emitted if "D-711001-P1B-02" in e]
    assert any("TEE" in e for e in branch2)


def test_absent_drain_segments_are_recorded_not_fabricated():
    """/D711001001 has no source entity, so it must not appear."""
    emitted = _emitted_source_entities()
    assert "/D711001001" not in emitted
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    assert "D711001001" in text


def test_scaffolding_is_complete_and_nothing_was_dropped():
    """The permanent scaffold must be fully carried, not partially pruned.

    One exception is now declared and measured: the four handrail sections that
    physically intersect the M1B manhole's modelled-open cover. This test still
    fails on any OTHER scaffold loss, so it keeps guarding against a partial
    prune reappearing.
    """
    emitted = _emitted_source_entities()
    scaffold = [e for e in emitted if "PRIMARY-P1-P1B-TS01" in e]
    assert len(scaffold) >= 2100

    fence = "OF FRMWORK /PRIMARY-P1-P1B-TS01-HANDRAILS"
    unexpected = []
    for r in _inventory()["excluded"]:
        name = str(r.get("source_entity_name", "")).upper()
        if "PRIMARY-P1-P1B-TS01" not in name:
            continue
        if fence in name and any(f"OF SCTN {s} {fence}" in name
                                 for s in CLASHING_HANDRAIL_SECTIONS):
            continue  # the declared clashing-run exception
        unexpected.append(name)
    assert not unexpected, (
        f"{len(unexpected)} scaffold entities were dropped without a declared "
        f"rule, e.g. {unexpected[:3]}")


# --- Canonical build input integrity ---

def test_canonical_build_input_and_manifest_agree():
    """The canonical input must be the validated artifact the manifest records."""
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"canonical build input not present at {BUILD_INPUT_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out = manifest["output"]
    assert out["path"] == "geometry/build_input.ifc"
    assert out["entity_count"] == 819281
    assert out["building_element_proxy_count"] == 25815
    assert out["entity_id_sha256"] == (
        "4ab7ab132e8364b4e3d8c311e99a58e5345c4f46dad9d6f203caecc915187f4a")

    size = os.path.getsize(BUILD_INPUT_PATH)
    assert size == out["bytes"] == 140506168
    assert manifest["mode"] == "passthrough"
    assert "passthrough_proven_exact" in manifest["reproduction"]



def test_canonical_build_input_bytes_match_recorded_hash():
    """Hash the build input so silent replacement is detected."""
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"canonical build input not present at {BUILD_INPUT_PATH}")
    h = hashlib.sha256()
    with open(BUILD_INPUT_PATH, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    assert h.hexdigest() == (
        "bd4fe9732bbd467f1bc64ed890a031b67b3e085a061cf56f90a0fd96d770f897"), (
        "geometry/build_input.ifc no longer matches the validated SEP export; "
        "see geometry/README.md before replacing it")


# The P&ID names four vessel internals. The authoritative IFC contains none of
# them, so they are ABSENT BY EVIDENCE rather than deferred pending placement.
# These tokens are the P&ID spellings; the source is scanned for them directly
# so that a future re-export carrying internals fails this test loudly instead
# of leaving the model quietly missing equipment the drawing requires.
PID_NAMED_INTERNALS = (
    "SCHOEPENTOETER",
    "VANEPACK",
    "VANE PACK",
    "BAFFLE",
    "DEMISTER",
    "MELLACHEVRON",
    "KNITMESH",
    "MESH PAD",
    "VORTEX BREAKER",
)

# IFCOLEFACES / non-geometry properties legitimately carry descriptive text
# such as 'LIQUID LEVEL GAUGE/TRANSMITTER (CONTROL)'. Only geometry-bearing
# entity types count as evidence that an element exists.
GEOMETRY_ENTITY_TYPES = (
    r"IFCBUILDINGELEMENTPROXY",
    r"IFCBUILDINGELEMENT",
    r"IFCPIPESEGMENT",
    r"IFCPIPEFITTING",
    r"IFCPIPEFLOWFITTING",
    r"IFCEQUIPMENT",
    r"IFCFLOWSEGMENT",
    r"IFCFLOWFITTING",
    r"IFCFLOWTERMINAL",
    r"IFCDISTRIBUTIONELEMENT",
    r"IFCDISTRIBUTIONFLOWELEMENT",
    r"IFCDISCRETEACCESSORY",
    r"IFCMECHANICALFASTENER",
    r"IFCMECHANICALFASTENERENCLOSURE",
    r"IFCPLATE",
    r"IFCMEMBER",
    r"IFCCOLUMN",
    r"IFCBEAM",
    r"IFCSLAB",
    r"IFCRAILING",
    r"IFCSTAIR",
    r"IFCSTAIRFLIGHT",
    r"IFCTRANSFORMER",
    r"IFCSANITARYTERMINAL",
    r"IFCLIGHTFIXTURE",
    r"IFCSYSTEMFURNITUREELEMENT",
    r"IFCFURNISHINGELEMENT",
    r"IFCGRATE",
    r"IFCREINFORCINGBAR",
    r"IFCREINFORCINGMESH",
    r"IFCTENDON",
    r"IFCTENDONCONDUIT",
    r"IFCVIBRATIONDAMPER",
    r"IFCMECHANICALEQUIPMENT",
)

_GEOM_TYPE_RE = re.compile(r"=(" + "|".join(GEOMETRY_ENTITY_TYPES) + r")\(")


def _split_args(arglist):
    """Split a STEP argument list on top-level commas only.

    Arguments may be references, numbers, $, or single-quoted strings that
    themselves contain commas, parentheses, and escaped quotes. Anything
    naive splits on those and corrupts the result.
    """
    args, buf, depth, in_str, i = [], [], 0, False, 0
    n = len(arglist)
    while i < n:
        c = arglist[i]
        if in_str:
            if c == "'":
                if i + 1 < n and arglist[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_str = False
            buf.append(c)
        else:
            if c == "'":
                in_str = True
                buf.append(c)
            elif c == "(":
                depth += 1
                buf.append(c)
            elif c == ")":
                depth -= 1
                buf.append(c)
            elif c == "," and depth == 0:
                args.append("".join(buf).strip())
                buf = []
            else:
                buf.append(c)
        i += 1
    if buf:
        args.append("".join(buf).strip())
    return args


def _ifc_geometry_entity_strings(blob):
    """Every string literal belonging to a geometry-bearing IFC entity.

    Deliberately position-agnostic. IFC entity signatures put the element name
    in different argument slots per type, and guessing the slot silently
    inspects the wrong field. Any string on a geometry entity counts, so an
    internal that appears in a name, description, ObjectType, or Tag is caught
    regardless of type.
    """
    found = []
    for m in _GEOM_TYPE_RE.finditer(blob):
        start = m.end()
        depth, in_str, i, n = 1, False, start, len(blob)
        while i < n and depth:
            c = blob[i]
            if in_str:
                if c == "'":
                    if i + 1 < n and blob[i + 1] == "'":
                        i += 2
                        continue
                    in_str = False
            elif c == "'":
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth:
            continue
        for a in _split_args(blob[start:i - 1]):
            if len(a) >= 2 and a[0] == a[-1] == "'":
                found.append((m.group(1), a[1:-1].replace("''", "'")))
    return found


def test_pid_named_internals_are_absent_from_the_authoritative_source():
    """The internals the P&ID requires must be proven absent, not assumed.

    If a future IFC export ever carries them, this fails and the internals
    become placeable geometry instead of a recorded limitation. That is the
    desired direction of failure: it turns an omission into a TODO.
    """
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"canonical build input not present at {BUILD_INPUT_PATH}")
    with open(BUILD_INPUT_PATH, "r", encoding="utf-8", errors="replace") as f:
        blob = f.read()
    entities = _ifc_geometry_entity_strings(blob)
    assert entities, (
        "no geometry-bearing entities parsed from the canonical IFC; the "
        "absence check below would silently pass on an unreadable file")
    present = sorted({f"{t}:{s}" for t, s in entities
                      for tok in PID_NAMED_INTERNALS
                      if tok in s.upper()})
    assert not present, (
        f"the source now carries P&ID-named internals {present}; they are no "
        f"longer an evidence-backed omission and must be placed as real "
        f"geometry, with known_limitations updated")


def test_no_separator_internals_were_fabricated_in_the_model():
    """Nothing in the emitted model may claim to be an internal the source lacks.

    A tripwire against inventing geometry. This does NOT forbid modelling the
    internals; it forbids them appearing without source provenance, which is
    what a hand-built mesh would look like.
    """
    h = _hierarchy()
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_text = f.read()

    # Inspect only identity-bearing fields. The hierarchy also embeds
    # battery_limit_evidence prose that legitimately NAMES the internals while
    # documenting their absence, so a whole-document substring scan would flag
    # the very record that proves the omission. The question is never "does the
    # word appear" but "does any emitted element CLAIM to be an internal".
    identity = []
    for lf in h.get("leaves", []):
        identity.append(str(lf.get("name", "")))
        identity.append(str(lf.get("gltf_node", "")))
        identity.append(str(lf.get("tag", "")))
        identity.extend(str(s) for s in lf.get("source_entities", []))
    emitted = "\n".join(identity).upper()
    assert emitted, "no leaf identity fields found in the hierarchy"
    for tok in PID_NAMED_INTERNALS:
        assert tok not in emitted, (
            f"an emitted element claims to be {tok!r} but the authoritative "
            f"IFC contains no such geometry; that is invented geometry. "
            f"Record it as a parametric, non-source element explicitly, or "
            f"remove it.")

    # The omission must stay on the record rather than being quietly dropped.
    assert "SCHOEPENTOETER" in manifest_text.upper(), (
        "the internals omission was deleted from selection_manifest.json; "
        "a known P&ID requirement must not disappear from the record")
    assert "known_limitations" in manifest_text, (
        "known_limitations is missing from the selection manifest")


# ---------------------------------------------------------------------------
# Identity-keyed exclusions (Navisworks SelectionID)
#
# A name token is a claim about a CLASS of names; a SelectionID is one specific
# element the engineer looked at. The SelectionID survives the IFC export even
# though the Navisworks assembly tree does not, because BIMCamel writes it
# verbatim into IfcRoot.GlobalId - verified 2026-09-28 on all 28,282 proxies in
# geometry/build_input.ifc against the 926,472-row BIMCamel manifest.
# ---------------------------------------------------------------------------

BCMANIFEST_PATH = os.path.join(
    REPO, "docs", "Mj-Real-Data", "navisworks", "New_model",
    "1st_Sperator_area.ifc.bcmanifest")

# Whole-field manifest retained for the PMJP positive control only: it carries
# the /PMJP1F and /MJP1A folders that the scoped SEP manifest cannot.
WHOLE_FIELD_MANIFEST_PATH = os.path.join(
    REPO, "docs", "Mj-Real-Data", "navisworks",
    "MGP1-CP2-PIL-MP-7180-0004_007",
    "19-01-2021_Majnoon_NAVIS_IFC4.ifczip.bcmanifest")

# The four measured M1B open-cover clashes, by SelectionID (SEP-export ids).
CLASHING_HANDRAIL_SELECTION_IDS = (
    "1JZHvsBIfXyUhMtIM41CfS",  # SCTN 55
    "33s8$ajl3Zy0Kz3FQhcJgN",  # SCTN 269
    "0spQvq4P92cCOlFhWzws1L",  # SCTN 275
    "25gL9Ga3DgerSDF0nKM0xa",  # SCTN 276
)

# The manholes, which must survive every exclusion rule. Re-derived from the SEP
# export: each manhole is a single instance, so there are exactly two protected
# ids (the old whole-field export carried two rows per manhole).
PROTECTED_SELECTION_IDS = (
    "06cIk4wBJeoRestptU6DD4",  # /CP2-V-71101/M1A
    "13k3cE7DcN811MxFqdiUBu",  # /CP2-V-71101/M1B
)


def _selection():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from geometry_selection import SelectionManifest
    return SelectionManifest(SELECTION_MANIFEST_PATH)


def _global_id_multiplicity(ifc_path):
    """Count IFCBUILDINGELEMENTPROXY instances per GlobalId in an IFC file.

    GlobalId is ATTRIBUTE 1 of IfcBuildingElementProxy, positional - the literal
    string "GlobalId" appears zero times in this export, so a named-attribute
    regex finds nothing. Verified against a real entity:
      #40=IFCBUILDINGELEMENTPROXY('2Ga3NUEbgwMJlnsJhYh2yZ',#13,'/EF-...')
    """
    import collections
    rx = re.compile(r"=IFCBUILDINGELEMENTPROXY\('([^']*)'")
    counts = collections.Counter()
    with open(ifc_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                counts[m.group(1)] += 1
    return counts


def test_identity_exclusions_target_exactly_one_instance_each():
    """A SelectionID is not necessarily one element - guard the blast radius.

    Measured 2026-09-28 on the SEP export: 25,815 IFCBUILDINGELEMENTPROXY
    instances carry only 23,478 *distinct* GlobalIds, and 671 of those
    SelectionIDs are instanced more than once, up to 50 times. The BIMCamel
    manifest has exactly one row per distinct SelectionID (23,478 rows, zero
    duplicates), so the manifest alone cannot reveal this.

    An identity exclusion therefore removes *every* instance of that
    SelectionID. The four declared handrail clashes are each instanced once, so
    they behave exactly as intended - but a future exclusion authored against a
    shared assembly SelectionID would silently delete up to 50 proxies, and
    the name token would never have warned anyone. This test fails loudly
    instead.
    """
    m = _selection()
    declared = sorted(
        sid for sid in m.identity_exclusions
        if not str(sid).startswith("__")
    )
    if not declared:
        return
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"build input not present at {BUILD_INPUT_PATH}")

    mult = _global_id_multiplicity(BUILD_INPUT_PATH)
    fat = {sid: mult.get(sid, 0) for sid in declared if mult.get(sid, 0) > 1}
    assert not fat, (
        "identity exclusion would remove more than one instance: "
        f"{fat}. Re-derive the exact SelectionID, or accept the blast radius "
        "explicitly - a shared SelectionID is how one exclusion deletes many "
        "elements by accident.")

    absent = [sid for sid in declared if mult.get(sid, 0) == 0]
    assert not absent, (
        f"declared identity exclusions absent from the build input: {absent}")


def test_identity_exclusions_are_declared_for_exactly_the_four_clashes():
    """The identity list mirrors the measured handrail removal, nothing else."""
    m = _selection()
    assert set(m.identity_exclusions) == set(CLASHING_HANDRAIL_SELECTION_IDS), (
        "identity_exclusions drifted from the four measured M1B clashes: "
        f"{sorted(m.identity_exclusions)}")

    for sid in CLASHING_HANDRAIL_SELECTION_IDS:
        entry = m.identity_exclusions[sid]
        assert entry["reason"], f"{sid} has no declared reason"
        assert entry["evidence"], f"{sid} has no evidence pointer"


def test_identity_exclusion_removes_the_element_and_nothing_else():
    """An identity fires on that element only, regardless of its name."""
    m = _selection()
    sid = CLASHING_HANDRAIL_SELECTION_IDS[0]

    # The declared element is removed, and the rule that fired is named.
    excluded, kind, detail = m.exclusion_rule(
        "POHEDRON 1 OF TMPLATE 1 OF FITTING 1 OF SCTN 55 OF FRMWORK "
        "/PRIMARY-P1-P1B-TS01-HANDRAILS", "Group", sid)
    assert excluded is True
    assert kind == "identity"
    assert detail == sid

    # The same NAME on a different element is NOT removed by the identity rule.
    # This is the property a name token cannot provide: two elements sharing a
    # name are distinguished because only one of them was ever declared.
    still_excluded, kind, _ = m.exclusion_rule(
        "POHEDRON 1 OF TMPLATE 1 OF FITTING 1 OF SCTN 55 OF FRMWORK "
        "/PRIMARY-P1-P1B-TS01-HANDRAILS", "Group", "aDifferentSelectionId")
    assert kind == "name", (
        "with an undeclared id the decision must fall back to the name token, "
        f"got {kind!r}")
    assert still_excluded is True  # the legacy name token still applies here

    # And an unrelated element with an unrelated name is untouched.
    excluded, kind, _ = m.exclusion_rule(
        "BOX 1 OF SUBEQUIPMENT /CP2-P-71102A/NOZZLES", "Group", "aDifferentSelectionId")
    assert excluded is False
    assert kind is None


def test_identity_matching_is_exact_and_never_extends_a_base():
    """A declared id must never capture a different, longer SelectionID.

    An earlier version of the rule accepted `base + '$' + digits`, on the belief
    that BIMCamel appends a `$<n>` when a SelectionID repeats. That belief was
    wrong. Measured 2026-09-28 against the SEP export, all 25,815
    `IfcBuildingElementProxy` instances carry a `GlobalId` of exactly 22
    characters with no exceptions, and the 23,478-row manifest holds 23,478
    distinct 22-character SelectionIDs with no duplicates - so there is no
    disambiguated-suffix form to accommodate.

    Worse, `$` is a legal character in a Navisworks InstanceGuid: 6,257 of the
    23,478 distinct manifest SelectionIDs contain one, and real source proxies
    end in `$` followed by digits. Those are real, unrelated elements, so a
    prefix rule would exclude the wrong thing.
    """
    m = _selection()
    sid = CLASHING_HANDRAIL_SELECTION_IDS[0]
    for suffix in ("$2", "$20", "$100"):
        longer = sid + suffix
        excluded, kind, detail = m.exclusion_rule("ANY NAME AT ALL", "Group", longer)
        assert excluded is False, (
            f"{longer!r} is a different SelectionID that merely shares a prefix "
            f"with the declared {sid!r}; it must not inherit its rule")
        assert kind != "identity"
        assert detail != sid


def test_a_real_id_ending_in_dollar_digits_resolves_to_itself():
    """SelectionIDs ending in `$digits` are real, first-class ids.

    Measured on the SEP manifest: ids whose tail looks like a numeric suffix are
    real (60 of 23,478 end in `$`+digits). Under the old prefix rule they
    could have been reached by stripping `$digits` to hit a shorter declared id,
    silently applying a rule to the wrong element. Exact matching removes the
    hazard: a rule resolves only to the full id.

    None of the six declared ids happens to end in `$digits`, so this test uses
    a real manifest id instead and proves the generic property on it.
    """
    if not os.path.exists(BCMANIFEST_PATH):
        pytest.skip(f"BIMCamel manifest not present at {BCMANIFEST_PATH}")
    m = _selection()
    with open(BCMANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            sid = line.split("\t", 1)[0]
            if len(sid) == 22 and sid[-1].isdigit() and sid[-2] == "$":
                break
        else:
            pytest.skip(
                "no real `$digits`-terminated SelectionID found in manifest")
    assert len(sid) == 22
    assert sid[-2] == "$" and sid[-1].isdigit()

    # The full id is treated as itself: no declared base, no protected entry,
    # and crucially no truncated prefix inheritance.
    excluded, kind, detail = m.exclusion_rule("/ANY/NAME", "Group", sid)
    assert detail in (None, "")
    assert kind in (None, "")
    assert excluded is False

    # The truncated form is a different id and must match nothing either. The
    # point is that neither form is silently rewritten into the other.
    truncated = sid[:-2]
    excluded, kind, _ = m.exclusion_rule("/ANY/NAME", "Group", truncated)
    assert excluded is False
    assert kind != "identity"



def test_protected_identities_survive_every_exclusion_rule():
    """A protected manhole is never removed, and is still classified normally."""
    m = _selection()

    for sid in PROTECTED_SELECTION_IDS:
        excluded, kind, detail = m.exclusion_rule("/CP2-V-71101/M1A", "Group", sid)
        assert excluded is False, f"{sid} must not be excluded"
        assert kind == "protected"
        assert detail == sid
        assert sid in m.protected_selection_ids

    # Protection must not change WHERE the element goes. Blocking classification
    # instead of blocking exclusion would drop proven-real hardware, which is
    # the exact M1A/M1B regression this guard exists to prevent.
    with_id = m.classify("/CP2-V-71101/M1A", "Group", None, None, None,
                         PROTECTED_SELECTION_IDS[0])
    without_id = m.classify("/CP2-V-71101/M1A", "Group")
    assert with_id == without_id, (
        f"protection altered classification: {with_id} != {without_id}")
    assert with_id[1] == "vessel_nozzles_saddles", (
        f"the manhole must still be modelled, got {with_id}")


def test_protection_overrides_a_name_token_that_would_exclude():
    """A protected identity beats a name token, so ACCESS cannot reach it."""
    m = _selection()
    # An ACCESS name is excluded for an ordinary element ...
    excluded, kind, _ = m.exclusion_rule("ACCESS ENVELOPE BOX 1", "Group",
                                         "anUndeclaredSelectionId")
    assert excluded is True and kind == "name"
    # ... but not for a protected one, because the exclusion is a mistake here.
    excluded, kind, _ = m.exclusion_rule("ACCESS ENVELOPE BOX 1", "Group",
                                         PROTECTED_SELECTION_IDS[0])
    assert excluded is False and kind == "protected"


def test_an_id_cannot_be_both_excluded_and_protected():
    """The manifest must not be able to encode a contradiction."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from geometry_selection import SelectionManifest
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["protected_selection_ids"]["entries"].append(
        {"selection_id": CLASHING_HANDRAIL_SELECTION_IDS[0], "reason": "conflict"})
    tmp = os.path.join(REPO, "geometry", ".tmp_conflict_manifest.json")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with pytest.raises(ValueError):
            SelectionManifest(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_every_removed_row_records_which_rule_fired():
    """An exclusion must be auditable without re-deriving the manifest.

    The inventory previously used leaf_reason() for removed rows, which produced
    the classification-oriented "no manifest selector matches this source
    identity" for all 58 removals - a sentence that describes an unclassified
    element, not a deliberately excluded one.
    """
    inv = _inventory()
    by_basis = inv["accounting"].get("excluded_by_basis")
    assert by_basis is not None, "accounting must break removals down by rule basis"
    assert by_basis["identity"] + by_basis["name"] == len(inv["excluded"])
    assert by_basis["identity"] == CLASHING_HANDRAIL_ELEMENT_COUNT, (
        "the four clashes are keyed by identity, so identity removals must be 4, "
        f"got {by_basis['identity']}")
    for row in inv["excluded"]:
        assert row.get("exclusion_basis") in ("identity", "name"), row
        assert row.get("exclusion_rule"), row
        assert row.get("reason"), row


def test_measured_handrail_counts_are_the_ones_recorded():
    """Lock the recount that corrected the 686/496 error in the documentation."""
    emitted = {s.upper() for s in _emitted_source_entities()}
    handrail = {s for s in emitted if "HANDRAIL" in s}
    assert len(handrail) == EMITTED_HANDRAIL_SUBITEMS_TOTAL, (
        f"expected {EMITTED_HANDRAIL_SUBITEMS_TOTAL} emitted handrail sub-items, "
        f"got {len(handrail)}")
    primary = {s for s in handrail
               if "OF FRMWORK /PRIMARY-P1-P1B-TS01-HANDRAILS" in s}
    platform = {s for s in handrail
                if "OF FRMWORK /PLATFORM-P1-P1B-EM01-HANDRAILS" in s}
    assert len(primary) == EMITTED_HANDRAIL_SUBITEMS_PRIMARY_P1B_TS01
    assert len(platform) == EMITTED_HANDRAIL_SUBITEMS_PLATFORM_P1B_EM01

    # 686 is the all-assembly total, NOT the P1B count, and the platform count
    # is 111, not the 496 an earlier pass claimed. Assert the distinction so the
    # two cannot be silently swapped back.
    assert EMITTED_HANDRAIL_SUBITEMS_TOTAL > \
        EMITTED_HANDRAIL_SUBITEMS_PRIMARY_P1B_TS01
    assert EMITTED_HANDRAIL_SUBITEMS_PLATFORM_P1B_EM01 != 496


def test_pmjp_nodes_are_measured_absent_while_crane_folders_are_present():
    """No speculative PMJP removal, and the reason is measured, not inferred.

    Measured 2026-09-28 across the whole-field 926,472-row BIMCamel manifest:
    `/PMJP1A`, `/PMJP1B` and `/PMJP1C` are each 0 occurrences, while the crane
    folders `/PMJP1F` (406) and `/MJP1A` (62) are present. The positive controls
    matter as much as the zeros - they prove the probe can see a `/PMJP1*`
    folder when one is actually exported, so the three zeros are absence, not
    blindness.

    The SEP-export reset narrows the shipped manifest (23,478 rows) to the
    first-separator scope, so BOTH the zeros and the positive controls come from
    the retained whole-field manifest (WHOLE_FIELD_MANIFEST_PATH), which still
    exists alongside the scoped build input. Reading the whole-field manifest
    keeps the positive control intact, which the scoped manifest cannot provide
    (every /PMJP1* and crane token is 0 there by scope).

    An earlier version of this test argued from 0 of 26,744 proxies being
    translucent that the node must be a Navisworks *Layer*. That argument is
    withdrawn: if the geometry was never exported, nothing translucent appears
    either way, so the measurement showed absence and proved nothing about
    layers. The engineer has since confirmed all three are tree nodes under
    `Majnoon_NAVIS_OBS.rvm`.

    If a future re-export appends OBS in scope these counts move, the test
    fails, and that failure is the signal to obtain the membership as exact
    SelectionIDs rather than to widen a name token.
    """
    m = _selection()
    assert not [s for s in m.identity_exclusions if "PMJP" in s.upper()], (
        "a PMJP identity exclusion was added without the engineer confirming "
        "that node's membership; unknown membership must not be guessed")

    if not os.path.exists(WHOLE_FIELD_MANIFEST_PATH):
        pytest.skip(
            "whole-field BIMCamel manifest not present at "
            f"{WHOLE_FIELD_MANIFEST_PATH}")

    counts = {"PMJP1A": 0, "PMJP1B": 0, "PMJP1C": 0, "PMJP1F": 0, "MJP1A": 0}
    with open(WHOLE_FIELD_MANIFEST_PATH, "r", encoding="utf-8",
              errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[2]
            for token in counts:
                if token in name:
                    counts[token] += 1

    for node in ("PMJP1A", "PMJP1B", "PMJP1C"):
        assert counts[node] == 0, (
            f"{node} is now present in the export ({counts[node]} rows). The "
            "export scope changed - obtain the node membership from a scoped "
            "BIMCamel run and add exact SelectionIDs, do not widen a name token")
    assert counts["PMJP1F"] > 0, (
        "positive control failed: /PMJP1F is 0, so the probe cannot see a "
        f"/PMJP1* folder and the three zeros prove nothing. got {counts}")


def test_every_declared_selection_id_exists_in_the_shipped_manifest():
    """A declared id must be a real Navisworks SelectionID, not a transcription.

    This is the check that stops a hand-typed id from silently never matching.
    """
    if not os.path.exists(BCMANIFEST_PATH):
        pytest.skip(f"BIMCamel manifest not present at {BCMANIFEST_PATH}")
    m = _selection()
    wanted = set(m.identity_exclusions) | set(m.protected_selection_ids)
    found = set()
    with open(BCMANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0] in wanted:
                found.add(parts[0])
    missing = wanted - found
    assert not missing, (
        "declared selection_ids that are not in the BIMCamel manifest and can "
        f"therefore never match anything: {sorted(missing)}")


def test_every_selection_id_is_exactly_22_characters_so_exact_matching_is_sound():
    """Pin the invariant that lets identity matching be exact rather than prefix.

    Measured 2026-09-28 on the SEP export: all 23,478 distinct rows in the
    23,482-line companion manifest hold a 22-character SelectionID - the
    `IfcGloballyUniqueId` limit - and all 25,815 `IfcBuildingElementProxy`
    GlobalIds in geometry/build_input.ifc are exactly 22 characters, with no
    exceptions.

    Uniform width is what makes exact matching provably complete: one id can
    never be a strict `$digits` extension of another, so there is no
    disambiguated-suffix form that exact equality would miss. If a future export
    ever produced a longer GlobalId, this test fails and the matching rule has to
    be revisited deliberately instead of by accident.
    """
    if not os.path.exists(BCMANIFEST_PATH):
        pytest.skip(f"BIMCamel manifest not present at {BCMANIFEST_PATH}")
    m = _selection()
    declared = sorted(set(m.identity_exclusions) | set(m.protected_selection_ids))
    bad = [(s, len(s)) for s in declared if len(s) != 22]
    assert not bad, (
        f"declared SelectionIDs are not 22 characters, so the uniform-width "
        f"invariant behind exact matching no longer holds: {bad}")

    widths = {}
    with open(BCMANIFEST_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            sid = line.split("\t", 1)[0]
            if not sid or " " in sid[:1]:
                continue
            widths[len(sid)] = widths.get(len(sid), 0) + 1
    total = sum(widths.values())
    # The four non-22 lines are the manifest's own metadata header.
    assert widths.get(22, 0) >= 23000, (
        f"expected ~23,478 22-character SelectionIDs, got {widths} ({total} lines)")
    assert set(widths) - {22} <= {12, 15, 20, 25}, (
        f"unexpected SelectionID widths appeared: {sorted(widths)}")


# --- Starved leaves: the guard for the bug class that hid the inlet valve station.
#
# A declared leaf whose selectors match nothing is invisible in every number the
# build reports. UZV-051, UZV-052 and RO-051 sat at tx 24-28 while the crop ended
# at 21.5; their selectors were correct the whole time, and the build still passed
# every population and QA check. Worse, the loss was then recorded as a *fact*
# ("no geometry at all in the source") in known_limitations and the tag registry,
# so a measured absence was believed over the model. The only symptom was two
# numbers in this file.

# Each entry must name a leaf that is genuinely empty, and say which of the two
# distinct causes applies. An empty leaf not on this list fails the test below.
STARVED_LEAF_ALLOWLIST = {
    "tag:FT-002": (
        "genuinely absent from source: zero proxies carry the name in the SEP "
        "export, and no export ever contained it. Not a defect."),
    "tag:SP-CP02": (
        "pre-existing registry demotion: the geometry IS emitted, as "
        "unregistered_tag:SP-CP02, because SP-CP02 has no tag_registry entry. "
        "Unrelated to the inlet station; tracked, not fixed here."),
}

REGISTRY_PATH = os.path.join(REPO, "config", "tag_registry.json")


def _registry_entry(reg, tag):
    """Find a tag in any registry section. Tags are unique across sections."""
    for section in ("equipment", "nozzles", "valves", "instruments", "controllers"):
        if tag in reg.get(section, {}):
            return reg[section][tag]
    return None


def test_no_declared_leaf_is_starved():
    """Every selector declared in the manifest must match real source geometry.

    The crop used to be a hardcoded literal in build_twin_pipeline.py while the
    manifest carried a decorative copy of the same bounds. Editing the manifest
    changed nothing at all, so the only way this class of loss is caught is by
    asserting declared-vs-emitted, which nothing else in the suite does.
    """
    m = _selection()
    emitted = {l["name"] for l in _hierarchy()["leaves"]}
    declared = [(g["id"], lf["name"])
                for g in m.raw["groups"] for lf in g["leaves"]]
    assert declared, "manifest declares no leaves at all; the guard would be vacuous"

    starved = {nm for _, nm in declared if nm not in emitted}
    assert starved == set(STARVED_LEAF_ALLOWLIST), (
        f"leaves declared in the manifest but absent from the model: "
        f"{sorted(starved - set(STARVED_LEAF_ALLOWLIST))}. If an item here is "
        f"real hardware, its geometry is being dropped before classification - "
        f"check the spatial_crop in the manifest FIRST, and only then suspect "
        f"the selectors. If it genuinely has no source geometry, add it to "
        f"STARVED_LEAF_ALLOWLIST with the reason. Stale allowlist entries: "
        f"{sorted(set(STARVED_LEAF_ALLOWLIST) - starved)}")

    for name, reason in STARVED_LEAF_ALLOWLIST.items():
        assert reason and reason.endswith("."), (
            f"allowlisted starved leaf {name} must carry a stated reason")


def test_the_inlet_valve_station_is_complete_and_addressable():
    """The four bodies the old crop cut off must be placed and individually addressable.

    Placement is not enough: each must also be named in the tag registry, or the
    2D/3D binding has no target and the tag silently cannot be driven.
    """
    h = _hierarchy()
    leaves = {l["name"]: l for l in h["leaves"]}
    # (leaf, minimum element count, minimum triangles) measured from the source.
    expected = {
        "tag:UZV-051": (1, 1000),   # 30" inlet trunk ESD, 1332 tris
        "tag:UZV-052": (1, 1000),   # 2" equalization/bypass, 1468 tris
        "tag:UZV-053": (1, 1000),   # 30" HIPPS partner to UZV-054, 1332 tris
        "tag:RO-051":  (1, 100),    # restriction orifice paired with UZV-052, 152
        "tag:PDZT-051": (2, 1000),  # permissive to open UZV-051/052, 1392
    }
    for name, (min_el, min_tri) in expected.items():
        assert name in leaves, f"{name} is not an emitted node at all"
        lf = leaves[name]
        assert lf.get("elementCount", 0) >= min_el, (
            f"{name} emitted {lf.get('elementCount')} elements, expected >= {min_el}")
        assert lf.get("triCount", 0) >= min_tri, (
            f"{name} emitted {lf.get('triCount')} triangles, expected >= {min_tri}")
        assert lf.get("geometry_state") == STATE_CORE, (
            f"{name} is marked {lf.get('geometry_state')!r}, not CORE_CONFIRMED")

    # UZV-053 must have left the unclassified catch-all to become a real tag.
    assert "tag:UZV-053" in leaves, "UZV-053 regressed into the review bucket"

    reg = json.load(open(REGISTRY_PATH, "r", encoding="utf-8"))
    for name in expected:
        tag = name.split(":", 1)[1]
        entry = _registry_entry(reg, tag)
        assert entry, f"{tag} has no tag_registry entry, so it cannot be bound 2D->3D"
        assert entry.get("gltf_ref") == name, (
            f"{tag} gltf_ref is {entry.get('gltf_ref')!r}, expected {name!r}")
        assert entry.get("addressable_3d_node") == "verified", (
            f"{tag} is not addressable: {entry.get('addressable_3d_node')!r}")


def test_the_pipeline_reads_the_crop_instead_of_restating_it():
    """The crop must have exactly one home: the manifest.

    build_twin_pipeline.py used to hardcode `-19.5 <= tx <= 21.5` while the
    manifest documented the same bounds. That duplication is what made the
    manifest's crop note decorative - a manifest-only edit rebuilt identically.
    """
    with open(BUILDER_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    assert "spatial_crop" in src, (
        "the builder must read spatial_crop from the manifest, but the token "
        "does not appear in it")
    # No bare crop bounds restated in the builder. These literals are the old
    # values; matching either means someone reintroduced a second home.
    for stale in ("-19.5 <= tx", "tx <= 21.5", "21.5 and", "-7.5 <= tz <= 14.5"):
        assert stale not in src, (
            f"build_twin_pipeline.py restates the crop bound {stale!r}; the "
            f"manifest is the single source of truth for spatial_crop")

# --- N1 tie-in pipe body: recovered by exact SelectionID, guarded against drift ---

N1_TIEIN_LEAF = "service_piping:02_INLET_N1_TIEIN"
N1_TIEIN_SELECTION_IDS = ("0gxq48Inht$0OKRcZRJ4k9", "1D6IkjrOTadX2TzxztOoyT")
# The engineer identified 1D6IkjrOTadX2TzxztOoyT in the live Navisworks model as
# the pipe connected to the vessel. Measured: ty=1.127, tz=0.000 (coaxial with
# nozzle N1), tx spanning -16.196..-8.579, coincident to 0.000 m with
# /P711101004, P-711101 its only branch line within 2.5 m, and its far end meets
# FLANGE 1 of /P-711101-P1B-01/B1 at tx=-8.579, 2.03 m short of N1 at tx=-7.550.
N1_TIEIN_COORD_M = (-16.196, -8.579)


def test_the_n1_tiein_leaf_is_not_silently_empty():
    """SelectionIDs are export-scoped, so a re-export must fail loudly.

    A re-export mints new SelectionIDs for the same elements, which would leave
    this leaf empty and quietly drop the inlet pipe from the model. A non-zero
    element count is asserted so that drift is a test failure, not a surprise.
    """
    h = _hierarchy()
    leaf = next((l for l in h["leaves"] if l["name"] == N1_TIEIN_LEAF), None)
    assert leaf is not None, f"{N1_TIEIN_LEAF} is missing from the hierarchy"
    assert leaf.get("elementCount", 0) > 0, (
        f"{N1_TIEIN_LEAF} emitted no geometry. Its two source proxies are "
        f"unnamed ('Cylinder' in the Navisworks manifest) and are selected by "
        f"exact SelectionID, which is scoped to the export run. Re-derive the "
        f"ids from this export's .bcmanifest and update the manifest."
    )
    assert leaf.get("geometry_state") == "CORE_CONFIRMED"


def test_the_n1_tiein_pipe_body_sits_on_the_n1_axis_between_flange_and_nozzle(glb):
    """The recovered pipe must physically abut the vessel tie-in it claims."""
    lo, hi, dim, _ = _subtree_aabb(glb, N1_TIEIN_LEAF)
    assert lo[0] <= N1_TIEIN_COORD_M[0] + 0.01
    assert hi[0] >= N1_TIEIN_COORD_M[1] - 0.01
    # Coaxial with nozzle N1 at ty=1.127, tz=0.000, so the run centre must be
    # near that axis rather than at the P-711100 elevation (tz ~11.8).
    assert abs(dim[1]) < 2.0 and abs(dim[2]) < 2.0, (
        f"tie-in run is {dim} m, not a nozzle-axis pipe stub")


def test_selection_id_selectors_are_exact_and_never_pattern_based():
    """A declared SelectionID must match only itself, never a longer id.

    '$' is a legal Navisworks InstanceGuid character: 6,257 of the 23,478 SEP
    manifest ids contain one and 60 already end in '$' plus digits, so any
    prefix/suffix extension would capture unrelated real elements. That exactness
    is already asserted for exclusions; this pins it for the positive selector.
    """
    m = _selection()
    for sid in N1_TIEIN_SELECTION_IDS:
        gid, leaf, _ = m.classify(None, "Cylinder", -16.5, 1.127, 0.0, sid)
        assert leaf == N1_TIEIN_LEAF, f"{sid} must select the tie-in leaf"

    # An extended id must NOT match.
    for ext in (sid + "$1" for sid in N1_TIEIN_SELECTION_IDS):
        _, leaf, _ = m.classify(None, "Cylinder", -16.5, 1.127, 0.0, ext)
        assert leaf != N1_TIEIN_LEAF, f"{ext} must not match the tie-in leaf"

    # An id that is in no leaf must claim nothing at all.
    _, leaf, _ = m.classify(None, "Cylinder", -16.5, 1.127, 0.0,
                            "AAAAAAAAAAAAAAAAAAAAAA")
    assert leaf is None


def test_the_n1_tiein_selector_cannot_steal_a_named_neighbour():
    """Exact identity must not pull a named element out of its own leaf.

    The tie-in corridor also contains FLANGE 1 of /P-711101-P1B-01/B1,
    /P711101004, PCOMPONENT 3 and /CP2-V-71101/N1. A catch-all selector (an
    empty name_token is a substring of every string) would have claimed them and
    regressed the existing inlet and nozzle leaves.
    """
    m = _selection()
    for sid in N1_TIEIN_SELECTION_IDS:
        for name, expect in (
            ("FLANGE 1 of BRANCH /P-711101-P1B-01/B1", "service_piping:02_INLET"),
            ("/P711101004", "service_piping:02_INLET"),
            ("PCOMPONENT 3 of BRANCH /P-711101-P1B-01/B1", "service_piping:02_INLET"),
            ("/CP2-V-71101/N1", "nozzle:N1"),
        ):
            _, leaf, _ = m.classify(name, "Group", -8.4, 1.127, 0.0, sid)
            assert leaf == expect, (
                f"{name} was claimed as {leaf!r} instead of {expect!r}")

# --- GLB integrity: the export must actually be loadable ------------------------
#
# `gltfpack -cc` produced a GLB that declared two glTF buffers but shipped the data
# for only one, so every bufferView pointed at a buffer that was never written and
# 59 of 68 accessors referenced bytes past the end of the file. The file could not
# be opened by a viewer at all: trimesh raised IndexError. It stayed invisible
# because every other test reads node names, `extras` and accessor min/max and
# never dereferences vertex data, so all 172 passed on an unopenable file.
#
# These tests dereference the actual bytes, which is the check that was missing.

def test_the_glb_declares_buffers_it_actually_writes():
    """A GLB must not reference a glTF buffer that carries no data.

    gltfpack -cc wrote buffers[0] (the real data) and buffers[1] sized to the full
    unquantized extent, never written, with every bufferView pointing at
    buffers[1]. Measured on the committed build: 2 buffers, 48 of 56 accessors
    past the end of the single BIN chunk.
    """
    import struct

    blob = open(GLB_PATH, "rb").read()
    assert blob[:4] == b"glTF", f"not a GLB: magic {blob[:4]!r}"
    off, chunks = 12, {}
    while off + 8 <= len(blob):
        clen, ctype = struct.unpack_from("<II", blob, off)
        chunks[ctype] = (off + 8, clen)
        off += 8 + clen
    assert 0x4E4F534A in chunks, "GLB has no JSON chunk"
    assert 0x004E4942 in chunks, "GLB has no BIN chunk"
    js, jl = chunks[0x4E4F534A]
    g = json.loads(blob[js:js + jl].decode("utf-8"))
    assert len(g.get("buffers", [])) == 1, (
        f"GLB declares {len(g.get('buffers', []))} buffers but only the BIN chunk "
        f"carries data; accessors referencing the others cannot resolve")


def test_every_glb_accessor_resolves_inside_the_bin_chunk():
    """Each accessor's byte range must fall inside the data actually present.

    This is the assertion whose absence let a broken export pass 172 tests. A
    viewer reading these offsets gets nothing to draw, so the symptom is an empty
    viewport - indistinguishable from "the change was too small to see".
    """
    import struct

    blob = open(GLB_PATH, "rb").read()
    off, chunks = 12, {}
    while off + 8 <= len(blob):
        clen, ctype = struct.unpack_from("<II", blob, off)
        chunks[ctype] = (off + 8, clen)
        off += 8 + clen
    js, jl = chunks[0x4E4F534A]
    g = json.loads(blob[js:js + jl].decode("utf-8"))
    bin_len = chunks[0x004E4942][1]

    dangling = []
    for i, a in enumerate(g["accessors"]):
        bv = g["bufferViews"][a["bufferView"]]
        base = (bv.get("byteOffset", 0) or 0) + (a.get("byteOffset", 0) or 0)
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        isize = {5126: 4, 5125: 4, 5123: 2, 5121: 1}[int(a["componentType"])]
        if base + a["count"] * ncomp * isize > bin_len:
            dangling.append(i)
    assert not dangling, (
        f"{len(dangling)} of {len(g['accessors'])} accessors point past the "
        f"{bin_len:,}-byte BIN chunk (first: {dangling[:8]}). The GLB cannot be "
        f"rendered; see the -cc note in scripts/build_twin_pipeline.py.")


def test_the_glb_loads_in_a_real_gltf_reader():
    """The strongest form: hand the file to a reader and demand real geometry.

    Every other test in this file can pass on a file no viewer can open, because
    they read metadata. This one dereferences the vertices.
    """
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(GLB_PATH, force="scene")
    geoms = list(scene.geometry.values())
    assert geoms, "trimesh loaded the GLB but found no geometry"
    total_verts = sum(len(g.vertices) for g in geoms)
    assert total_verts > 100_000, (
        f"only {total_verts:,} vertices decoded; expected the full scene, so the "
        f"buffer references are still not resolving")

    # trimesh hands back geometry in LOCAL coordinates, so the node transforms
    # have to be applied before the plant position can be checked.
    # `to_geometry()` bakes the world transforms in (`dump` is deprecated).
    world = scene.to_geometry()
    pts = np.asarray(world.vertices)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    assert pts.shape[0] == total_verts, (
        f"concatenated {pts.shape[0]:,} verts but the geometries hold {total_verts:,}")
    # The unit is a ~50 m vessel: reject a coordinate frame that is not metres.
    extent = hi - lo
    assert 10.0 < extent[0] < 200.0, (
        f"scene x extent {extent[0]:.1f} is not a plausible metre-scale plant; "
        f"bounds {np.round(lo, 1).tolist()}..{np.round(hi, 1).tolist()} suggest the "
        f"node transforms were not applied")
    # The N1 tie-in sits at ty 1.127, tz 0.0, spanning tx -17.00..-8.58, so the
    # scene must straddle tz=0 and reach negative tx. This is the specific piece of
    # geometry that was invisible while the export was broken.
    assert lo[0] < -17.0, f"scene tx min {lo[0]:.2f} does not cover the N1 tie-in"
    assert hi[0] > -8.0, f"scene tx max {hi[0]:.2f} does not cover the N1 tie-in"
    assert lo[2] < 0.0 < hi[2], "scene does not straddle tz=0 where N1 sits"
