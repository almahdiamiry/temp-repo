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
})
BUILD_INPUT_PATH = os.path.join(REPO, "geometry", "build_input.ifc")
MANIFEST_PATH = os.path.join(REPO, "geometry", "build_input.manifest.json")

# Authentic engineering source extents for the N1 nozzle stub.
N1_SOURCE_DIMS_M = (1.495, 1.090, 1.090)
N1_ELEMENT_COUNT = 1
N1_TRI_COUNT = 240

# M1A/M1B, the vessel's two manhole/access openings, are excluded by declared
# rule. Each carries two IFC group proxies, so the exclusion removes four rows
# and 8,288 triangles from the container that used to hold them. This is a
# measured, declared removal, not drift.
M1A_M1B_ELEMENT_COUNT = 4
M1A_M1B_TRI_COUNT = 8288

# Post-correction container figures, locked as the regression baseline.
#
# The container grew from 61/10,943 to 63/15,647 because the connectivity prune
# no longer discards manifest-tagged fragments (SP-CP02 among them) and because
# exact duplicate geometry is collapsed before emission. Both changes add real
# equipment, so the N1 transfer proof is re-anchored on the current extraction.
#
# It then shrank from 63/15,647 to 59/7,359 when M1A/M1B were excluded.
#
# The pre-correction figures are defined by the transfer itself, not chosen: they
# are the same container with N1 still inside it, so they are exactly the
# post-correction figures plus N1_ELEMENT_COUNT / N1_TRI_COUNT. That is how the
# original 62/11,183 related to 61/10,943, and keeping the relationship (rather
# than hardcoding unrelated numbers) is what makes the delta below meaningful.
PRE_M1A_M1B_CONTAINER_ELEMENT_COUNT = 63
PRE_M1A_M1B_CONTAINER_TRI_COUNT = 15647
CONTAINER_ELEMENT_COUNT = PRE_M1A_M1B_CONTAINER_ELEMENT_COUNT - M1A_M1B_ELEMENT_COUNT
CONTAINER_TRI_COUNT = PRE_M1A_M1B_CONTAINER_TRI_COUNT - M1A_M1B_TRI_COUNT
PRE_CORRECTION_CONTAINER_ELEMENT_COUNT = CONTAINER_ELEMENT_COUNT + N1_ELEMENT_COUNT
PRE_CORRECTION_CONTAINER_TRI_COUNT = CONTAINER_TRI_COUNT + N1_TRI_COUNT

# The six level-instrument interface nozzles, newly placed in their own leaf.
LEVEL_INTERFACE_ELEMENT_COUNT = 6
LEVEL_INTERFACE_TRI_COUNT = 1808

# Validated scene triangle total for this extraction. Asserting the absolute
# total is what catches geometry arriving without being inventoried, or being
# inventoried without being emitted.
#
# The total moved 1,583,143 -> 1,573,691. Both directions are declared changes,
# not drift: -8,288 triangles of M1A/M1B access hardware left the model, -2,972
# of other equipment's geometry left (18 foreign entities that a bare nozzle-tag
# substring match had asserted into the model), and +1,808 of level-instrument
# nozzles entered. The inventory's own accounting closes to unaccounted: 0.
VALIDATED_SCENE_TRIANGLE_TOTAL = 1573691

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


def test_m1a_m1b_access_openings_are_excluded_and_recorded_not_deleted():
    """M1A/M1B are manhole/access openings, not nozzles.

    Engineering inspection identifies them as access hardware, so they must not
    be modelled as vessel nozzles. They must still be accounted for: the rule is
    declared in the manifest and the elements land in the inventory under an
    explicit status, so exclusion is not silent deletion.
    """
    with open(SELECTION_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tokens = [t.upper() for t in manifest["excluded_tokens"]["tokens"]]
    assert "/CP2-V-71101/M1A" in tokens
    assert "/CP2-V-71101/M1B" in tokens

    emitted = _emitted_source_entities()
    assert "/CP2-V-71101/M1A" not in emitted
    assert "/CP2-V-71101/M1B" not in emitted

    inv = _inventory()
    excluded = {r.get("source_entity_name") for r in inv["excluded"]}
    assert "/CP2-V-71101/M1A" in excluded
    assert "/CP2-V-71101/M1B" in excluded


def test_the_m1a_axial_clamp_that_corrupted_geometry_is_gone():
    """The x=7.72 clamp moved vertices and was order-dependent.

    It flattened the vertices of whichever element happened to follow an /M1A
    proxy in document order - /STIFFENE-R-1 was emitted with all eight vertices
    at x=7.72 instead of its true 0.911..1.100. With M1A/M1B excluded by rule
    the clamp is unreachable, so it is removed rather than left as dead code
    that could silently corrupt a future build.
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
    """The permanent scaffold must be fully carried, not partially pruned."""
    emitted = _emitted_source_entities()
    scaffold = [e for e in emitted if "PRIMARY-P1-P1B-TS01" in e]
    assert len(scaffold) >= 2100
    assert not any("PRIMARY-P1-P1B-TS01" in r.get("source_entity_name", "")
                   for r in _inventory()["excluded"])


# --- Canonical build input integrity ---

def test_canonical_build_input_and_manifest_agree():
    """The canonical input must be the validated artifact the manifest records."""
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"canonical build input not present at {BUILD_INPUT_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out = manifest["output"]
    assert out["path"] == "geometry/build_input.ifc"
    assert out["entity_count"] == 787056
    assert out["building_element_proxy_count"] == 28282
    assert out["entity_id_sha256"] == (
        "85613443c8ca0b1696aabbbc92cfe90f47e018c4955e686b13f76577005e627e")

    size = os.path.getsize(BUILD_INPUT_PATH)
    assert size == out["bytes"] == 121855826
    assert manifest["mode"] == "describe-existing"
    assert "equivalence" in manifest["reproduction"]


def test_canonical_build_input_bytes_match_recorded_hash():
    """Hash the build input so silent replacement is detected."""
    if not os.path.exists(BUILD_INPUT_PATH):
        pytest.skip(f"canonical build input not present at {BUILD_INPUT_PATH}")
    h = hashlib.sha256()
    with open(BUILD_INPUT_PATH, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    assert h.hexdigest() == (
        "6c4815daa89ba83735754f8aec4462372051bfee4ff97b7594783778040a1171"), (
        "geometry/build_input.ifc no longer matches the validated artifact; "
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
