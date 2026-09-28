import json
import os
import sys
import pytest

# Ensure simulator/stage1 is on sys.path
sys.path.insert(0, os.path.abspath('simulator/stage1'))
import dynamic

REGISTRY_PATH = os.path.abspath('config/tag_registry.json')
PID_EXTRACT_PATH = os.path.abspath('docs/index/extracts/cad_pid_extraction.json')
GLB_PATH = os.path.abspath('simulator/stage1/assets/final_twin.glb')
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.fixture(scope="module")
def registry():
    assert os.path.exists(REGISTRY_PATH), f"Registry missing: {REGISTRY_PATH}"
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def pid_extract():
    assert os.path.exists(PID_EXTRACT_PATH), f"PID extract missing: {PID_EXTRACT_PATH}"
    with open(PID_EXTRACT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_registry_schema(registry):
    assert "_metadata" in registry
    assert "equipment" in registry
    assert "nozzles" in registry
    assert "valves" in registry
    assert "instruments" in registry
    assert "controllers" in registry
    assert "CP2-V-71101" in registry["equipment"]

def test_representative_set_present(registry):
    representative_set = [
        "UZV-054",
        "PV-003B",
        "PSV-001A",
        "PSV-001B",
        "LT-002",
        "PT-003",
        "FT-002"
    ]
    all_tags = {**registry["valves"], **registry["instruments"]}
    for tag in representative_set:
        assert tag in all_tags, f"Representative tag {tag} missing from tag registry!"

# Keys whose string values ASSERT an equipment identity. A designator that only
# ever appears in a free-text note is a mention, not an attestation, so prose
# keys are deliberately excluded from grounding.
_IDENTITY_KEYS = frozenset({
    "tag", "primary_element", "associated_restriction", "associated_instrument",
    "associated_valve", "associated_transmitter", "controller", "controllers",
    "indicator", "pipe", "twin_train_tag", "ifc_source_element",
})


def _asserted_designators(node, out=None):
    """Designators the extract asserts, taken from identity-bearing fields only.

    Two failure modes are closed here:

    1. Substring-on-serialised-blob. Prose counted as grounding, so a note
       reading "HIPPS 2oo3 with 711-UZV-053" satisfied a pid_ref even when
       UZV-053 had no row at all - and a note could even declare a tag
       unattested and still pass.
    2. Requiring exact whole-value equality. That is too strict, because the
       extractor qualifies a value: primary_element is
       "711-FE-001 (Orifice Plate)", so the asserted designator is the value's
       leading token, not the value.

    So: read identity fields only, then take the leading token of each value.
    """
    if out is None:
        out = set()
    stack = [(False, node)]
    while stack:
        is_identity, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                stack.append((k in _IDENTITY_KEYS, v))
        elif isinstance(cur, list):
            for v in cur:
                stack.append((is_identity, v))
        elif isinstance(cur, str) and is_identity:
            out.add(cur)
            head = cur.split("(")[0].strip()
            if head:
                out.add(head)
    return out


def test_pid_identity_matches_dwg_extract(registry, pid_extract):
    extracted_valve_tags = {v["tag"] for v in pid_extract["valves"]}
    extracted_instr_tags = {i["tag"] for i in pid_extract["instruments"]}
    # A primary element such as FE-001 is not its own instrument row: the P&ID
    # names it inside its transmitter's primary_element field, and a secondary
    # element such as RO-051 is named inside associated_restriction. Those are
    # field VALUES equal to the designator, so exact value equality covers them
    # without admitting prose.
    extract_values = _asserted_designators(pid_extract)

    # Check valves
    for tag, v in registry["valves"].items():
        assert "pid_ref" in v
        assert "cad_dwg_source" in v
        if v["pid_ref"] is None:
            # An entry may be IFC-attested only, but then it must SAY so. A
            # null pid_ref is not a loophole: a fabricated designator still
            # fails the grounding assert below. UZV-053 is the live case - real
            # geometry and a real SelectionID, but absent from the extract, so
            # its rating and duty stay unknown rather than being copied from the
            # neighbouring UZV-054.
            assert v.get("ifc_only_attestation"), (
                f"Valve {tag} has no pid_ref and no ifc_only_attestation "
                f"statement. Either ground it in the CAD extract or declare "
                f"explicitly that only the IFC attests it.")
            continue
        # Verify that the pid_ref is grounded in CAD extract. A secondary element
        # such as RO-051 has no row of its own: the P&ID names it inside
        # UZV-052's associated_restriction field, so grounding must also accept
        # the extract's asserted values. A fabricated designator matches neither.
        assert (v["pid_ref"] in extracted_valve_tags
                or v["pid_ref"] in extract_values), f"Valve {tag} pid_ref {v['pid_ref']} not found in CAD extract!"

    # Check instruments
    for tag, inst in registry["instruments"].items():
        assert "pid_ref" in inst
        assert "cad_dwg_source" in inst
        if inst["pid_ref"] is None:
            assert inst.get("ifc_only_attestation"), (
                f"Instrument {tag} has no pid_ref and no ifc_only_attestation "
                f"statement. PDZT-051 is the live case.")
            continue
        grounded = (inst["pid_ref"] in extracted_instr_tags
                    or inst["pid_ref"] in extract_values)
        assert grounded, f"Instrument {tag} pid_ref {inst['pid_ref']} not found in CAD extract!"

    # FE-001 is a primary element, so require the specific field that proves it
    # rather than accepting a bare substring match.
    fe_owners = [i for i in pid_extract["instruments"]
                 if "FE-001" in str(i.get("primary_element", ""))]
    assert fe_owners, (
        "FE-001 claims to be a primary element but no P&ID instrument row "
        "names it in primary_element")
    assert registry["instruments"]["FE-001"]["primary_element_of"] in {
        i["tag"].split("-", 1)[1] for i in fe_owners}

def test_3d_identity_consistency(registry):
    import pygltflib
    assert os.path.exists(GLB_PATH)
    glb = pygltflib.GLTF2().load(GLB_PATH)
    glb_node_names = {node.name for node in glb.nodes if node.name}
    
    all_elements = {**registry["valves"], **registry["instruments"]}
    for tag, elem in all_elements.items():
        gltf_ref = elem.get("gltf_ref")
        if gltf_ref:
            # Addressable 3D node must be present in GLB nodes or match the tag namespace
            assert elem.get("addressable_3d_node") == "verified", f"Tag {tag} has gltf_ref but addressable_3d_node is not verified"
            assert gltf_ref in glb_node_names or gltf_ref.replace("tag:", "") in glb_node_names, (
                f"Tag {tag} declares gltf_ref '{gltf_ref}' which was not found in final_twin.glb nodes!"
            )
        else:
            # When gltf_ref is None, addressable_3d_node must NOT be verified
            assert elem.get("addressable_3d_node") in ("not_yet_available", "not_in_local_skid")


def test_glb_tag_nodes_are_registry_listed(registry):
    """Governance invariant: the registry is the authority for 2D-to-3D binding.

    Its own rule states a tag may appear in the scene only if the registry lists
    it. This test enforces the reverse direction, which nothing else covered:
    every ``tag:*`` node in final_twin.glb must be declared by some registry
    entry's ``gltf_ref``.

    The builder enforces the same rule at the source: it materialises a ``tag:``
    node only when the registry addresses that exact node name, and otherwise
    emits the geometry as ``unregistered_tag:<ID>``. So no undeclared ``tag:``
    node may exist at all.

    Two identities still lack registry grounding, because registering them
    requires a Phase 1 P&ID/DWG extraction refresh and inventing extract rows is
    forbidden by the registry's zero-guessing rule. Their geometry must remain in
    the asset - asserted below - so the gap stays visible instead of drifting.
    """
    import pygltflib
    glb = pygltflib.GLTF2().load(GLB_PATH)
    glb_tag_nodes = {n.name for n in glb.nodes if n.name and n.name.startswith("tag:")}

    declared = set()
    for section in ("valves", "instruments", "controllers", "equipment"):
        for entry in registry.get(section, {}).values():
            if isinstance(entry, dict) and entry.get("gltf_ref"):
                declared.add(entry["gltf_ref"])

    undeclared = glb_tag_nodes - declared
    assert undeclared == set(), (
        "Undeclared GLB tag nodes present. "
        f"undeclared={sorted(undeclared)}. The builder only materialises a "
        "`tag:` node when the registry addresses it by that exact name; anything "
        "else must be emitted as `unregistered_tag:<ID>` so its geometry "
        "survives without becoming an unaddressable tag. Register the tag with "
        "P&ID/DWG evidence, or check why the registry no longer addresses it."
    )

    # The ungrounded identities must still be present, so the gap stays visible
    # instead of drifting. The two are in different states, and each is checked
    # where its provenance actually lives:
    #
    #   SP-CP02 has a manifest tag identity, so it is emitted as
    #   `unregistered_tag:SP-CP02` and carries its source entities in the asset.
    #   UZV-053 has no grounded selector, so it stays unclassified in the review
    #   group and its identity is recorded in the canonical hierarchy.
    #
    # Losing either means tagged equipment silently vanished - the failure mode
    # the prune exemption exists to prevent.
    demoted = {n.name for n in glb.nodes
               if n.name and n.name.startswith("unregistered_tag:")}
    assert "unregistered_tag:SP-CP02" in demoted, (
        "SP-CP02 geometry is no longer emitted. Its manifest tag identity is "
        "exempt from the connectivity prune precisely so this tag cannot vanish."
    )

    hier_path = os.path.join(REPO, "geometry", "canonical_hierarchy.json")
    assert os.path.exists(hier_path), (
        f"canonical hierarchy missing at {hier_path}; the generated provenance "
        "record is required to verify that no tagged identity was dropped")
    with open(hier_path, encoding="utf-8") as _f:
        hier_text = json.dumps(json.load(_f))
    assert "UZV-053" in hier_text, (
        "UZV-053 is no longer present in the canonical hierarchy. It has no "
        "grounded selector, so it belongs in the review group; if it has been "
        "dropped, the known-gap list should be updated only once the identity is "
        "genuinely resolved by P&ID/DWG evidence."
    )


def test_simulation_identity_consistency(registry):
    sim = dynamic.SeparatorDynamicSimulator()
    state = sim.state
    state_fields = set(state.__dataclass_fields__.keys())

    all_elements = {**registry["valves"], **registry["instruments"], **registry["controllers"]}
    for tag, elem in all_elements.items():
        sim_ref = elem.get("simulation_ref")
        if sim_ref:
            refs = [r.strip() for r in sim_ref.split(',')]
            for r in refs:
                # Must be a state field, or constant, or simulator attribute
                exists = (r in state_fields) or hasattr(dynamic, r) or hasattr(sim, r)
                assert exists, f"Tag {tag} declares simulation_ref '{r}' which is missing in dynamic.py!"

def test_uzv054_simulation_identity_not_conflated(registry):
    """Regression test: UZV-054 must not be conflated with UZV-051 simulation variable."""
    uzv054 = registry["valves"]["UZV-054"]
    uzv051 = registry["valves"]["UZV-051"]
    
    # Equipment identity: UZV-054 has its own distinct P&ID and 3D node
    assert uzv054["pid_ref"] == "711-UZV-054"
    assert uzv054["gltf_ref"] == "tag:UZV-054"
    assert uzv054["addressable_3d_node"] == "verified"
    
    # Simulation identity: must NOT be mapped to uzv051_open
    assert uzv054["simulation_ref"] is None, "UZV-054 must not force mapping to uzv051_open!"
    assert "control_interlock_dependency" in uzv054, "UZV-054 must document control/interlock dependency"
    assert uzv054["status"] == "PARTIALLY VERIFIED"
    assert uzv054["verified_triplet"] is False
    
    # UZV-051 legitimately owns uzv051_open in dynamic.py
    assert uzv051["simulation_ref"] == "uzv051_open"
    assert uzv051["pid_ref"] == "711-UZV-051"

def test_embedded_3d_assets_explicit_status(registry):
    """Regression test: line instrument FT-002 without discrete CAD solid must not claim addressable GLTF node."""
    entry = registry["instruments"].get("FT-002")
    assert entry is not None, "FT-002 missing from registry"
    assert entry["physical_3d_presence"] == "verified"
    assert entry["addressable_3d_node"] == "not_yet_available"
    assert entry["gltf_ref"] is None, "FT-002 must have gltf_ref=None while not modeled as discrete 3D solid"
    assert entry["status"] == "PARTIALLY VERIFIED"
    assert entry["verified_triplet"] is False, "FT-002 cannot be a full verified triplet without addressable 3D node"

def test_fv001_isolated_addressable_node(registry):
    """Workstream A: FV-001 isolated as verified addressable 3D node and complete triplet."""
    fv001 = registry["valves"].get("FV-001")
    assert fv001 is not None, "FV-001 missing from registry"
    assert fv001["physical_3d_presence"] == "verified"
    assert fv001["addressable_3d_node"] == "verified"
    assert fv001["gltf_ref"] == "tag:FV-001"
    assert fv001["status"] == "VERIFIED"
    assert fv001["verified_triplet"] is True

def test_unknown_mappings_remain_unset(registry):
    """Regression test: unknown or unseparated fields must remain None / not_yet_available."""
    # UZV-001 has no direct dynamic simulation ODE
    assert registry["valves"]["UZV-001"]["simulation_ref"] is None
    assert registry["valves"]["UZV-001"]["verified_triplet"] is False
    assert registry["valves"]["UZV-001"]["status"] == "PARTIALLY VERIFIED"
    
    # PT-052 has no direct simulation ODE
    assert registry["instruments"]["PT-052"]["simulation_ref"] is None
    assert registry["instruments"]["PT-052"]["verified_triplet"] is False
    
    # RESOLVED, and the previous assertion is deliberately inverted here.
    # PZT-001 was recorded as unaddressable on the assumption that it was the
    # same instrument as PT-003 (sharing a nozzle bridle). The authoritative
    # IFC carries distinct groups /CP2-711-PT-003 and /CP2-711-PZT-001, and
    # the P&ID gives PZT-001 its own entry and its own Cable 1003 trip, so it
    # is a separate instrument with its own addressable node.
    pzt001 = registry["instruments"]["PZT-001"]
    assert pzt001["addressable_3d_node"] == "verified"
    assert pzt001["gltf_ref"] == "tag:PZT-001"
    assert pzt001["verified_triplet"] is True

    # FE-001 is placed geometry but has no independent dynamic state of its own;
    # it is the orifice plate read BY FT-001, so the field stays honestly unset.
    fe001 = registry["instruments"]["FE-001"]
    assert fe001["gltf_ref"] == "tag:FE-001"
    assert fe001["addressable_3d_node"] == "verified"
    assert fe001["simulation_ref"] is None, (
        "FE-001 is a primary element measured by FT-001; giving it its own "
        "simulation_ref would double-count the flow measurement")

    # LZT-001 shares the same resolution as PZT-001 and is asserted positively
    # in test_fully_verified_triplets; guard against silent regression here.
    assert registry["instruments"]["LZT-001"]["gltf_ref"] == "tag:LZT-001"

def test_fully_verified_triplets(registry):
    """Verify elements that satisfy all 3 criteria: P&ID + addressable 3D node + simulation state."""
    full_triplets = [
        ("valves", "PV-003B"),
        ("valves", "UZV-002"),
        ("valves", "UZV-003"),
        ("valves", "PV-003A"),
        ("valves", "FV-001"),
        ("instruments", "LT-002"),
        ("instruments", "LZT-001"),
        ("instruments", "PZT-001"),
        ("instruments", "PT-003"),
        ("instruments", "TT-051")
    ]
    for category, tag in full_triplets:
        entry = registry[category][tag]
        assert entry["pid_ref"] is not None
        assert entry["gltf_ref"] is not None
        assert entry["addressable_3d_node"] == "verified"
        assert entry["simulation_ref"] is not None
        assert entry["status"] == "VERIFIED"
        assert entry["verified_triplet"] is True


def test_psv_flare_skid_status(registry):
    """PSV-001A and PSV-001B are located downstream on HP flare header skid per authentic plant model."""
    for tag in ("PSV-001A", "PSV-001B"):
        psv = registry["valves"][tag]
        assert psv["gltf_ref"] is None
        assert psv["addressable_3d_node"] == "not_in_local_skid"
        assert psv["status"] == "PARTIALLY VERIFIED"
        assert psv["verified_triplet"] is False

