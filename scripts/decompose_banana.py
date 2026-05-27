"""
Convex decomposition of banana_g_collision.stl using CoACD.

Produces banana_col_N.stl files (one per convex hull piece) in the
pick_banana meshes/ directory, then prints the XML snippet to paste
into pick_banana.xml.template.

Usage:
    python3 scripts/decompose_banana.py
"""

import pathlib

MESHES_DIR = pathlib.Path("src/sim_models/scenes/pick_banana/meshes")
INPUT_STL = MESHES_DIR / "banana_g_collision.stl"
OUTPUT_PREFIX = "banana_col_"


def run_coacd():
    import coacd
    import trimesh

    mesh = trimesh.load(str(INPUT_STL))
    print(f"Input: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    # CoACD parameters tuned for a C-shaped banana:
    #   threshold=0.04  — concavity tolerance (lower = more pieces, more accurate)
    #   max_convex_hull=6 — cap at 6 pieces to keep XML manageable
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=0.04,
        max_convex_hull=6,
        preprocess_mode="auto",
    )
    print(f"CoACD produced {len(parts)} convex pieces")

    # Remove old decomposition files
    for old in MESHES_DIR.glob(f"{OUTPUT_PREFIX}*.stl"):
        old.unlink()
        print(f"  removed {old.name}")

    xml_assets = []
    xml_geoms = []

    for i, (verts, faces) in enumerate(parts):
        name = f"{OUTPUT_PREFIX}{i}"
        out_path = MESHES_DIR / f"{name}.stl"
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        part_mesh.export(str(out_path))
        vcount = len(verts)
        fcount = len(faces)
        print(f"  piece {i}: {vcount} verts, {fcount} faces → {out_path.name}")

        xml_assets.append(f'    <mesh name="{name}" file="{{{{MESHES_DIR}}}}/{name}.stl" scale="1 1 1"/>')
        xml_geoms.append(
            f'      <geom type="mesh" mesh="{name}" rgba="0 0 0 0"\n'
            f'            contype="1" conaffinity="1" group="0"\n'
            f'            friction="3.0 0.5 0.05" condim="4" mass="0"\n'
            f'            solimp="0.9 0.95 0.002 0.5 2" solref="0.010 1"/>'
        )

    print("\n--- XML asset entries (add to <asset>) ---")
    for line in xml_assets:
        print(line)

    print("\n--- XML geom entries (replace banana_col_mesh geom in banana body) ---")
    print("      <!-- mass on first piece only, rest mass=0 -->")
    # Put mass on first piece
    xml_geoms[0] = xml_geoms[0].replace('mass="0"', 'mass="0.03"')
    for line in xml_geoms:
        print(line)

    return len(parts)


if __name__ == "__main__":
    try:
        n = run_coacd()
        print(f"\nDone. {n} STL files written to {MESHES_DIR}/")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install coacd trimesh scipy")
