"""Instance-local cuRobo 0.8 mesh loader for rapidly replaced validation scenes."""

from types import MethodType

import numpy as np


def _load_validation_mesh(mesh_data, mesh):
    import warp as wp
    from curobo._src.geom.data.data_mesh import WarpMeshCache

    # Match cuRobo's vertex/index conversion, ownership and cache representation.
    # Only the BVH construction algorithm changes; every triangle is retained.
    vertices, faces = mesh.get_mesh_data()
    vertices = wp.array(vertices, dtype=wp.vec3, device=mesh_data._wp_device)
    faces = wp.array(np.ravel(faces), dtype=int, device=mesh_data._wp_device)
    value = wp.Mesh(points=vertices, indices=faces, bvh_constructor="lbvh")
    return WarpMeshCache(mesh.name, value.id, vertices, faces, value)


def use_gpu_validation_meshes(scene_model):
    """Use Warp's GPU BVH builder on this fine checker's future scene updates.

    Optimizer meshes retain their original builder/gradient-query ordering.
    The owner still synchronizes before replacing or releasing mesh buffers and
    clears the name-based cache on every update. Nothing is patched globally.
    """
    meshes = scene_model.data.meshes
    if meshes is not None:
        meshes._load_mesh_to_warp = MethodType(_load_validation_mesh, meshes)
