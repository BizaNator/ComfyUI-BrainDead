"""
Helper utilities for BD TRELLIS2 nodes.

These run in the main ComfyUI process (no subprocess isolation).
"""

import numpy as np
from PIL import Image
import torch


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor to PIL Image.

    Args:
        tensor: ComfyUI IMAGE format [B, H, W, C] or [H, W, C], float32 0-1

    Returns:
        PIL Image (RGB)
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch

    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI IMAGE tensor.

    Args:
        pil_image: PIL Image (RGB or RGBA)

    Returns:
        ComfyUI IMAGE format [1, H, W, C], float32 0-1
    """
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    np_img = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).unsqueeze(0)
    return tensor


def smart_crop_square(
    pil_image: Image.Image,
    mask_np: np.ndarray,
    margin_ratio: float = 0.1,
    background_color: tuple = (128, 128, 128),
) -> Image.Image:
    """
    Extract object with margin, pad to square.

    Args:
        pil_image: Input RGBA image (after mask applied)
        mask_np: Numpy mask array (H, W), values 0-255
        margin_ratio: Padding around object (default 10%)
        background_color: RGB tuple for background (default gray)

    Returns:
        RGB PIL Image - square, with specified background color
    """
    alpha_threshold = 0.8 * 255
    bbox_coords = np.argwhere(mask_np > alpha_threshold)

    if len(bbox_coords) == 0:
        print("[BD TRELLIS2] Warning: No object found in mask, returning original image")
        w, h = pil_image.size
        size = max(w, h)
        canvas = Image.new('RGB', (size, size), background_color)
        canvas.paste(pil_image.convert('RGB'), ((size - w) // 2, (size - h) // 2))
        return canvas

    y_min, x_min = bbox_coords.min(axis=0)
    y_max, x_max = bbox_coords.max(axis=0)

    obj_w = x_max - x_min
    obj_h = y_max - y_min
    obj_size = max(obj_w, obj_h)
    margin = int(obj_size * margin_ratio)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_size = (obj_size / 2) + margin

    crop_x1 = int(center_x - half_size)
    crop_y1 = int(center_y - half_size)
    crop_x2 = int(center_x + half_size)
    crop_y2 = int(center_y + half_size)
    crop_size = crop_x2 - crop_x1

    if crop_size < 1:
        crop_size = 1
        crop_x2 = crop_x1 + 1
        crop_y2 = crop_y1 + 1

    img_w, img_h = pil_image.size
    canvas = Image.new('RGB', (crop_size, crop_size), background_color)

    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(img_w, crop_x2)
    src_y2 = min(img_h, crop_y2)

    dst_x = src_x1 - crop_x1
    dst_y = src_y1 - crop_y1

    cropped = pil_image.crop((src_x1, src_y1, src_x2, src_y2))

    cropped_np = np.array(cropped.convert('RGBA')).astype(np.float32) / 255
    alpha = cropped_np[:, :, 3:4]
    bg = np.array(background_color, dtype=np.float32) / 255
    blended = cropped_np[:, :, :3] * alpha + bg * (1 - alpha)
    cropped_rgb = Image.fromarray((blended * 255).astype(np.uint8))

    canvas.paste(cropped_rgb, (dst_x, dst_y))

    return canvas


# Config dataclass for model settings
class Trellis2ModelConfig:
    """Lightweight config object for TRELLIS2 model settings."""

    def __init__(
        self,
        model_name: str = "microsoft/TRELLIS.2-4B",
        resolution: str = "1024_cascade",
        attn_backend: str = "flash_attn",
        vram_mode: str = "keep_loaded",
    ):
        self.model_name = model_name
        self.resolution = resolution
        self.attn_backend = attn_backend
        self.vram_mode = vram_mode

    def __repr__(self):
        return f"Trellis2ModelConfig(resolution={self.resolution}, vram_mode={self.vram_mode})"


def fix_normals_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Fix face normals to point outward after CuMesh unify_face_orientations.

    CuMesh.unify_face_orientations() ensures consistent winding per connected
    component, but does NOT guarantee outward-facing normals. This function
    checks each component and flips it if normals point inward.

    Uses connected-component analysis with a centroid-based heuristic:
    for each component, if the average face normal points toward the mesh
    center rather than away, flip all faces in that component.

    Args:
        vertices: (V, 3) float32 mesh vertices
        faces: (F, 3) int mesh faces

    Returns:
        faces: (F, 3) with corrected winding order
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_faces = len(faces)
    if n_faces == 0:
        return faces

    faces = faces.copy()

    # Compute face normals via cross product
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Normalize (avoid division by zero)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    face_normals = face_normals / norms

    # Compute face centers
    face_centers = (v0 + v1 + v2) / 3.0

    # Build face adjacency graph from shared edges
    # Each edge is represented by sorted vertex pair
    edge_to_faces = {}
    for fi in range(n_faces):
        for ei in range(3):
            e = tuple(sorted([faces[fi, ei], faces[fi, (ei + 1) % 3]]))
            if e in edge_to_faces:
                edge_to_faces[e].append(fi)
            else:
                edge_to_faces[e] = [fi]

    # Build sparse adjacency matrix
    rows, cols = [], []
    for edge_faces in edge_to_faces.values():
        if len(edge_faces) == 2:
            rows.extend([edge_faces[0], edge_faces[1]])
            cols.extend([edge_faces[1], edge_faces[0]])

    if not rows:
        # No adjacency — use global heuristic
        mesh_center = vertices.mean(axis=0)
        to_outside = face_centers - mesh_center
        dots = np.sum(face_normals * to_outside, axis=1)
        if dots.mean() < 0:
            faces = faces[:, ::-1]
            print(f"[BD TRELLIS2] fix_normals_outward: flipped all {n_faces} faces (no adjacency)")
        return faces

    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = connected_components(graph, directed=False)

    # For each component, check if normals point outward
    mesh_center = vertices.mean(axis=0)
    total_flipped = 0

    for c in range(n_comp):
        comp_mask = labels == c
        comp_centers = face_centers[comp_mask]
        comp_normals = face_normals[comp_mask]

        # Direction from mesh center to face center = "outward"
        to_outside = comp_centers - mesh_center
        dots = np.sum(comp_normals * to_outside, axis=1)

        if dots.mean() < 0:
            # Majority of normals point inward — flip this component
            idx = np.where(comp_mask)[0]
            faces[idx] = faces[idx][:, ::-1]
            total_flipped += len(idx)

    if total_flipped > 0:
        print(f"[BD TRELLIS2] fix_normals_outward: flipped {total_flipped}/{n_faces} faces ({100*total_flipped/n_faces:.1f}%) across {n_comp} components")
    else:
        print(f"[BD TRELLIS2] fix_normals_outward: all {n_faces} faces OK ({n_comp} components)")

    return faces
