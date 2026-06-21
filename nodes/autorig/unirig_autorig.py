"""
BD_AutoRigUniRig — placeholder node for UniRig (SIGGRAPH 2025).

UniRig has heavy dependencies (spconv, flash_attn, torch_scatter,
torch_cluster) that don't auto-install cleanly in every ComfyUI
environment. This node is registered as a stub that returns a clear
error message until the dependencies are vendored or auto-installed.

For humanoid characters (the studio's 27-char batch), use
BD_AutoRigMIA — it's faster (<1s vs several seconds), gives better
humanoid quality, and has no external dependencies beyond the
auto-downloaded MIA weights.

TODO: vendor the UniRig inference code following the MIA pattern.
Upstream: https://github.com/VAST-AI-Research/UniRig (MIT)
"""

from comfy_api.latest import io

from ..mesh.types import TrimeshInput


class BD_AutoRigUniRig(io.ComfyNode):
    """Placeholder — UniRig integration TBD. Use BD_AutoRigMIA for now."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_AutoRigUniRig",
            display_name="BD AutoRig (UniRig) — TBD",
            category="🧠BrainDead/AutoRig",
            description=(
                "PLACEHOLDER. UniRig integration is planned but not yet "
                "vendored — its dependencies (spconv, flash_attn, etc.) "
                "don't auto-install cleanly. For humanoid characters, "
                "use BD_AutoRigMIA instead.\n\n"
                "Tracking: ARTS-37 follow-up. PRs welcome."
            ),
            inputs=[
                TrimeshInput("mesh"),
            ],
            outputs=[
                io.String.Output(display_name="fbx_path"),
            ],
        )

    @classmethod
    def execute(cls, mesh) -> io.NodeOutput:
        raise NotImplementedError(
            "BD_AutoRigUniRig is a placeholder. The UniRig inference code "
            "isn't vendored into ComfyUI-BrainDead yet (waiting on a clean "
            "install path for spconv / flash_attn). Use BD_AutoRigMIA for "
            "humanoid characters in the meantime."
        )


UNIRIG_V3_NODES = [BD_AutoRigUniRig]
UNIRIG_NODES = {"BD_AutoRigUniRig": BD_AutoRigUniRig}
UNIRIG_DISPLAY_NAMES = {"BD_AutoRigUniRig": "BD AutoRig (UniRig) — TBD"}
