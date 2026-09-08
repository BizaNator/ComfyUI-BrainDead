"""Native Device completion for a rigged FBX or morph-enabled Blender source."""
import folder_paths
from comfy_api.latest import io

from ...utils.native_skeleton_conversion import convert


class BD_TargetFortniteSkeleton(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Files can change in place while their string paths stay unchanged.
        # Re-read inputs and produce a newly audited bundle on every queue run.
        return float('nan')

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='BD_TargetFortniteSkeleton', display_name='BD Native Fortnite Skeleton',
            category='🧠BrainDead/AutoRig',
            description='Repose a rigged FBX or blend to the measured Character Device reference on CPU. Keeps existing weights and morphs; exports and audits a new copy.',
            inputs=[
                io.String.Input('source_asset', tooltip='Rigged FBX or .blend with canonical bone names.'),
                io.Combo.Input('target_profile', options=['NATIVE_DEVICE', 'FAB_UEFN'], default='NATIVE_DEVICE'),
                io.String.Input('filename', default='NativeCharacter'),
                io.Boolean.Input('require_fingers', default=True, tooltip='Full characters must weight all 30 finger bones. Disable for standalone non-hand parts.'),
                io.String.Input('source_rig', default='root', optional=True),
                io.String.Input('reference_fbx', default='', optional=True),
                io.String.Input('reference_contract', default='', optional=True),
                io.String.Input('converter_script', default='', optional=True),
                io.String.Input('blender_executable', default='', optional=True),
            ],
            outputs=[io.String.Output(display_name='fbx_path'), io.String.Output(display_name='blend_path'),
                     io.String.Output(display_name='report_path')],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source_asset, target_profile='NATIVE_DEVICE', filename='NativeCharacter',
                require_fingers=True, source_rig='root', reference_fbx='', reference_contract='',
                converter_script='', blender_executable=''):
        result = convert(source_asset, folder_paths.get_output_directory(), target_profile=target_profile,
            filename=filename, require_fingers=require_fingers, source_rig=source_rig,
            reference_fbx=reference_fbx, reference_contract=reference_contract,
            converter_script=converter_script, blender_executable=blender_executable)
        return io.NodeOutput(result['fbx_path'], result['blend_path'], result['report_path'])


NATIVE_SKEL_V3_NODES = [BD_TargetFortniteSkeleton]
NATIVE_SKEL_NODES = {'BD_TargetFortniteSkeleton': BD_TargetFortniteSkeleton}
NATIVE_SKEL_DISPLAY_NAMES = {'BD_TargetFortniteSkeleton': 'BD Native Fortnite Skeleton'}
