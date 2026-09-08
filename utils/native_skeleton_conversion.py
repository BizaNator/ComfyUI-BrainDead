"""CPU native completion through the studio's shared BrainDead Blender CLI."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def defaults(profile):
    brains = Path(r'B:\Brains') if os.name == 'nt' else Path('/mnt/tank/Studio/Brains')
    script = brains / 'Tools/BrainDeadBlender/scripts/utils/convert_skeleton_target.py'
    folder = brains / 'Characters/_uefn_reference/native_contracts/codex_device_v01'
    if profile == 'NATIVE_DEVICE':
        return script, folder / 'CP_Device_Mannequin_named.fbx', folder / 'CP_Device_Mannequin_named.provenance.json'
    if profile == 'FAB_UEFN':
        return script, brains / 'Skills/char-designer/skm_uefn_mannequin.FBX', None
    raise ValueError('Unknown target profile: ' + profile)


def convert(source_asset, output_root, *, target_profile='NATIVE_DEVICE', filename='NativeCharacter',
            source_rig='root', reference_fbx='', reference_contract='', converter_script='',
            blender_executable='', require_fingers=True):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', filename):
        raise ValueError('Filename must use letters, digits and underscores')
    source = Path(source_asset).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in ('.blend', '.fbx'):
        raise ValueError('Choose an existing rigged .blend or .fbx')
    default_script, default_reference, default_contract = defaults(target_profile)
    script = Path(converter_script or default_script)
    reference = Path(reference_fbx or default_reference)
    contract = Path(reference_contract or default_contract) if reference_contract or default_contract else None
    for path in (script, reference, contract):
        if path and not path.is_file():
            raise FileNotFoundError(str(path))
    blender = blender_executable or shutil.which('blender')
    if not blender:
        raise FileNotFoundError('Set blender_executable to Blender 5.1.2')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    version = subprocess.run([blender, '--version'], capture_output=True, text=True, check=True, env=env).stdout
    if not version.startswith('Blender 5.1.2'):
        raise ValueError('The verified native completion recipe requires Blender 5.1.2')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix=filename + '_', dir=root))
    output = job / 'bundle'
    command = [blender, '--background', '--factory-startup', '--disable-autoexec', '--threads', '2',
               '--python-exit-code', '1', '--python', str(script), '--',
               '--source-blend' if source.suffix.lower() == '.blend' else '--source-fbx', str(source),
               '--source-rig', source_rig, '--profile', target_profile, '--reference-fbx', str(reference),
               '--asset-name', filename, '--output', str(output)]
    if contract:
        command += ['--reference-contract', str(contract)]
    if require_fingers:
        command += ['--require-fingers']
    log_path = job / 'blender.log'
    with log_path.open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
    if result.returncode:
        raise RuntimeError('Native conversion failed. Log: ' + str(log_path) + '\n' + log_path.read_text()[-6000:])
    report_path = output / 'conversion.json'
    report = json.loads(report_path.read_text())
    audits = json.loads((output / 'export_audits.json').read_text())
    if not audits or not all(row['structural_pass'] for row in audits.values()):
        raise ValueError('Native conversion audit failed: ' + str(output / 'export_audits.json'))
    fbx = output / 'exports' / report['primary_fbx']
    blend = output / (filename + '_' + target_profile + '.blend')
    if not fbx.is_file() or not blend.is_file():
        raise ValueError('Conversion returned no actual asset')
    return {'fbx_path': str(fbx), 'blend_path': str(blend), 'report_path': str(report_path)}
