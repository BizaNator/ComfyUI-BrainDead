"""CPU wrapper checks; studio integration runs when its local assets are available."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'native_skeleton_conversion', ROOT / 'utils/native_skeleton_conversion.py')
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)
SOURCE = Path(os.environ.get('BDB_NATIVE_TEST_FBX',
    '/mnt/tank/Studio/Brains/Characters/_base_models/codex_male_v02/exports/SK_COB_Male_Base_v02.fbx'))
BLEND = Path(os.environ.get('BDB_NATIVE_TEST_BLEND',
    '/mnt/tank/Studio/Brains/Characters/_base_models/codex_male_v02/COB_Male_Base_v02.blend'))
BLENDER = os.environ.get('BDB_NATIVE_TEST_BLENDER') or shutil.which('blender')
BACKEND = os.environ.get('BDB_NATIVE_TEST_CONVERTER', '')
SCRIPT, REFERENCE, CONTRACT = conversion.defaults('NATIVE_DEVICE', BACKEND)
HAS_STUDIO = bool(BLENDER and all(p.is_file() for p in (SOURCE, BLEND, SCRIPT, REFERENCE, CONTRACT)))


class NativeInputTests(unittest.TestCase):
    def test_filename_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            with self.assertRaisesRegex(ValueError, 'Filename'):
                conversion.convert('missing.fbx', output, filename='../escape')
            self.assertFalse(output.exists())

    def test_missing_source_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            with self.assertRaisesRegex(ValueError, 'existing rigged'):
                conversion.convert(Path(folder) / 'missing.fbx', output)
            self.assertFalse(output.exists())

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown target'):
            conversion.defaults('UNKNOWN_TARGET')


@unittest.skipUnless(HAS_STUDIO, 'Requires studio reference bundle, reviewed v02 and CPU Blender 5.1.2')
class NativeBlenderIntegrationTests(unittest.TestCase):
    def test_real_fbx_keeps_fingers_and_morphs(self):
        before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='bdb_native_test_') as folder:
            result = conversion.convert(SOURCE, folder, filename='NativeProbe', blender_executable=BLENDER,
                                        converter_script=BACKEND)
            fbx = Path(result['fbx_path'])
            self.assertTrue(fbx.is_file())
            self.assertTrue(Path(result['blend_path']).is_file())
            report = json.loads(Path(result['report_path']).read_text())
            self.assertEqual(report['profile'], 'NATIVE_DEVICE')
            self.assertFalse(report['engine_playback_verified'])
            self.assertEqual(hashlib.sha256(fbx.read_bytes()).hexdigest(), report['full_fbx_sha256'])
            audits = json.loads((fbx.parent.parent / 'export_audits.json').read_text())
            full = audits[fbx.name]
            self.assertTrue(full['structural_pass'], full['errors'])
            fingers = [f'{d}_{i:02}_{s}' for d in ('thumb','index','middle','ring','pinky')
                       for i in (1,2,3) for s in ('l','r')]
            self.assertTrue(all(full['weight_totals'][name] > 0 for name in fingers))
            self.assertEqual(len(full['morphs']), 8)
            self.assertTrue((fbx.parent.parent.parent / 'blender.log').is_file())
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), before)

    def test_missing_actual_finger_weight_stops_before_export(self):
        with tempfile.TemporaryDirectory(prefix='bdb_native_negative_') as folder:
            root = Path(folder)
            fixture = root / 'missing_finger.blend'
            script = root / 'break_finger.py'
            script.write_text('import bpy\n'
                'bpy.ops.wm.open_mainfile(filepath=' + repr(str(BLEND)) + ')\n'
                "hand=bpy.data.objects['Hands_L']\n"
                "hand.vertex_groups.remove(hand.vertex_groups['index_03_l'])\n"
                'bpy.ops.wm.save_as_mainfile(filepath=' + repr(str(fixture)) + ')\n')
            subprocess.run([BLENDER, '--background', '--factory-startup', '--disable-autoexec',
                '--threads', '2', '--python-exit-code', '1', '--python', str(script)],
                check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            with self.assertRaisesRegex(RuntimeError, 'Source needs actual finger weights.*index_03_l'):
                conversion.convert(fixture, root, filename='MustFail', blender_executable=BLENDER,
                                   converter_script=BACKEND)
            self.assertFalse(list(root.glob('MustFail_*/bundle/exports/*.fbx')))

    @unittest.skipUnless(BACKEND, 'Set BDB_NATIVE_TEST_CONVERTER to BrainDead Blender 1.3.0')
    def test_player_profile_uses_shared_resolver_and_exports(self):
        script, reference, contract = conversion.defaults('NATIVE_PLAYER', BACKEND)
        self.assertEqual(script, Path(BACKEND))
        self.assertTrue(reference.is_file() and contract.is_file())
        before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='bdb_player_test_') as folder:
            result = conversion.convert(SOURCE, folder, filename='PlayerProbe',
                target_profile='NATIVE_PLAYER', converter_script=BACKEND, blender_executable=BLENDER)
            report = json.loads(Path(result['report_path']).read_text())
            self.assertEqual(report['profile'], 'NATIVE_PLAYER')
            self.assertFalse(report['engine_playback_verified'])
            audits = json.loads((Path(result['fbx_path']).parent.parent / 'export_audits.json').read_text())
            full = audits[Path(result['fbx_path']).name]
            self.assertTrue(full['structural_pass'], full['errors'])
            self.assertEqual(full['target_bones'], 279)  # root object makes 280 FBX hierarchy entries
            self.assertEqual(len(full['morphs']), 8)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
