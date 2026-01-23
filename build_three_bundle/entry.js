/**
 * Three.js bundle entry point for BD Mesh Inspector.
 * Bundles only what we need: core THREE, OrbitControls, GLTFLoader.
 * Exports as window globals for use in viewer_inspector.html.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

window.THREE = THREE;
window.OrbitControls = OrbitControls;
window.GLTFLoader = GLTFLoader;
