"""
BrainDead Prompt Iterator Nodes for ComfyUI
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Provides nodes for iterating through multiple prompts with automatic filename generation.
Perfect for batch processing character sheets, multi-view renders, and sequential generation.
"""

import json
import random
from typing import Dict, List, Tuple, Any

# Global state management for tracking iteration position
ITERATOR_STATE: Dict[str, Dict[str, Any]] = {}


class BD_PromptIterator:
    """
    Basic prompt iterator that cycles through a list of prompts
    and generates corresponding filenames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "prompt 1\nprompt 2\nprompt 3",
                    "dynamicPrompts": False
                }),
                "mode": (["sequential", "manual", "single"], {
                    "default": "sequential"
                }),
                "base_filename": ("STRING", {
                    "default": "output",
                    "multiline": False
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False
                }),
                "manual_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Reset",
                    "label_off": "Continue"
                }),
                "workflow_id": ("STRING", {
                    "default": "default",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "filename", "current_index", "total_count", "status")
    FUNCTION = "iterate_prompt"
    CATEGORY = "BrainDead/Prompt"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = kwargs.get("mode", "sequential")
        if mode == "sequential":
            return float("NaN")
        return False

    def iterate_prompt(self, prompts: str, mode: str, base_filename: str,
                      filenames: str = "", manual_index: int = 0,
                      reset: bool = False, workflow_id: str = "default") -> Tuple:
        global ITERATOR_STATE

        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        filename_list = [f.strip() for f in filenames.strip().split('\n') if f.strip()] if filenames else []

        if not prompt_list:
            return ("", base_filename, 0, 0, "Error: No prompts provided")

        total_count = len(prompt_list)

        if workflow_id not in ITERATOR_STATE:
            ITERATOR_STATE[workflow_id] = {"index": 0, "iteration": 0}

        state = ITERATOR_STATE[workflow_id]

        if reset:
            state["index"] = 0
            state["iteration"] = 0

        if mode == "manual":
            current_index = max(0, min(manual_index, total_count - 1))
        elif mode == "single":
            current_index = 0
        else:  # sequential
            current_index = state["index"]
            state["index"] = (state["index"] + 1) % total_count
            if state["index"] == 0:
                state["iteration"] += 1

        current_prompt = prompt_list[current_index]

        if filename_list and current_index < len(filename_list):
            current_filename = filename_list[current_index]
        else:
            current_filename = f"{base_filename}_{current_index:03d}"

        status = f"Prompt {current_index + 1}/{total_count}"
        if mode == "sequential":
            status += f" (Iteration {state['iteration'] + 1})"

        return (current_prompt, current_filename, current_index, total_count, status)


class BD_PromptIteratorAdvanced:
    """
    Advanced prompt iterator with templates, suffix lists, seed modes,
    and more control options for complex batch workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "default": "face only headshot, facing camera directly\nLeft profile View - rotate face 90 degrees left\nRight profile View - rotate face 90 degrees right\nBack View - direct back of the head",
                    "dynamicPrompts": False
                }),
                "mode": (["sequential", "manual", "random", "single"], {
                    "default": "sequential"
                }),
                "filename_mode": (["list", "suffix_list", "template", "index"], {
                    "default": "suffix_list"
                }),
                "base_filename": ("STRING", {
                    "default": "character",
                    "multiline": False
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "placeholder": "One filename per line (for 'list' mode)"
                }),
                "suffixes": ("STRING", {
                    "multiline": True,
                    "default": "_front\n_left\n_right\n_back",
                    "dynamicPrompts": False,
                    "placeholder": "One suffix per line (for 'suffix_list' mode)"
                }),
                "filename_template": ("STRING", {
                    "default": "{base}_{index:03d}_{suffix}",
                    "multiline": False,
                    "placeholder": "Template with {base}, {index}, {suffix} placeholders"
                }),
                "prepend_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add before each prompt"
                }),
                "append_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add after each prompt"
                }),
                "manual_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "loop_mode": (["once", "loop", "ping_pong"], {
                    "default": "loop"
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Reset",
                    "label_off": "Continue"
                }),
                "generation_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
                "seed_mode": (["fixed", "increment_batch", "increment_prompt", "random"], {
                    "default": "increment_batch"
                }),
                "workflow_id": ("STRING", {
                    "default": "default",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "filename", "current_index", "total_count", "status", "seed", "debug_info")
    FUNCTION = "iterate_prompt_advanced"
    CATEGORY = "BrainDead/Prompt"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = kwargs.get("mode", "sequential")
        if mode in ["sequential", "random"]:
            return float("NaN")
        return False

    def iterate_prompt_advanced(self, prompts: str, mode: str, filename_mode: str,
                               base_filename: str, filenames: str = "",
                               suffixes: str = "", filename_template: str = "",
                               prepend_text: str = "", append_text: str = "",
                               manual_index: int = 0, loop_mode: str = "loop",
                               reset: bool = False, generation_seed: int = -1,
                               seed_mode: str = "increment_batch",
                               workflow_id: str = "default") -> Tuple:
        global ITERATOR_STATE

        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        filename_list = [f.strip() for f in filenames.strip().split('\n') if f.strip()] if filenames else []
        suffix_list = [s.strip() for s in suffixes.strip().split('\n') if s.strip()] if suffixes else []

        if not prompt_list:
            return ("", base_filename, 0, 0, "Error: No prompts provided", 0, "")

        total_count = len(prompt_list)

        state_key = f"{workflow_id}_advanced"
        if state_key not in ITERATOR_STATE:
            ITERATOR_STATE[state_key] = {
                "index": 0,
                "iteration": 0,
                "direction": 1,
                "random_order": list(range(total_count)),
                "base_seed": generation_seed if generation_seed >= 0 else random.randint(0, 2147483647),
                "current_seed": generation_seed if generation_seed >= 0 else random.randint(0, 2147483647)
            }

        state = ITERATOR_STATE[state_key]

        if reset:
            state["index"] = 0
            state["iteration"] = 0
            state["direction"] = 1
            state["random_order"] = list(range(total_count))
            if generation_seed >= 0:
                state["base_seed"] = generation_seed
                state["current_seed"] = generation_seed
            else:
                state["base_seed"] = random.randint(0, 2147483647)
                state["current_seed"] = state["base_seed"]

        if mode == "random":
            random.shuffle(state["random_order"])

        if mode == "manual":
            current_index = max(0, min(manual_index, total_count - 1))
        elif mode == "single":
            current_index = 0
        elif mode == "random":
            current_index = state["random_order"][state["index"]]
            state["index"] = (state["index"] + 1) % total_count
            if state["index"] == 0:
                state["iteration"] += 1
        else:  # sequential
            current_index = state["index"]

            if loop_mode == "once":
                if state["index"] < total_count - 1:
                    state["index"] += 1
            elif loop_mode == "ping_pong":
                state["index"] += state["direction"]
                if state["index"] >= total_count - 1:
                    state["direction"] = -1
                    state["index"] = total_count - 1
                elif state["index"] <= 0:
                    state["direction"] = 1
                    state["index"] = 0
                    state["iteration"] += 1
            else:  # loop
                state["index"] = (state["index"] + 1) % total_count
                if state["index"] == 0:
                    state["iteration"] += 1

        base_prompt = prompt_list[current_index]
        current_prompt = f"{prepend_text}{base_prompt}{append_text}".strip()

        if filename_mode == "list" and filename_list and current_index < len(filename_list):
            current_filename = filename_list[current_index]
        elif filename_mode == "suffix_list" and suffix_list:
            suffix = suffix_list[current_index] if current_index < len(suffix_list) else f"_{current_index:03d}"
            current_filename = f"{base_filename}{suffix}"
        elif filename_mode == "template":
            suffix = suffix_list[current_index] if current_index < len(suffix_list) else ""
            current_filename = filename_template.format(
                base=base_filename,
                index=current_index,
                suffix=suffix.lstrip('_')
            )
        else:  # index mode
            current_filename = f"{base_filename}_{current_index:03d}"

        status = f"Prompt {current_index + 1}/{total_count}"
        if mode == "sequential":
            status += f" | Iteration {state['iteration'] + 1}"
            if loop_mode == "ping_pong":
                status += " (ping-pong)"
        elif mode == "random":
            status += " (random)"

        output_seed = state["current_seed"]

        should_increment = False
        if seed_mode == "increment_prompt":
            should_increment = True
        elif seed_mode == "increment_batch" and current_index == 0 and state["iteration"] > 0:
            should_increment = True
        elif seed_mode == "random":
            output_seed = random.randint(0, 2147483647)
            state["current_seed"] = output_seed

        if should_increment and seed_mode != "random":
            state["current_seed"] = (state["current_seed"] + 1) % 2147483648
            output_seed = state["current_seed"]

        debug_info = json.dumps({
            "mode": mode,
            "filename_mode": filename_mode,
            "current_index": current_index,
            "state_index": state["index"],
            "iteration": state["iteration"],
            "loop_mode": loop_mode,
            "filename": current_filename,
            "seed": output_seed,
            "seed_mode": seed_mode
        }, indent=2)

        return (current_prompt, current_filename, current_index, total_count, status, output_seed, debug_info)


class BD_PromptIteratorDynamic:
    """
    Dynamic prompt iterator that accepts multiple string inputs
    and cycles through them with automatic filename generation.
    Supports up to 20 connected prompt inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "mode": (["sequential", "manual", "random", "single"], {
                    "default": "sequential"
                }),
                "filename_mode": (["auto_index", "suffix_list", "template"], {
                    "default": "auto_index"
                }),
                "base_filename": ("STRING", {
                    "default": "output",
                    "multiline": False
                }),
            },
            "optional": {
                "prompt_1": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_2": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_3": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_4": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_5": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_6": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_7": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "prompt_8": ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False}),
                "suffixes": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False,
                    "placeholder": "One suffix per line (for suffix_list mode)"
                }),
                "filename_template": ("STRING", {
                    "default": "{base}_{index:03d}",
                    "multiline": False,
                    "placeholder": "{base}, {index}, {suffix}"
                }),
                "manual_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "Reset", "label_off": "Continue"}),
                "generation_seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1, "display": "number"}),
                "seed_mode": (["fixed", "increment_batch", "increment_prompt", "random"], {"default": "increment_batch"}),
                "workflow_id": ("STRING", {"default": "default", "multiline": False}),
            }
        }

        # Add more prompt inputs (up to 20)
        for i in range(9, 21):
            inputs["optional"][f"prompt_{i}"] = ("STRING", {"multiline": True, "default": "", "forceInput": True, "dynamicPrompts": False})

        return inputs

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("prompt", "filename", "current_index", "total_count", "status", "seed")
    FUNCTION = "iterate_prompts"
    CATEGORY = "BrainDead/Prompt"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        mode = kwargs.get("mode", "sequential")
        if mode in ["sequential", "random"]:
            return float("NaN")
        return False

    def iterate_prompts(self, mode: str, filename_mode: str, base_filename: str,
                       suffixes: str = "", filename_template: str = "",
                       manual_index: int = 0, reset: bool = False,
                       generation_seed: int = -1, seed_mode: str = "increment_batch",
                       workflow_id: str = "default", **kwargs) -> Tuple:
        global ITERATOR_STATE

        # Collect all prompt inputs
        prompt_list = []
        for i in range(1, 21):
            prompt_key = f"prompt_{i}"
            if prompt_key in kwargs and kwargs[prompt_key]:
                prompt_value = kwargs[prompt_key]
                if isinstance(prompt_value, str) and prompt_value.strip():
                    prompt_list.append(prompt_value.strip())

        if not prompt_list:
            return ("", base_filename, 0, 0, "Error: No prompts provided", 0)

        suffix_list = [s.strip() for s in suffixes.strip().split('\n') if s.strip()] if suffixes else []
        total_count = len(prompt_list)

        state_key = f"{workflow_id}_dynamic"
        if state_key not in ITERATOR_STATE:
            ITERATOR_STATE[state_key] = {
                "index": 0,
                "iteration": 0,
                "random_order": list(range(total_count)),
                "base_seed": generation_seed if generation_seed >= 0 else random.randint(0, 2147483647),
                "current_seed": generation_seed if generation_seed >= 0 else random.randint(0, 2147483647)
            }

        state = ITERATOR_STATE[state_key]

        if reset:
            state["index"] = 0
            state["iteration"] = 0
            state["random_order"] = list(range(total_count))
            if generation_seed >= 0:
                state["base_seed"] = generation_seed
                state["current_seed"] = generation_seed
            else:
                state["base_seed"] = random.randint(0, 2147483647)
                state["current_seed"] = state["base_seed"]

        if mode == "random":
            random.shuffle(state["random_order"])

        if mode == "manual":
            current_index = max(0, min(manual_index, total_count - 1))
        elif mode == "single":
            current_index = 0
        elif mode == "random":
            current_index = state["random_order"][state["index"]]
            state["index"] = (state["index"] + 1) % total_count
            if state["index"] == 0:
                state["iteration"] += 1
        else:  # sequential
            current_index = state["index"]
            state["index"] = (state["index"] + 1) % total_count
            if state["index"] == 0:
                state["iteration"] += 1

        current_prompt = prompt_list[current_index]

        if filename_mode == "suffix_list" and suffix_list:
            suffix = suffix_list[current_index] if current_index < len(suffix_list) else f"_{current_index:03d}"
            current_filename = f"{base_filename}{suffix}"
        elif filename_mode == "template":
            suffix = suffix_list[current_index] if current_index < len(suffix_list) else ""
            current_filename = filename_template.format(
                base=base_filename,
                index=current_index,
                suffix=suffix.lstrip('_')
            )
        else:  # auto_index
            current_filename = f"{base_filename}_{current_index:03d}"

        output_seed = state["current_seed"]

        should_increment = False
        if seed_mode == "increment_prompt":
            should_increment = True
        elif seed_mode == "increment_batch" and current_index == 0 and state["iteration"] > 0:
            should_increment = True
        elif seed_mode == "random":
            output_seed = random.randint(0, 2147483647)

        if should_increment and seed_mode != "random":
            state["current_seed"] = (state["current_seed"] + 1) % 2147483648
            output_seed = state["current_seed"]

        status = f"Prompt {current_index + 1}/{total_count}"
        if mode == "sequential":
            status += f" (Iteration {state['iteration'] + 1})"
        elif mode == "random":
            status += " (random)"

        return (current_prompt, current_filename, current_index, total_count, status, output_seed)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "BD_PromptIterator": BD_PromptIterator,
    "BD_PromptIteratorAdvanced": BD_PromptIteratorAdvanced,
    "BD_PromptIteratorDynamic": BD_PromptIteratorDynamic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BD_PromptIterator": "BD Prompt Iterator",
    "BD_PromptIteratorAdvanced": "BD Prompt Iterator (Advanced)",
    "BD_PromptIteratorDynamic": "BD Prompt Iterator (Dynamic)",
}
