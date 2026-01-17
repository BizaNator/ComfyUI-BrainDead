"""
V3 API BrainDead Prompt Iterator Nodes for ComfyUI
Created by BizaNator for BrainDeadGuild.com
A Biloxi Studios Inc. Production

Provides nodes for iterating through multiple prompts with automatic filename generation.
Perfect for batch processing character sheets, multi-view renders, and sequential generation.
"""

import json
import random
import time
from typing import Dict, Any, Tuple

from comfy_api.latest import io

# Global state management for tracking iteration position
ITERATOR_STATE: Dict[str, Dict[str, Any]] = {}


class BD_PromptIterator(io.ComfyNode):
    """
    Basic prompt iterator that cycles through a list of prompts
    and generates corresponding filenames.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PromptIterator",
            display_name="BD Prompt Iterator",
            category="🧠BrainDead/Prompt",
            description="Basic prompt iterator that cycles through a list of prompts and generates corresponding filenames.",
            inputs=[
                io.String.Input("prompts", multiline=True, default="prompt 1\nprompt 2\nprompt 3", dynamic_prompts=False),
                io.Combo.Input("mode", options=["sequential", "manual", "single"], default="sequential"),
                io.String.Input("base_filename", multiline=False, default="output"),
                io.String.Input("filenames", multiline=True, default="", dynamic_prompts=False, optional=True),
                io.Int.Input("manual_index", default=0, min=0, max=9999, step=1, optional=True),
                io.Boolean.Input("reset", default=False, optional=True),
                io.String.Input("workflow_id", multiline=False, default="default", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="filename"),
                io.Int.Output(display_name="current_index"),
                io.Int.Output(display_name="total_count"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, prompts, mode, base_filename, filenames="",
                           manual_index=0, reset=False, workflow_id="default") -> str:
        if mode == "sequential":
            return f"seq_{time.time()}"  # Always different = always re-execute
        return f"{mode}_{manual_index}_{workflow_id}"

    @classmethod
    def execute(cls, prompts: str, mode: str, base_filename: str,
                filenames: str = "", manual_index: int = 0,
                reset: bool = False, workflow_id: str = "default") -> io.NodeOutput:
        global ITERATOR_STATE

        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        filename_list = [f.strip() for f in filenames.strip().split('\n') if f.strip()] if filenames else []

        if not prompt_list:
            return io.NodeOutput("", base_filename, 0, 0, "Error: No prompts provided")

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

        return io.NodeOutput(current_prompt, current_filename, current_index, total_count, status)


class BD_PromptIteratorAdvanced(io.ComfyNode):
    """
    Advanced prompt iterator with templates, suffix lists, seed modes,
    and more control options for complex batch workflows.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BD_PromptIteratorAdvanced",
            display_name="BD Prompt Iterator (Advanced)",
            category="🧠BrainDead/Prompt",
            description="Advanced prompt iterator with templates, suffix lists, seed modes, and more control options.",
            inputs=[
                io.String.Input("prompts", multiline=True,
                               default="face only headshot, facing camera directly\nLeft profile View - rotate face 90 degrees left\nRight profile View - rotate face 90 degrees right\nBack View - direct back of the head",
                               dynamic_prompts=False),
                io.Combo.Input("mode", options=["sequential", "manual", "random", "single"], default="sequential"),
                io.Combo.Input("filename_mode", options=["list", "suffix_list", "template", "index"], default="suffix_list"),
                io.String.Input("base_filename", multiline=False, default="character"),
                io.String.Input("filenames", multiline=True, default="", dynamic_prompts=False, optional=True),
                io.String.Input("suffixes", multiline=True, default="_front\n_left\n_right\n_back", dynamic_prompts=False, optional=True),
                io.String.Input("filename_template", multiline=False, default="{base}_{index:03d}_{suffix}", optional=True),
                io.String.Input("prepend_text", multiline=False, default="", optional=True),
                io.String.Input("append_text", multiline=False, default="", optional=True),
                io.Int.Input("manual_index", default=0, min=0, max=9999, step=1, optional=True),
                io.Combo.Input("loop_mode", options=["once", "loop", "ping_pong"], default="loop", optional=True),
                io.Boolean.Input("reset", default=False, optional=True),
                io.Int.Input("generation_seed", default=-1, min=-1, max=2147483647, step=1, optional=True),
                io.Combo.Input("seed_mode", options=["fixed", "increment_batch", "increment_prompt", "random"], default="increment_batch", optional=True),
                io.String.Input("workflow_id", multiline=False, default="default", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="filename"),
                io.Int.Output(display_name="current_index"),
                io.Int.Output(display_name="total_count"),
                io.String.Output(display_name="status"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="debug_info"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, prompts, mode, filename_mode, base_filename,
                           filenames="", suffixes="", filename_template="",
                           prepend_text="", append_text="", manual_index=0,
                           loop_mode="loop", reset=False, generation_seed=-1,
                           seed_mode="increment_batch", workflow_id="default") -> str:
        if mode in ["sequential", "random"]:
            return f"adv_{time.time()}"  # Always different = always re-execute
        return f"{mode}_{manual_index}_{workflow_id}"

    @classmethod
    def execute(cls, prompts: str, mode: str, filename_mode: str,
                base_filename: str, filenames: str = "",
                suffixes: str = "", filename_template: str = "",
                prepend_text: str = "", append_text: str = "",
                manual_index: int = 0, loop_mode: str = "loop",
                reset: bool = False, generation_seed: int = -1,
                seed_mode: str = "increment_batch",
                workflow_id: str = "default") -> io.NodeOutput:
        global ITERATOR_STATE

        prompt_list = [p.strip() for p in prompts.strip().split('\n') if p.strip()]
        filename_list = [f.strip() for f in filenames.strip().split('\n') if f.strip()] if filenames else []
        suffix_list = [s.strip() for s in suffixes.strip().split('\n') if s.strip()] if suffixes else []

        if not prompt_list:
            return io.NodeOutput("", base_filename, 0, 0, "Error: No prompts provided", 0, "")

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

        return io.NodeOutput(current_prompt, current_filename, current_index, total_count, status, output_seed, debug_info)


class BD_PromptIteratorDynamic(io.ComfyNode):
    """
    Dynamic prompt iterator that accepts multiple string inputs
    and cycles through them with automatic filename generation.
    Supports up to 20 connected prompt inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Build inputs list
        inputs = [
            io.Combo.Input("mode", options=["sequential", "manual", "random", "single"], default="sequential"),
            io.Combo.Input("filename_mode", options=["auto_index", "suffix_list", "template"], default="auto_index"),
            io.String.Input("base_filename", multiline=False, default="output"),
        ]

        # Add 20 optional prompt inputs
        for i in range(1, 21):
            inputs.append(io.String.Input(f"prompt_{i}", multiline=True, default="", dynamic_prompts=False, optional=True, force_input=True))

        # Add remaining optional inputs
        inputs.extend([
            io.String.Input("suffixes", multiline=True, default="", dynamic_prompts=False, optional=True),
            io.String.Input("filename_template", multiline=False, default="{base}_{index:03d}", optional=True),
            io.Int.Input("manual_index", default=0, min=0, max=9999, step=1, optional=True),
            io.Boolean.Input("reset", default=False, optional=True),
            io.Int.Input("generation_seed", default=-1, min=-1, max=2147483647, step=1, optional=True),
            io.Combo.Input("seed_mode", options=["fixed", "increment_batch", "increment_prompt", "random"], default="increment_batch", optional=True),
            io.String.Input("workflow_id", multiline=False, default="default", optional=True),
        ])

        return io.Schema(
            node_id="BD_PromptIteratorDynamic",
            display_name="BD Prompt Iterator (Dynamic)",
            category="🧠BrainDead/Prompt",
            description="Dynamic prompt iterator that accepts up to 20 string inputs and cycles through them with automatic filename generation.",
            inputs=inputs,
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="filename"),
                io.Int.Output(display_name="current_index"),
                io.Int.Output(display_name="total_count"),
                io.String.Output(display_name="status"),
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, mode, filename_mode, base_filename,
                           suffixes="", filename_template="",
                           manual_index=0, reset=False,
                           generation_seed=-1, seed_mode="increment_batch",
                           workflow_id="default", **kwargs) -> str:
        if mode in ["sequential", "random"]:
            return f"dyn_{time.time()}"  # Always different = always re-execute
        return f"{mode}_{manual_index}_{workflow_id}"

    @classmethod
    def execute(cls, mode: str, filename_mode: str, base_filename: str,
                suffixes: str = "", filename_template: str = "",
                manual_index: int = 0, reset: bool = False,
                generation_seed: int = -1, seed_mode: str = "increment_batch",
                workflow_id: str = "default", **kwargs) -> io.NodeOutput:
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
            return io.NodeOutput("", base_filename, 0, 0, "Error: No prompts provided", 0)

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

        return io.NodeOutput(current_prompt, current_filename, current_index, total_count, status, output_seed)


# =============================================================================
# V3 Node List for Extension
# =============================================================================

PROMPT_V3_NODES = [
    BD_PromptIterator,
    BD_PromptIteratorAdvanced,
    BD_PromptIteratorDynamic,
]

# =============================================================================
# V1 Compatibility - Node Mappings
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
