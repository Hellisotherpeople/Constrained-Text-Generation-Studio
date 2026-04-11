"""
Constrained Text Generation Studio (CTGS)
An AI writing assistant for constrained text generation using large language models.

Features progressive relaxation and backtracking to avoid crippling model vocabulary
when constraints are enabled.
"""

# ============================================================================
# Imports
# ============================================================================
import os
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import numpy as np
import dearpygui.dearpygui as dpg
import fasttext
import Levenshtein
import pronouncing
import torch
from huggingface_hub import hf_hub_download
from pyphonetics import Metaphone
from scipy import spatial
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from screeninfo import get_monitors


# ============================================================================
# State Management
# ============================================================================

@dataclass
class ConstraintState:
    lipogram_banned_list: List[str] = field(default_factory=list)
    weak_lipogram_banned_list: List[str] = field(default_factory=list)
    reverse_lipogram_required_list: List[str] = field(default_factory=list)
    weak_reverse_lipogram_required_list: List[str] = field(default_factory=list)
    string_in_position_list: List[str] = field(default_factory=list)
    string_in_position_index_list: List[int] = field(default_factory=list)
    starts_with_string: str = ""
    ends_with_string: str = ""
    phonetic_matching_string: str = ""
    semantic_matching_string: str = ""
    semantic_distance_threshold: float = 0.0
    string_edit_string: str = ""
    string_edit_distance_threshold: int = 0
    syllable_number: int = 0
    meter_string: str = ""
    rhyme_string: str = ""
    constrained_length: int = 0
    constrained_gt_length: int = 0
    constrained_lt_length: int = 0
    palindrome_enabled: bool = False
    anagram_string: str = ""
    partial_anagram_string: str = ""
    isogram_count: int = 0
    reverse_isogram_count: int = 0

    def has_active_constraints(self):
        return (
            bool(self.lipogram_banned_list) or bool(self.weak_lipogram_banned_list) or
            bool(self.reverse_lipogram_required_list) or bool(self.weak_reverse_lipogram_required_list) or
            bool(self.string_in_position_list) or bool(self.starts_with_string) or
            bool(self.ends_with_string) or self.constrained_length > 0 or
            self.constrained_gt_length > 0 or self.constrained_lt_length > 0 or
            self.palindrome_enabled or bool(self.phonetic_matching_string) or
            bool(self.semantic_matching_string) or bool(self.anagram_string) or
            bool(self.partial_anagram_string) or bool(self.rhyme_string) or
            bool(self.meter_string) or bool(self.string_edit_string) or
            self.syllable_number > 0 or self.isogram_count > 0 or self.reverse_isogram_count > 0
        )

    def reset(self):
        self.__init__()


@dataclass
class ModelState:
    tokenizer: object = None
    model: object = None
    loading: bool = False
    name: str = ""


@dataclass
class AnalysisState:
    """Holds last generation results for visualization (no extra model calls)."""
    all_candidates: list = field(default_factory=list)
    passing_candidates: list = field(default_factory=list)
    failing_candidates: list = field(default_factory=list)
    per_constraint_stats: dict = field(default_factory=dict)
    generation_history: list = field(default_factory=list)


constraints = ConstraintState()
model_state = ModelState()
analysis_state = AnalysisState()
phonetic_encoder = Metaphone()
fasttext_model = None
_dynamic_tags = []  # tracks filter status tags for cleanup on reset
_generation_thread = None


# ============================================================================
# Constraint Filter Logic
# ============================================================================

def passes_all_constraints(word, cs):
    """Check if (text, probability) tuple passes all active constraints. Early-return on failure."""
    text = word[0]

    if cs.lipogram_banned_list:
        if not all(c not in text for c in cs.lipogram_banned_list):
            return False
    if cs.weak_lipogram_banned_list:
        if not any(c not in cs.weak_lipogram_banned_list for c in text):
            return False
    if cs.reverse_lipogram_required_list:
        if not all(c in text for c in cs.reverse_lipogram_required_list):
            return False
    if cs.weak_reverse_lipogram_required_list:
        if not any(c in cs.weak_reverse_lipogram_required_list for c in text):
            return False
    if cs.string_in_position_list:
        for idx, ch in enumerate(cs.string_in_position_list):
            try:
                if text[cs.string_in_position_index_list[idx]] != ch:
                    return False
            except (IndexError, TypeError):
                return False
    if cs.starts_with_string and not text.startswith(cs.starts_with_string):
        return False
    if cs.ends_with_string and not text.endswith(cs.ends_with_string):
        return False
    if cs.constrained_length > 0 and len(text) != cs.constrained_length:
        return False
    if cs.constrained_gt_length > 0 and len(text) <= cs.constrained_gt_length:
        return False
    if cs.constrained_lt_length > 0 and len(text) >= cs.constrained_lt_length:
        return False
    if cs.palindrome_enabled and text != text[::-1]:
        return False
    if cs.phonetic_matching_string:
        try:
            if phonetic_encoder.phonetics(text) != phonetic_encoder.phonetics(cs.phonetic_matching_string):
                return False
        except Exception:
            return False
    if cs.semantic_matching_string and fasttext_model is not None:
        try:
            v1 = fasttext_model.get_word_vector(text)
            v2 = fasttext_model.get_word_vector(cs.semantic_matching_string)
            sim = 1 - spatial.distance.cosine(v1, v2)
            if sim < cs.semantic_distance_threshold:
                return False
        except Exception:
            return False
    if cs.anagram_string and Counter(text) != Counter(cs.anagram_string):
        return False
    if cs.partial_anagram_string and (Counter(text) - Counter(cs.partial_anagram_string)):
        return False
    if cs.rhyme_string and text not in pronouncing.rhymes(cs.rhyme_string):
        return False
    if cs.meter_string:
        ref_phones = pronouncing.phones_for_word(cs.meter_string)
        word_phones = pronouncing.phones_for_word(text)
        if not ref_phones or not word_phones:
            return False
        if pronouncing.stresses(word_phones[0]) != pronouncing.stresses(ref_phones[0]):
            return False
    if cs.string_edit_string:
        if Levenshtein.distance(text, cs.string_edit_string) > cs.string_edit_distance_threshold:
            return False
    if cs.syllable_number > 0:
        phones = pronouncing.phones_for_word(text)
        if not phones or pronouncing.syllable_count(phones[0]) != cs.syllable_number:
            return False
    if cs.isogram_count > 0:
        for ch in set(text):
            if text.count(ch) > cs.isogram_count:
                return False
    if cs.reverse_isogram_count > 0:
        for ch in set(text):
            if text.count(ch) < cs.reverse_isogram_count:
                return False

    return True


def _compute_per_constraint_stats(word_list):
    """For each active constraint, count how many tokens it alone would reject."""
    stats = {}
    checks = []
    cs = constraints
    if cs.lipogram_banned_list:
        checks.append(("All Strings Banned", ConstraintState(lipogram_banned_list=cs.lipogram_banned_list)))
    if cs.weak_lipogram_banned_list:
        checks.append(("Any Strings Banned", ConstraintState(weak_lipogram_banned_list=cs.weak_lipogram_banned_list)))
    if cs.reverse_lipogram_required_list:
        checks.append(("All Strings Required", ConstraintState(reverse_lipogram_required_list=cs.reverse_lipogram_required_list)))
    if cs.weak_reverse_lipogram_required_list:
        checks.append(("Any Strings Required", ConstraintState(weak_reverse_lipogram_required_list=cs.weak_reverse_lipogram_required_list)))
    if cs.string_in_position_list:
        checks.append(("String In Position", ConstraintState(string_in_position_list=cs.string_in_position_list, string_in_position_index_list=cs.string_in_position_index_list)))
    if cs.starts_with_string:
        checks.append(("Starts With", ConstraintState(starts_with_string=cs.starts_with_string)))
    if cs.ends_with_string:
        checks.append(("Ends With", ConstraintState(ends_with_string=cs.ends_with_string)))
    if cs.constrained_length > 0:
        checks.append(("Length Equal To", ConstraintState(constrained_length=cs.constrained_length)))
    if cs.constrained_gt_length > 0:
        checks.append(("Length Greater Than", ConstraintState(constrained_gt_length=cs.constrained_gt_length)))
    if cs.constrained_lt_length > 0:
        checks.append(("Length Less Than", ConstraintState(constrained_lt_length=cs.constrained_lt_length)))
    if cs.palindrome_enabled:
        checks.append(("Palindrome", ConstraintState(palindrome_enabled=True)))
    if cs.phonetic_matching_string:
        checks.append(("Phonetic", ConstraintState(phonetic_matching_string=cs.phonetic_matching_string)))
    if cs.semantic_matching_string:
        checks.append(("Semantic", ConstraintState(semantic_matching_string=cs.semantic_matching_string, semantic_distance_threshold=cs.semantic_distance_threshold)))
    if cs.anagram_string:
        checks.append(("Anagram", ConstraintState(anagram_string=cs.anagram_string)))
    if cs.partial_anagram_string:
        checks.append(("Partial Anagram", ConstraintState(partial_anagram_string=cs.partial_anagram_string)))
    if cs.rhyme_string:
        checks.append(("Rhyme", ConstraintState(rhyme_string=cs.rhyme_string)))
    if cs.meter_string:
        checks.append(("Meter", ConstraintState(meter_string=cs.meter_string)))
    if cs.string_edit_string:
        checks.append(("Edit Distance", ConstraintState(string_edit_string=cs.string_edit_string, string_edit_distance_threshold=cs.string_edit_distance_threshold)))
    if cs.syllable_number > 0:
        checks.append(("Syllable Count", ConstraintState(syllable_number=cs.syllable_number)))
    if cs.isogram_count > 0:
        checks.append(("Isogram", ConstraintState(isogram_count=cs.isogram_count)))
    if cs.reverse_isogram_count > 0:
        checks.append(("Reverse Isogram", ConstraintState(reverse_isogram_count=cs.reverse_isogram_count)))

    for name, temp_cs in checks:
        rejected = sum(1 for w in word_list if not passes_all_constraints(w, temp_cs))
        stats[name] = rejected

    return stats


# ============================================================================
# Top-K / Top-P Filtering (replaces deprecated transformers function)
# ============================================================================

def apply_top_k_top_p(logits, top_k=0, top_p=0.0):
    """Filter logits using top-k and/or top-p (nucleus) sampling."""
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = float('-inf')
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
        logits[indices_to_remove] = float('-inf')
    return logits


# ============================================================================
# Text Transforms
# ============================================================================

def apply_text_transforms(text):
    """Apply all active text transforms to a decoded token string."""
    if dpg.get_value("upper_case_transform"):
        text = text.upper()
    if dpg.get_value("lower_case_transform"):
        text = text.lower()
    if dpg.get_value("replace_spaces"):
        text = text.replace(' ', '')
    if dpg.get_value("lstrip_transform"):
        text = text.lstrip()
    if dpg.get_value("rstrip_transform"):
        text = text.rstrip()
    if dpg.get_value("strip_transform"):
        text = text.strip()
    if dpg.get_value("capitalize_first_letter_transform"):
        text = text.capitalize()
    if dpg.get_value("alphanumeric_transform"):
        text = ''.join(ch for ch in text if ch.isalnum())
    if dpg.get_value("alpha_transform"):
        text = ''.join(ch for ch in text if ch.isalpha())
    if dpg.get_value("digit_transform"):
        text = ''.join(ch for ch in text if ch.isdigit())
    if dpg.get_value("ascii_transform"):
        text = ''.join(ch for ch in text if ch.isascii())
    return text


# ============================================================================
# Token Generation Engine (progressive relaxation + constraint filtering)
# ============================================================================

def _relaxation_levels(base_k, base_p):
    """Yield progressively relaxed (k, p) pairs to avoid empty results."""
    yield (base_k, base_p)
    if base_k > 0:
        yield (base_k * 2, base_p)
        yield (base_k * 4, base_p)
    if base_p > 0:
        yield (0, min(base_p * 1.5, 0.99))
        yield (0, min(base_p * 2.0, 0.99))
    yield (0, 0.0)  # full vocabulary


def get_constrained_tokens(sequence):
    """Generate candidate tokens with progressive relaxation to prevent vocabulary crippling.

    Instead of applying top-k/top-p first and hoping enough tokens survive constraint
    filtering, this function progressively expands the candidate pool when too few
    tokens pass. It also supports soft constraints (penalty-based) via the constraint
    strength slider.
    """
    if model_state.model is None or model_state.tokenizer is None:
        return []

    try:
        with torch.no_grad():
            input_ids = model_state.tokenizer.encode(sequence, return_tensors="pt")
            try:
                device = next(model_state.model.parameters()).device
                input_ids = input_ids.to(device)
            except StopIteration:
                pass

            outputs = model_state.model(input_ids)
            # Move logits to CPU immediately to free GPU memory
            raw_logits = outputs.logits[:, -1, :].float().cpu()
            del outputs

            temperature = dpg.get_value("temperature")
            if temperature > 0 and temperature != 1.0:
                raw_logits = raw_logits / temperature

            top_p = dpg.get_value("top_p")
            top_k = dpg.get_value("top_k")
            # If user left both at 0, use the configurable default vocabulary size
            if top_k == 0 and top_p == 0.0:
                default_vocab = dpg.get_value("default_vocab_size")
                if default_vocab > 0:
                    top_k = default_vocab
            strength = dpg.get_value("constraint_strength")
            use_relaxation = dpg.get_value("progressive_relaxation")
            has_constraints = constraints.has_active_constraints()

            if use_relaxation and has_constraints:
                levels = list(_relaxation_levels(top_k, top_p))
            else:
                levels = [(top_k, top_p)]

            filter_blanks = dpg.get_value("filter_blank_outputs")

            for eff_k, eff_p in levels:
                logits = apply_top_k_top_p(raw_logits, top_k=eff_k, top_p=eff_p)
                probs = F.softmax(logits, dim=-1)

                # Only decode tokens with non-zero probability (avoids decoding entire vocab)
                nonzero_mask = probs[0] > 0
                nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
                nonzero_probs = probs[0][nonzero_indices]

                # Sort by probability descending
                sorted_order = nonzero_probs.argsort(descending=True)
                sorted_indices = nonzero_indices[sorted_order]
                sorted_probs = nonzero_probs[sorted_order]

                # Decode each token individually — batch_decode with a 1D tensor
                # treats it as one sequence, so we wrap each ID in its own list
                decoded = [model_state.tokenizer.decode([tid]) for tid in sorted_indices.tolist()]

                word_list = []
                for i, raw_text in enumerate(decoded):
                    prob = sorted_probs[i].item()
                    text = apply_text_transforms(raw_text)
                    if filter_blanks and text == "":
                        continue
                    word_list.append((text, prob))

                if not has_constraints:
                    analysis_state.all_candidates = word_list
                    analysis_state.passing_candidates = word_list
                    analysis_state.failing_candidates = []
                    analysis_state.per_constraint_stats = {}
                    return word_list

                # Apply constraints with strength control
                if strength >= 1.0:
                    filtered = [w for w in word_list if passes_all_constraints(w, constraints)]
                    passing = filtered
                    failing = [w for w in word_list if w not in filtered]
                elif strength <= 0.0:
                    filtered = word_list
                    passing = word_list
                    failing = []
                else:
                    passing = []
                    failing = []
                    for w in word_list:
                        (passing if passes_all_constraints(w, constraints) else failing).append(w)
                    penalty_factor = 1.0 - strength
                    filtered = passing + [(t, p * penalty_factor) for t, p in failing]
                    filtered.sort(key=lambda x: x[1], reverse=True)

                # Populate analysis state for visualization (no extra model call)
                analysis_state.all_candidates = word_list
                analysis_state.passing_candidates = passing
                analysis_state.failing_candidates = failing
                analysis_state.per_constraint_stats = _compute_per_constraint_stats(word_list)

                if filtered:
                    return filtered

            return []

    except Exception as e:
        dpg.set_value("status_text", f"Generation error: {e}")
        return []


# ============================================================================
# UI Helper Functions
# ============================================================================

def _help(message):
    """Add a green (?) tooltip next to the last-added DPG item."""
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)


def add_and_load_image(image_path, parent=None):
    width, height, channels, data = dpg.load_image(image_path)
    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)
    if parent is None:
        return dpg.add_image(texture_id)
    return dpg.add_image(texture_id, parent=parent)


# ============================================================================
# UI Callbacks
# ============================================================================

def refresh_token_popup(*args):
    """Refresh the right-click popup with new constrained token predictions."""
    if model_state.model is None:
        dpg.set_value("status_text", "No model loaded. Load a model first.")
        return

    dpg.set_value("status_text", "Predicting tokens...")
    candidates = get_constrained_tokens(dpg.get_value("input_string"))

    dpg.delete_item("token_popup", children_only=True)
    dpg.add_text("Suggested Continuations", parent="token_popup")
    dpg.add_separator(parent="token_popup")

    if candidates:
        max_display = dpg.get_value("max_display_tokens")
        show_all = dpg.get_value("show_all_tokens")
        display_list = candidates if (show_all or max_display == 0) else candidates[:max_display]
        for text, prob in display_list:
            dpg.add_selectable(
                label=f"{text}  ({prob:.2%})",
                user_data=text,
                callback=_add_word_callback,
                parent="token_popup"
            )
        dpg.set_value("status_text", f"Ready - {len(candidates)} candidates (showing {len(display_list)})")
    else:
        dpg.add_text("No results with current constraints", parent="token_popup")
        dpg.set_value("status_text", "No tokens match current constraints")

    # Update analysis visualizations (no extra model call)
    _refresh_analysis()


def _add_word_callback(sender, app_data, user_data):
    """Append clicked token to the input text."""
    current = dpg.get_value("input_string")
    dpg.set_value("input_string", current + str(user_data))
    refresh_token_popup()


def tab_key_generate_callback(*args):
    """F2: Insert a single token directly into the text."""
    if model_state.model is None:
        dpg.set_value("status_text", "No model loaded.")
        return

    text = dpg.get_value("input_string")
    candidates = get_constrained_tokens(text)

    if not candidates:
        dpg.set_value("status_text", "No tokens match current constraints")
        return

    if dpg.get_value("greedy_decoding"):
        chosen = candidates[0][0]
    else:
        words, probs = zip(*candidates)
        p = np.array(probs)
        p = p / p.sum()
        chosen = words[np.random.choice(len(words), p=p)]

    dpg.set_value("input_string", text + chosen)


def generate_tokens_callback(*args):
    """Generate multiple tokens with backtracking on dead ends."""
    global _generation_thread
    if model_state.model is None:
        dpg.set_value("status_text", "No model loaded.")
        return
    if _generation_thread and _generation_thread.is_alive():
        dpg.set_value("status_text", "Generation already in progress...")
        return

    _generation_thread = threading.Thread(target=_generate_tokens_worker, daemon=True)
    _generation_thread.start()


def _generate_tokens_worker():
    """Worker thread for multi-token generation with backtracking."""
    num_tokens = dpg.get_value("num_tokens_to_generate")
    max_bt = dpg.get_value("max_backtracks")
    greedy = dpg.get_value("greedy_decoding")
    text = dpg.get_value("input_string")

    history = []
    backtracks_used = 0
    tokens_generated = 0
    prev_backtracks = 0
    analysis_state.generation_history = []

    while tokens_generated < num_tokens:
        candidates = get_constrained_tokens(text)

        if not candidates:
            # Backtrack: try next-best alternative from a previous step
            if backtracks_used < max_bt and history:
                prev_text, alts = history.pop()
                if alts:
                    chosen_text = alts.pop(0)[0]
                    text = prev_text + chosen_text
                    if alts:
                        history.append((prev_text, alts))
                    backtracks_used += 1
                    tokens_generated += 1
                    dpg.set_value("input_string", text)
                    dpg.set_value("status_text", f"Backtracked! Token {tokens_generated}/{num_tokens}")
                    continue
            dpg.set_value("status_text", f"Dead end after {tokens_generated} tokens")
            return

        if greedy:
            chosen = candidates[0]
            remaining = candidates[1:6]
        else:
            words, probs = zip(*candidates)
            p = np.array(probs)
            p = p / p.sum()
            idx = np.random.choice(len(candidates), p=p)
            chosen = candidates[idx]
            remaining = [c for j, c in enumerate(candidates) if j != idx][:5]

        was_backtrack = backtracks_used > prev_backtracks
        prev_backtracks = backtracks_used
        analysis_state.generation_history.append((chosen[0], chosen[1], was_backtrack))

        history.append((text, remaining))
        text += chosen[0]
        tokens_generated += 1
        dpg.set_value("input_string", text)
        dpg.set_value("status_text", f"Generated {tokens_generated}/{num_tokens} tokens")

    dpg.set_value("status_text", f"Done! Generated {num_tokens} tokens")
    _refresh_history_plot()


def load_model_callback(sender=None, app_data=None, user_data=None, model_name_override=None):
    """Load a HuggingFace model in a background thread."""
    if model_state.loading:
        dpg.set_value("model_status", "Already loading a model...")
        return

    model_name = model_name_override if model_name_override else dpg.get_value("model_name")
    model_state.loading = True
    dpg.set_value("model_status", f"Loading {model_name}...")
    dpg.configure_item("load_model_btn", enabled=False)
    dpg.show_item("model_loading_indicator")

    def _load():
        try:
            precision = dpg.get_value("precision")
            model_state.tokenizer = AutoTokenizer.from_pretrained(model_name)

            dtype = None
            if precision == "16bit":
                dtype = torch.float16

            if torch.cuda.is_available():
                kwargs = {"device_map": "auto"}
                if precision == "8bit":
                    kwargs["load_in_8bit"] = True
                if dtype:
                    kwargs["torch_dtype"] = dtype
                model_state.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                device_label = "CUDA GPU"
            elif torch.backends.mps.is_available():
                # MPS (Apple Silicon): load to CPU first, then move to MPS
                # device_map='auto' doesn't handle MPS well
                load_dtype = dtype or torch.float32
                model_state.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=load_dtype
                ).to("mps")
                device_label = "MPS (Apple Silicon)"
            else:
                load_dtype = dtype or torch.float32
                model_state.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=load_dtype
                )
                device_label = "CPU"

            model_state.name = model_name
            dpg.set_value("model_status", f"{model_name} loaded on {device_label} ({precision})")
        except Exception as e:
            dpg.set_value("model_status", f"Error: {e}")
        finally:
            model_state.loading = False
            dpg.configure_item("load_model_btn", enabled=True)
            dpg.hide_item("model_loading_indicator")

    threading.Thread(target=_load, daemon=True).start()


def turn_filters_off_callback(*args):
    """Reset all constraints and clean up status UI."""
    constraints.reset()
    for tag in _dynamic_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    dpg.set_value("status_text", "All filters reset")


# ============================================================================
# Analysis & Visualization Callbacks
# ============================================================================

def _refresh_analysis(*args):
    """Rebuild all analysis visualizations from the last generation results."""
    _refresh_bar_chart()
    _refresh_pie_chart()
    _refresh_constraint_breakdown()
    _refresh_token_table()


def _refresh_bar_chart():
    """Update the token probability bar chart."""
    if not dpg.does_item_exist("prob_bar_y"):
        return
    dpg.delete_item("prob_bar_y", children_only=True)

    n = dpg.get_value("chart_top_n")
    top_n = analysis_state.all_candidates[:n]
    if not top_n:
        return

    passing_set = set(t for t, p in analysis_state.passing_candidates)
    pass_x, pass_y, fail_x, fail_y = [], [], [], []
    tick_labels = []

    for i, (text, prob) in enumerate(top_n):
        label = text.replace(" ", "\u2423").replace("\n", "\\n")[:12]
        tick_labels.append((label, float(i)))
        if text in passing_set:
            pass_x.append(float(i))
            pass_y.append(prob)
        else:
            fail_x.append(float(i))
            fail_y.append(prob)

    if pass_x:
        dpg.add_bar_series(pass_x, pass_y, label="Pass", parent="prob_bar_y", tag="bar_pass_s")
        if dpg.does_item_exist("green_bar_theme"):
            dpg.bind_item_theme("bar_pass_s", "green_bar_theme")
    if fail_x:
        dpg.add_bar_series(fail_x, fail_y, label="Fail", parent="prob_bar_y", tag="bar_fail_s")
        if dpg.does_item_exist("red_bar_theme"):
            dpg.bind_item_theme("bar_fail_s", "red_bar_theme")

    if dpg.does_item_exist("prob_bar_x"):
        dpg.set_axis_ticks("prob_bar_x", tick_labels)


def _refresh_pie_chart():
    """Update the pass/fail pie chart."""
    if not dpg.does_item_exist("pie_y"):
        return
    dpg.delete_item("pie_y", children_only=True)

    n_pass = len(analysis_state.passing_candidates)
    n_fail = len(analysis_state.failing_candidates)
    n_total = n_pass + n_fail

    if n_total == 0:
        dpg.set_value("constraint_summary_text", "No data yet")
        return

    pct = n_pass / n_total * 100
    dpg.set_value("constraint_summary_text",
                  f"{n_pass} of {n_total} tokens pass ({pct:.1f}%)")

    dpg.add_pie_series(0.5, 0.5, 0.4,
                       [float(n_pass), float(n_fail)],
                       ["Pass", "Fail"],
                       parent="pie_y", normalize=True, format="%0.1f%%")
    dpg.set_axis_limits("pie_x", 0, 1)
    dpg.set_axis_limits("pie_y", 0, 1)


def _refresh_constraint_breakdown():
    """Show how many tokens each individual constraint filters."""
    if not dpg.does_item_exist("constraint_breakdown_group"):
        return
    dpg.delete_item("constraint_breakdown_group", children_only=True)

    stats = analysis_state.per_constraint_stats
    total = len(analysis_state.all_candidates)
    if not stats or total == 0:
        dpg.add_text("No active constraints", parent="constraint_breakdown_group",
                      color=[180, 180, 180])
        return

    for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        dpg.add_text(f"{name}: filters {count}/{total} ({pct:.1f}%)",
                     parent="constraint_breakdown_group")
        dpg.add_progress_bar(default_value=count / total, overlay=f"{count}/{total}",
                             width=-1, parent="constraint_breakdown_group")


def _refresh_token_table(*args):
    """Rebuild the token table from analysis_state."""
    if not dpg.does_item_exist("token_table"):
        return
    # Delete rows (slot=1) but keep column headers (slot=0)
    dpg.delete_item("token_table", children_only=True, slot=1)

    passing_set = set(t for t, p in analysis_state.passing_candidates)
    search_filter = dpg.get_value("token_search_input").lower() if dpg.does_item_exist("token_search_input") else ""

    max_rows = dpg.get_value("max_display_tokens") if dpg.does_item_exist("max_display_tokens") else 200
    show_all = dpg.get_value("show_all_tokens") if dpg.does_item_exist("show_all_tokens") else False
    candidates = analysis_state.all_candidates
    if not show_all and max_rows > 0:
        candidates = candidates[:max_rows]

    for rank, (text, prob) in enumerate(candidates, 1):
        if search_filter and search_filter not in text.lower():
            continue
        passes = text in passing_set
        with dpg.table_row(parent="token_table"):
            dpg.add_text(str(rank))
            dpg.add_selectable(label=text, callback=_table_insert_token,
                               user_data=text, span_columns=False)
            dpg.add_text(f"{prob:.4%}")
            dpg.add_text("Yes" if passes else "No",
                         color=[60, 180, 75] if passes else [200, 60, 60])


def _table_insert_token(sender, app_data, user_data):
    """Insert a token from the table into the input text."""
    current = dpg.get_value("input_string")
    dpg.set_value("input_string", current + str(user_data))


def _refresh_history_plot(*args):
    """Update the generation history line plot."""
    if not dpg.does_item_exist("history_y"):
        return
    dpg.delete_item("history_y", children_only=True)

    hist = analysis_state.generation_history
    if not hist:
        return

    xs = [float(i) for i in range(len(hist))]
    ys = [h[1] for h in hist]

    dpg.add_line_series(xs, ys, label="Token probability", parent="history_y")

    # Mark backtrack points with scatter markers
    bt_xs = [float(i) for i, h in enumerate(hist) if h[2]]
    bt_ys = [h[1] for h in hist if h[2]]
    if bt_xs:
        dpg.add_scatter_series(bt_xs, bt_ys, label="Backtrack", parent="history_y")

    tick_labels = [(h[0].replace(" ", "")[:8], float(i)) for i, h in enumerate(hist)]
    if tick_labels and dpg.does_item_exist("history_x"):
        dpg.set_axis_ticks("history_x", tick_labels)


def _save_text_callback(sender, app_data):
    """Save current text to a file."""
    filepath = app_data.get("file_path_name", "")
    if filepath:
        with open(filepath, "w") as f:
            f.write(dpg.get_value("input_string"))
        dpg.set_value("status_text", f"Saved to {filepath}")


def _load_text_callback(sender, app_data):
    """Load text from a file."""
    filepath = app_data.get("file_path_name", "")
    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            dpg.set_value("input_string", f.read())
        dpg.set_value("status_text", f"Loaded from {filepath}")


# ============================================================================
# Constraint UI Builder
# ============================================================================

def _add_constraint(parent, cid, label, help_text, build_options_fn, height=100, toggle_attr=None):
    """Add a constraint checkbox + collapsible options section.

    build_options_fn(options_tag): called to populate the options child window.
    toggle_attr: if set, toggling the checkbox directly sets this ConstraintState attribute.
    """
    options_tag = f"{cid}_options"
    check_tag = f"{cid}_check"

    def toggle(s, a, u):
        dpg.show_item(options_tag) if a else dpg.hide_item(options_tag)
        if toggle_attr:
            setattr(constraints, toggle_attr, a)

    dpg.add_checkbox(tag=check_tag, label=label, callback=toggle, parent=parent)
    _help(help_text)

    with dpg.child_window(tag=options_tag, show=False, height=height, width=-1, parent=parent):
        build_options_fn(options_tag)


def _make_apply_callback(cid, label, load_fn):
    """Create an 'Apply' callback that sets constraint state and updates filter display."""
    applied_tag = f"{cid}_applied"
    filter_tag = f"{cid}_filter"
    _dynamic_tags.extend([applied_tag, filter_tag])

    def cb(*args):
        display = load_fn()
        if dpg.does_item_exist(applied_tag):
            dpg.set_value(filter_tag, f"{label}: {display}")
        else:
            dpg.add_text(tag=applied_tag, default_value=f"{label} applied!",
                         parent=f"{cid}_options")
            dpg.add_text(tag=filter_tag, default_value=f"{label}: {display}",
                         parent="active_filters")
        refresh_token_popup()

    return cb


def build_all_constraints(tab_bar_tag):
    """Build the complete constraint UI organized into tabs."""

    # --- Lexical Tab ---
    with dpg.tab(label="Lexical", parent=tab_bar_tag) as lexical:

        def _lipogram_opts(p):
            dpg.add_text("Add banned letters or strings separated by spaces", parent=p)
            dpg.add_input_text(tag="lipogram_input", label="Banned Strings", width=-100, parent=p)
            def load():
                raw = dpg.get_value("lipogram_input")
                constraints.lipogram_banned_list = raw.split() if raw.strip() else []
                return " ".join(constraints.lipogram_banned_list)
            dpg.add_button(label="Apply", callback=_make_apply_callback("lipogram", "All Strings Banned", load), parent=p)
        _add_constraint(lexical, "lipogram", "All Strings Banned",
                        "A lipogram avoids particular letters or strings.", _lipogram_opts)

        def _weak_lipogram_opts(p):
            dpg.add_text("Add banned letters or strings separated by spaces", parent=p)
            dpg.add_input_text(tag="weak_lipogram_input", label="Banned Strings", width=-100, parent=p)
            def load():
                raw = dpg.get_value("weak_lipogram_input")
                constraints.weak_lipogram_banned_list = raw.split() if raw.strip() else []
                return " ".join(constraints.weak_lipogram_banned_list)
            dpg.add_button(label="Apply", callback=_make_apply_callback("weak_lipogram", "Any Strings Banned", load), parent=p)
        _add_constraint(lexical, "weak_lipogram", "Any Strings Banned",
                        "A weak lipogram avoids at least one of the specified letters/strings.", _weak_lipogram_opts)

        def _rev_lipogram_opts(p):
            dpg.add_text("Add required letters or strings separated by spaces", parent=p)
            dpg.add_input_text(tag="rev_lipogram_input", label="Required Strings", width=-100, parent=p)
            def load():
                raw = dpg.get_value("rev_lipogram_input")
                constraints.reverse_lipogram_required_list = raw.split() if raw.strip() else []
                return " ".join(constraints.reverse_lipogram_required_list)
            dpg.add_button(label="Apply", callback=_make_apply_callback("rev_lipogram", "All Strings Required", load), parent=p)
        _add_constraint(lexical, "rev_lipogram", "All Strings Required",
                        "A reverse lipogram forces particular letters or strings to appear.", _rev_lipogram_opts)

        def _weak_rev_opts(p):
            dpg.add_text("Add required letters or strings separated by spaces", parent=p)
            dpg.add_input_text(tag="weak_rev_input", label="Required Strings", width=-100, parent=p)
            def load():
                raw = dpg.get_value("weak_rev_input")
                constraints.weak_reverse_lipogram_required_list = raw.split() if raw.strip() else []
                return " ".join(constraints.weak_reverse_lipogram_required_list)
            dpg.add_button(label="Apply", callback=_make_apply_callback("weak_rev", "Any Strings Required", load), parent=p)
        _add_constraint(lexical, "weak_rev", "Any Strings Required",
                        "Forces at least one of the specified letters/strings to appear.", _weak_rev_opts)

        def _str_pos_opts(p):
            dpg.add_text("Characters separated by spaces", parent=p)
            dpg.add_input_text(tag="pos_chars_input", label="Characters", width=-100, parent=p)
            dpg.add_text("Corresponding position indexes separated by spaces", parent=p)
            dpg.add_input_text(tag="pos_index_input", label="Positions", width=-100, parent=p)
            def load():
                chars_raw = dpg.get_value("pos_chars_input")
                idx_raw = dpg.get_value("pos_index_input")
                constraints.string_in_position_list = chars_raw.split() if chars_raw.strip() else []
                try:
                    constraints.string_in_position_index_list = [int(i) for i in idx_raw.split()] if idx_raw.strip() else []
                except ValueError:
                    constraints.string_in_position_index_list = []
                return f"chars={constraints.string_in_position_list} at={constraints.string_in_position_index_list}"
            dpg.add_button(label="Apply", callback=_make_apply_callback("str_pos", "String In Position", load), parent=p)
        _add_constraint(lexical, "str_pos", "String In Position",
                        "Force particular characters at specific positions.\nRecommended: combine with whitespace stripping.",
                        _str_pos_opts, height=160)

        def _starts_opts(p):
            dpg.add_text("String that tokens must start with", parent=p)
            dpg.add_input_text(tag="starts_input", label="Prefix", width=-100, parent=p)
            def load():
                constraints.starts_with_string = dpg.get_value("starts_input")
                return constraints.starts_with_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("starts", "Starts With", load), parent=p)
        _add_constraint(lexical, "starts", "String Starts With",
                        "Guarantee tokens start with a particular string.", _starts_opts)

        def _ends_opts(p):
            dpg.add_text("String that tokens must end with", parent=p)
            dpg.add_input_text(tag="ends_input", label="Suffix", width=-100, parent=p)
            def load():
                constraints.ends_with_string = dpg.get_value("ends_input")
                return constraints.ends_with_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("ends", "Ends With", load), parent=p)
        _add_constraint(lexical, "ends", "String Ends With",
                        "Guarantee tokens end with a particular string.", _ends_opts)

        def _len_eq_opts(p):
            dpg.add_text("Exact length tokens must match", parent=p)
            dpg.add_input_int(tag="len_eq_input", label="Length", parent=p)
            def load():
                constraints.constrained_length = dpg.get_value("len_eq_input")
                return str(constraints.constrained_length)
            dpg.add_button(label="Apply", callback=_make_apply_callback("len_eq", "Length Equal To", load), parent=p)
        _add_constraint(lexical, "len_eq", "String Length Equal To",
                        "Guarantee tokens are exactly this length.\nRecommended: combine with whitespace stripping.",
                        _len_eq_opts)

        def _len_gt_opts(p):
            dpg.add_text("Minimum length (exclusive)", parent=p)
            dpg.add_input_int(tag="len_gt_input", label="Min Length", parent=p)
            def load():
                constraints.constrained_gt_length = dpg.get_value("len_gt_input")
                return str(constraints.constrained_gt_length)
            dpg.add_button(label="Apply", callback=_make_apply_callback("len_gt", "Length Greater Than", load), parent=p)
        _add_constraint(lexical, "len_gt", "String Length Greater Than",
                        "Guarantee tokens are longer than this length.", _len_gt_opts)

        def _len_lt_opts(p):
            dpg.add_text("Maximum length (exclusive)", parent=p)
            dpg.add_input_int(tag="len_lt_input", label="Max Length", parent=p)
            def load():
                constraints.constrained_lt_length = dpg.get_value("len_lt_input")
                return str(constraints.constrained_lt_length)
            dpg.add_button(label="Apply", callback=_make_apply_callback("len_lt", "Length Less Than", load), parent=p)
        _add_constraint(lexical, "len_lt", "String Length Less Than",
                        "Guarantee tokens are shorter than this length.", _len_lt_opts)

    # --- Phonetic Tab ---
    with dpg.tab(label="Phonetic", parent=tab_bar_tag) as phonetic:

        def _phonetic_opts(p):
            dpg.add_text("Word to phonetically match against", parent=p)
            dpg.add_input_text(tag="phonetic_input", label="Match Word", width=-100, parent=p)
            def load():
                constraints.phonetic_matching_string = dpg.get_value("phonetic_input")
                return constraints.phonetic_matching_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("phonetic", "Phonetic Matching", load), parent=p)
        _add_constraint(phonetic, "phonetic", "Phonetic Matching",
                        "Uses the Metaphone algorithm to phonetically match tokens against the given word.",
                        _phonetic_opts)

        def _syllable_opts(p):
            dpg.add_text("Number of syllables required", parent=p)
            dpg.add_input_int(tag="syllable_input", label="Syllables", parent=p)
            def load():
                constraints.syllable_number = dpg.get_value("syllable_input")
                return str(constraints.syllable_number)
            dpg.add_button(label="Apply", callback=_make_apply_callback("syllable", "Syllable Count", load), parent=p)
        _add_constraint(phonetic, "syllable", "Syllable Count",
                        "Return tokens with the specified number of syllables.", _syllable_opts)

        def _meter_opts(p):
            dpg.add_text("Word whose stress pattern (meter) to match", parent=p)
            dpg.add_input_text(tag="meter_input", label="Meter Word", width=-100, parent=p)
            def load():
                constraints.meter_string = dpg.get_value("meter_input")
                return constraints.meter_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("meter", "Meter", load), parent=p)
        _add_constraint(phonetic, "meter", "Meter",
                        "Return tokens matching the stress pattern (meter) of the given word.", _meter_opts)

        def _rhyme_opts(p):
            dpg.add_text("Word to rhyme with", parent=p)
            dpg.add_input_text(tag="rhyme_input", label="Rhyme Word", width=-100, parent=p)
            def load():
                constraints.rhyme_string = dpg.get_value("rhyme_input")
                return constraints.rhyme_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("rhyme", "Rhyme", load), parent=p)
        _add_constraint(phonetic, "rhyme", "Rhyme",
                        "Return tokens that rhyme with the given word.", _rhyme_opts)

    # --- Semantic Tab ---
    with dpg.tab(label="Semantic", parent=tab_bar_tag) as semantic:

        def _semantic_opts(p):
            dpg.add_text("Word to match semantically (uses FastText vectors)", parent=p)
            dpg.add_input_text(tag="semantic_input", label="Word", width=-100, parent=p)
            dpg.add_input_float(tag="semantic_threshold", label="Min Similarity (0-1)", default_value=0.5, parent=p)
            def load():
                constraints.semantic_matching_string = dpg.get_value("semantic_input")
                constraints.semantic_distance_threshold = dpg.get_value("semantic_threshold")
                return f"{constraints.semantic_matching_string} (>= {constraints.semantic_distance_threshold:.2f})"
            dpg.add_button(label="Apply", callback=_make_apply_callback("semantic", "Semantic Matching", load), parent=p)
        _add_constraint(semantic, "semantic", "Semantic Matching",
                        "Return tokens semantically similar to the given word using FastText vectors.",
                        _semantic_opts, height=130)

        def _edit_dist_opts(p):
            dpg.add_text("Word to match using Levenshtein edit distance", parent=p)
            dpg.add_input_text(tag="edit_dist_input", label="Word", width=-100, parent=p)
            dpg.add_input_int(tag="edit_dist_threshold", label="Max Distance", default_value=2, parent=p)
            def load():
                constraints.string_edit_string = dpg.get_value("edit_dist_input")
                constraints.string_edit_distance_threshold = dpg.get_value("edit_dist_threshold")
                return f"{constraints.string_edit_string} (<= {constraints.string_edit_distance_threshold})"
            dpg.add_button(label="Apply", callback=_make_apply_callback("edit_dist", "Edit Distance", load), parent=p)
        _add_constraint(semantic, "edit_dist", "String Edit Distance",
                        "Return tokens within the specified Levenshtein edit distance of the given word.",
                        _edit_dist_opts, height=130)

    # --- Structural Tab ---
    with dpg.tab(label="Structural", parent=tab_bar_tag) as structural:

        def _palindrome_opts(p):
            dpg.add_text("Palindrome constraint is controlled by the checkbox above.", parent=p)
        _add_constraint(structural, "palindrome", "Palindrome",
                        "A palindrome reads the same backward as forward (madam, racecar).",
                        _palindrome_opts, height=40, toggle_attr="palindrome_enabled")

        def _anagram_opts(p):
            dpg.add_text("String that tokens must be anagrams of", parent=p)
            dpg.add_input_text(tag="anagram_input", label="Anagram String", width=-100, parent=p)
            def load():
                constraints.anagram_string = dpg.get_value("anagram_input")
                return constraints.anagram_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("anagram", "Anagram", load), parent=p)
        _add_constraint(structural, "anagram", "Anagram",
                        "An anagram is formed by rearranging all letters of a different string.", _anagram_opts)

        def _partial_anagram_opts(p):
            dpg.add_text("String that tokens must be partial anagrams of", parent=p)
            dpg.add_input_text(tag="partial_anagram_input", label="Source String", width=-100, parent=p)
            def load():
                constraints.partial_anagram_string = dpg.get_value("partial_anagram_input")
                return constraints.partial_anagram_string
            dpg.add_button(label="Apply", callback=_make_apply_callback("partial_anagram", "Partial Anagram", load), parent=p)
        _add_constraint(structural, "partial_anagram", "Partial Anagram",
                        "A partial anagram uses some or all letters of a different string.", _partial_anagram_opts)

        def _isogram_opts(p):
            dpg.add_text("Max times any character may repeat", parent=p)
            dpg.add_input_int(tag="isogram_input", label="Max Repeats", default_value=1, parent=p)
            def load():
                constraints.isogram_count = dpg.get_value("isogram_input")
                return str(constraints.isogram_count)
            dpg.add_button(label="Apply", callback=_make_apply_callback("isogram", "Isogram", load), parent=p)
        _add_constraint(structural, "isogram", "Isogram",
                        "An isogram has no character appearing more than N times.", _isogram_opts)

        def _rev_isogram_opts(p):
            dpg.add_text("Min times every character must repeat", parent=p)
            dpg.add_input_int(tag="rev_isogram_input", label="Min Repeats", default_value=1, parent=p)
            def load():
                constraints.reverse_isogram_count = dpg.get_value("rev_isogram_input")
                return str(constraints.reverse_isogram_count)
            dpg.add_button(label="Apply", callback=_make_apply_callback("rev_isogram", "Reverse Isogram", load), parent=p)
        _add_constraint(structural, "rev_isogram", "Reverse Isogram",
                        "A reverse isogram has every character appearing at least N times.", _rev_isogram_opts)


# ============================================================================
# Application Entry Point
# ============================================================================

def main():
    global fasttext_model

    # Load FastText model for semantic matching
    try:
        model_path = hf_hub_download("osanseviero/fasttext_embedding", "model.bin")
        fasttext_model = fasttext.load_model(model_path)
    except Exception as e:
        print(f"Warning: Could not load FastText model: {e}")
        print("Semantic matching will be unavailable.")

    # --- DearPyGUI Setup ---
    dpg.create_context()
    dpg.configure_app(docking=False, docking_space=False)

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    dpg.create_viewport(title="CTGS - Constrained Text Generation Studio",
                        width=screen_width, height=screen_height)
    dpg.setup_dearpygui()

    # Font (with fallback)
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "OpenSans-VariableFont_wdth,wght.ttf")
    if os.path.exists(font_path):
        with dpg.font_registry():
            font = dpg.add_font(font_path, 16, tag="ttf-font")
        dpg.bind_font(font)

    # Theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (50, 90, 140))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 110, 160))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 135, 190))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 85, 130))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (50, 90, 140))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (70, 120, 170))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (60, 110, 160))
    dpg.bind_theme(global_theme)

    # Plot color themes for bar chart
    with dpg.theme(tag="green_bar_theme"):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (60, 180, 75, 200))
            dpg.add_theme_color(dpg.mvPlotCol_Line, (60, 180, 75, 255))
    with dpg.theme(tag="red_bar_theme"):
        with dpg.theme_component(dpg.mvBarSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (200, 60, 60, 200))
            dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 60, 60, 255))

    # File dialogs
    with dpg.file_dialog(tag="save_dialog", callback=_save_text_callback,
                         show=False, default_filename="output.txt",
                         width=700, height=400):
        dpg.add_file_extension(".txt", color=(150, 255, 150))
        dpg.add_file_extension(".*")
    with dpg.file_dialog(tag="load_dialog", callback=_load_text_callback,
                         show=False, width=700, height=400):
        dpg.add_file_extension(".txt", color=(150, 255, 150))
        dpg.add_file_extension(".*")

    # ===================================================================
    # Single primary window with tabbed layout
    # ===================================================================
    with dpg.window(tag="main_window", label="CTGS", no_close=True):

        # -- Top: always-visible text editor + action buttons --
        dpg.add_input_text(tag="input_string", width=-1, height=200, multiline=True,
                           default_value="Type something here!")
        with dpg.popup(parent="input_string", tag="token_popup"):
            dpg.add_text("Press 'Predict New Tokens' or F1 to see suggestions")

        with dpg.group(horizontal=True):
            dpg.add_button(label="Predict New Tokens (F1)", callback=refresh_token_popup)
            dpg.add_button(label="Generate Tokens (F2)", callback=generate_tokens_callback)
            dpg.add_button(label="Save Text", callback=lambda: dpg.show_item("save_dialog"))
            dpg.add_button(label="Load Text", callback=lambda: dpg.show_item("load_dialog"))
            dpg.add_text(tag="status_text", default_value="Loading model...", color=[120, 200, 120])

        dpg.add_separator()

        # -- Bottom: tabbed sections --
        with dpg.tab_bar(tag="main_tabs"):

            # ============ Settings Tab ============
            with dpg.tab(label="Settings"):
                with dpg.group(horizontal=True):
                    # Left column: Model
                    with dpg.child_window(width=screen_width // 2 - 20, height=-1):
                        dpg.add_text("Model", color=[120, 200, 255])
                        dpg.add_input_text(tag="model_name", width=-1, default_value="EleutherAI/pythia-1b",
                                           label="HuggingFace Model")
                        with dpg.group(horizontal=True):
                            dpg.add_combo(items=["32bit", "16bit", "8bit"], tag="precision",
                                          label="Precision", default_value="32bit", width=120)
                            dpg.add_button(tag="load_model_btn", label="Load Model", callback=load_model_callback)
                            dpg.add_loading_indicator(tag="model_loading_indicator", show=False,
                                                      style=0, radius=2.0, color=(60, 110, 160))
                        dpg.add_text(tag="model_status", default_value="No model loaded", color=[200, 200, 100])

                        dpg.add_separator()
                        dpg.add_text("Sampling", color=[120, 200, 255])
                        dpg.add_slider_float(tag="top_p", label="Top-P",
                                             default_value=0.0, min_value=0.0, max_value=1.0, format="%.2f")
                        _help("Top-p (nucleus) sampling. 0 = disabled.")
                        dpg.add_input_int(tag="top_k", label="Top-K", default_value=0, min_value=0)
                        _help("Top-K sampling. 0 = disabled.")
                        dpg.add_slider_float(tag="temperature", label="Temperature",
                                             default_value=1.0, min_value=0.1, max_value=3.0, format="%.2f")
                        _help("Lower = more deterministic, higher = more random.")
                        dpg.add_input_int(tag="num_tokens_to_generate", label="Tokens to generate",
                                          default_value=20, min_value=1)
                        dpg.add_checkbox(tag="greedy_decoding", label="Greedy decoding")
                        _help("Always select the most likely token.")

                    # Right column: Constraint engine + Display
                    with dpg.child_window(width=-1, height=-1):
                        dpg.add_text("Constraint Engine", color=[120, 200, 255])
                        dpg.add_slider_float(tag="constraint_strength", label="Constraint Strength",
                                             default_value=1.0, min_value=0.0, max_value=1.0, format="%.2f")
                        _help("1.0 = hard filter. Lower = soft penalties. Prevents vocabulary crippling.")
                        dpg.add_checkbox(tag="progressive_relaxation", label="Progressive relaxation", default_value=True)
                        _help("Auto-expands candidate pool when constraints are too strict.")
                        dpg.add_input_int(tag="max_backtracks", label="Max backtracks",
                                          default_value=3, min_value=0, max_value=50)
                        _help("Backtrack this many times on dead ends during generation.")

                        dpg.add_separator()
                        dpg.add_text("Display & Vocabulary", color=[120, 200, 255])
                        dpg.add_input_int(tag="max_display_tokens", label="Max tokens to display",
                                          default_value=200, min_value=0)
                        _help("Max tokens in popup and table. 0 = unlimited.")
                        dpg.add_checkbox(tag="show_all_tokens", label="Show all tokens")
                        _help("Override display limit and show ALL tokens.")
                        dpg.add_input_int(tag="default_vocab_size", label="Default vocabulary size",
                                          default_value=500, min_value=0)
                        _help("Default Top-K when both are 0. Set to 0 for full vocab (slow).")

                        dpg.add_separator()
                        dpg.add_text("Active Filters:", color=[120, 200, 255])
                        with dpg.group(tag="active_filters"):
                            pass

            # ============ Filters Tab ============
            with dpg.tab(label="Filters"):
                with dpg.tab_bar(tag="filter_tabs"):
                    build_all_constraints("filter_tabs")
                dpg.add_separator()
                dpg.add_button(tag="reset_filters_btn", label="Reset All Filters",
                               callback=turn_filters_off_callback)

            # ============ Transforms Tab ============
            with dpg.tab(label="Transforms"):
                dpg.add_text("Text transforms applied to all tokens BEFORE constraint filtering.",
                             color=[180, 180, 180])
                dpg.add_text("Using transforms increases the vocabulary that survives filtering.")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=screen_width // 3, height=-1):
                        dpg.add_text("Case", color=[120, 200, 255])
                        dpg.add_checkbox(tag="upper_case_transform", label="Uppercase")
                        dpg.add_checkbox(tag="lower_case_transform", label="Lowercase")
                        dpg.add_checkbox(tag="capitalize_first_letter_transform", label="Capitalize first letter")
                    with dpg.child_window(width=screen_width // 3, height=-1):
                        dpg.add_text("Whitespace", color=[120, 200, 255])
                        dpg.add_checkbox(tag="replace_spaces", label="Remove spaces")
                        dpg.add_checkbox(tag="lstrip_transform", label="Left strip")
                        dpg.add_checkbox(tag="rstrip_transform", label="Right strip")
                        dpg.add_checkbox(tag="strip_transform", label="Full strip")
                    with dpg.child_window(width=-1, height=-1):
                        dpg.add_text("Character Filters", color=[120, 200, 255])
                        dpg.add_checkbox(tag="alphanumeric_transform", label="Alphanumeric only")
                        dpg.add_checkbox(tag="alpha_transform", label="Alphabetic only")
                        dpg.add_checkbox(tag="digit_transform", label="Digits only")
                        dpg.add_checkbox(tag="ascii_transform", label="ASCII only")
                        dpg.add_checkbox(tag="filter_blank_outputs", label="Filter blanks")

            # ============ Analysis Tab ============
            with dpg.tab(label="Analysis"):
                with dpg.group(horizontal=True):
                    dpg.add_input_int(tag="chart_top_n", label="Top N bars",
                                      default_value=30, min_value=5, max_value=500, width=150)
                    dpg.add_button(label="Refresh Charts", callback=_refresh_analysis)
                    dpg.add_text(tag="constraint_summary_text",
                                 default_value="Predict tokens to see analysis", color=[180, 180, 180])

                with dpg.tab_bar(tag="analysis_tabs"):

                    # -- Probability chart sub-tab --
                    with dpg.tab(label="Probability Chart"):
                        with dpg.plot(tag="prob_bar_plot",
                                      label="Token Probabilities (green=pass, red=fail)",
                                      height=-1, width=-1):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, tag="prob_bar_x", label="Token")
                            dpg.add_plot_axis(dpg.mvYAxis, tag="prob_bar_y", label="Probability")

                    # -- Constraint impact sub-tab --
                    with dpg.tab(label="Constraint Impact"):
                        with dpg.group(horizontal=True):
                            with dpg.child_window(width=350, height=-1):
                                with dpg.plot(tag="constraint_pie_plot", label="Pass vs Fail",
                                              height=300, width=-1, equal_aspects=True,
                                              no_mouse_pos=True):
                                    dpg.add_plot_axis(dpg.mvXAxis, tag="pie_x",
                                                     no_tick_labels=True, no_tick_marks=True, no_gridlines=True)
                                    dpg.add_plot_axis(dpg.mvYAxis, tag="pie_y",
                                                     no_tick_labels=True, no_tick_marks=True, no_gridlines=True)
                            with dpg.child_window(width=-1, height=-1):
                                dpg.add_text("Per-Constraint Breakdown:", color=[200, 200, 100])
                                with dpg.group(tag="constraint_breakdown_group"):
                                    dpg.add_text("Predict tokens to see breakdown", color=[180, 180, 180])

                    # -- Token table sub-tab --
                    with dpg.tab(label="Token Table"):
                        dpg.add_input_text(tag="token_search_input", label="Search",
                                           width=300, callback=_refresh_token_table, on_enter=False)
                        with dpg.table(tag="token_table", header_row=True,
                                       resizable=True, borders_innerH=True, borders_outerH=True,
                                       borders_innerV=True, borders_outerV=True,
                                       scrollY=True, height=-1, row_background=True):
                            dpg.add_table_column(label="Rank", width_fixed=True, init_width_or_weight=60)
                            dpg.add_table_column(label="Token")
                            dpg.add_table_column(label="Probability")
                            dpg.add_table_column(label="Passes", width_fixed=True, init_width_or_weight=70)

                    # -- Generation history sub-tab --
                    with dpg.tab(label="Generation History"):
                        with dpg.plot(tag="history_plot", label="Token Confidence Over Generation",
                                      height=-1, width=-1):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, tag="history_x", label="Step")
                            dpg.add_plot_axis(dpg.mvYAxis, tag="history_y", label="Probability")

            # ============ Help Tab ============
            with dpg.tab(label="Help"):
                dpg.add_text("Quick Start Guide", color=[120, 200, 255])
                dpg.add_separator()
                dpg.add_text("First run downloads the model + FastText from HuggingFace (may take minutes).", bullet=True)
                dpg.add_text("Right-click in the text box to see AI-suggested continuations.", bullet=True)
                dpg.add_text("F1 = Predict tokens (populates right-click menu).", bullet=True)
                dpg.add_text("F2 = Insert a single token directly.", bullet=True)
                dpg.add_text("Enable 'Full strip' + 'Filter blanks' in Transforms for best results.", bullet=True)
                dpg.add_text("Use Constraint Strength < 1.0 for soft penalties instead of hard bans.", bullet=True)
                dpg.add_text("Progressive Relaxation auto-expands the pool when constraints are strict.", bullet=True)
                dpg.add_text("Backtracking recovers from dead ends during multi-token generation.", bullet=True)
                dpg.add_separator()
                dpg.add_text("Hover over green (?) icons for contextual help on any setting.")

    dpg.set_primary_window("main_window", True)

    # --- Key Bindings ---
    with dpg.handler_registry():
        dpg.add_key_press_handler(key=dpg.mvKey_F1, callback=refresh_token_popup)
        dpg.add_key_press_handler(key=dpg.mvKey_F2, callback=tab_key_generate_callback)

    # --- Start ---
    dpg.set_global_font_scale(1.0)
    dpg.show_viewport()

    # Load default model in background
    load_model_callback(model_name_override="EleutherAI/pythia-1b")

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
