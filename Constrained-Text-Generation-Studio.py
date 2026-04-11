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


constraints = ConstraintState()
model_state = ModelState()
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
            raw_logits = outputs.logits[:, -1, :]

            temperature = dpg.get_value("temperature")
            if temperature > 0 and temperature != 1.0:
                raw_logits = raw_logits / temperature

            top_p = dpg.get_value("top_p")
            top_k = dpg.get_value("top_k")
            strength = dpg.get_value("constraint_strength")
            use_relaxation = dpg.get_value("progressive_relaxation")
            has_constraints = constraints.has_active_constraints()

            if use_relaxation and has_constraints:
                levels = list(_relaxation_levels(top_k, top_p))
            else:
                levels = [(top_k, top_p)]

            for eff_k, eff_p in levels:
                logits = apply_top_k_top_p(raw_logits, top_k=eff_k, top_p=eff_p)
                probs = F.softmax(logits.float(), dim=-1).sort(descending=True)

                word_list = []
                decoded = model_state.tokenizer.batch_decode(probs[1][0])
                filter_blanks = dpg.get_value("filter_blank_outputs")

                for i, raw_text in enumerate(decoded):
                    prob = probs[0][0][i].item()
                    if prob <= 0:
                        break  # remaining tokens have zero probability

                    text = apply_text_transforms(raw_text)
                    if filter_blanks and text == "":
                        continue
                    word_list.append((text, prob))

                if not has_constraints:
                    return word_list

                # Apply constraints with strength control
                if strength >= 1.0:
                    filtered = [w for w in word_list if passes_all_constraints(w, constraints)]
                elif strength <= 0.0:
                    filtered = word_list
                else:
                    passing = []
                    failing = []
                    for w in word_list:
                        (passing if passes_all_constraints(w, constraints) else failing).append(w)
                    penalty_factor = 1.0 - strength
                    filtered = passing + [(t, p * penalty_factor) for t, p in failing]
                    filtered.sort(key=lambda x: x[1], reverse=True)

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
        for text, prob in candidates[:200]:
            dpg.add_selectable(
                label=f"{text}  ({prob:.2%})",
                user_data=text,
                callback=_add_word_callback,
                parent="token_popup"
            )
        dpg.set_value("status_text", f"Ready - {len(candidates)} candidates")
    else:
        dpg.add_text("No results with current constraints", parent="token_popup")
        dpg.set_value("status_text", "No tokens match current constraints")


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

        history.append((text, remaining))
        text += chosen[0]
        tokens_generated += 1
        dpg.set_value("input_string", text)
        dpg.set_value("status_text", f"Generated {tokens_generated}/{num_tokens} tokens")

    dpg.set_value("status_text", f"Done! Generated {num_tokens} tokens")


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
            kwargs = {"device_map": "auto"}
            if precision == "8bit":
                kwargs["load_in_8bit"] = True
            elif precision == "16bit":
                kwargs["torch_dtype"] = torch.float16
            model_state.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            model_state.name = model_name

            device = "GPU" if torch.cuda.is_available() else "CPU"
            dpg.set_value("model_status", f"{model_name} loaded on {device} ({precision})")
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
    dpg.configure_app(docking=True, docking_space=True)

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

    # Layout sizes
    w = screen_width // 2
    h = int(screen_height // 2.5)

    # --- Main Text Window ---
    with dpg.window(tag="main_window", label="CTGS - Constrained Text Generation Studio",
                    no_close=True, width=w, height=h, pos=(0, 0)):
        dpg.add_text("Right-click in the text box for AI-suggested continuations with constraints applied")
        dpg.add_input_text(tag="input_string", width=-1, height=300, multiline=True,
                           default_value="Type something here!")

        with dpg.popup(parent="input_string", tag="token_popup"):
            dpg.add_text("Press 'Predict New Tokens' or F1 to see suggestions")

        dpg.add_text(tag="status_text", default_value="Loading model...", color=[120, 200, 120])
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(label="Predict New Tokens (F1)", callback=refresh_token_popup)
            dpg.add_button(label="Generate Tokens (F2)", callback=generate_tokens_callback)

        dpg.add_separator()
        dpg.add_text("Sampling Settings")
        dpg.add_slider_float(tag="top_p", label="Top-P (nucleus sampling)",
                             default_value=0.0, min_value=0.0, max_value=1.0, format="%.2f")
        _help("Top-p sampling chooses from the smallest set of words whose cumulative probability exceeds p. 0 = disabled.")
        dpg.add_input_int(tag="top_k", label="Top-K sampling", default_value=0, min_value=0)
        _help("Top-K keeps only the K most likely next words. 0 = disabled.")
        dpg.add_slider_float(tag="temperature", label="Temperature",
                             default_value=1.0, min_value=0.1, max_value=3.0, format="%.2f")
        _help("Lower = more deterministic, higher = more random.")
        dpg.add_input_int(tag="num_tokens_to_generate", label="Tokens to generate",
                          default_value=20, min_value=1)
        dpg.add_checkbox(tag="greedy_decoding", label="Greedy decoding")
        _help("Always select the most likely token (overrides top-p/top-k).")

        dpg.add_separator()
        dpg.add_text("Constraint Engine Settings")
        dpg.add_slider_float(tag="constraint_strength", label="Constraint Strength",
                             default_value=1.0, min_value=0.0, max_value=1.0, format="%.2f")
        _help("1.0 = hard filter (pass/fail). Lower values apply soft penalties instead, "
              "allowing near-misses with reduced probability. Prevents vocabulary crippling.")
        dpg.add_checkbox(tag="progressive_relaxation", label="Progressive relaxation", default_value=True)
        _help("When enabled, automatically expands the candidate pool if constraints filter out all tokens "
              "at the current top-k/top-p level.")
        dpg.add_input_int(tag="max_backtracks", label="Max backtracks",
                          default_value=3, min_value=0, max_value=50)
        _help("During multi-token generation, backtrack this many times on dead ends to try alternative tokens.")

        dpg.add_separator()
        dpg.add_text("Active Filters:")
        with dpg.group(tag="active_filters"):
            pass

    # --- Model Settings Window ---
    with dpg.window(tag="model_window", label="Model Settings",
                    no_close=True, width=w, height=h, pos=(w, h)):
        dpg.add_text("Load a causal language model from HuggingFace for text generation")
        _help("Make sure torch.cuda.is_available() returns True for GPU acceleration!")
        dpg.add_input_text(tag="model_name", width=-100, default_value="EleutherAI/pythia-1b",
                           label="HuggingFace Model")
        dpg.add_combo(items=["32bit", "16bit", "8bit"], tag="precision",
                      label="Precision", default_value="32bit")
        dpg.add_button(tag="load_model_btn", label="Load Model", callback=load_model_callback)
        _help("Downloads the model from HuggingFace. Check terminal for download progress.")
        dpg.add_loading_indicator(tag="model_loading_indicator", show=False,
                                  style=0, radius=2.0, color=(60, 110, 160))
        dpg.add_text(tag="model_status", default_value="No model loaded", color=[200, 200, 100])

    # --- Filter Options Window ---
    with dpg.window(tag="filter_window", label="Filters",
                    no_close=True, width=w, height=h, pos=(w, 0)):
        dpg.add_text("Select and configure constraint filters")

        with dpg.tab_bar(tag="filter_tabs"):
            build_all_constraints("filter_tabs")

        dpg.add_separator()
        dpg.add_button(tag="reset_filters_btn", label="Reset All Filters",
                       callback=turn_filters_off_callback)

    # --- Text Transforms Window ---
    with dpg.window(tag="transforms_window", label="Text Transforms",
                    no_close=True, width=w, height=h, pos=(0, h)):
        dpg.add_text("Transforms applied to all tokens BEFORE constraint filtering")
        dpg.add_text("Using transforms increases the vocabulary that survives filtering", color=[180, 180, 180])
        dpg.add_separator()
        dpg.add_checkbox(tag="upper_case_transform", label="Uppercase")
        _help("Force all tokens to UPPERCASE")
        dpg.add_checkbox(tag="lower_case_transform", label="Lowercase")
        _help("Force all tokens to lowercase")
        dpg.add_checkbox(tag="replace_spaces", label="Remove spaces")
        _help("Remove all spaces from tokens")
        dpg.add_checkbox(tag="lstrip_transform", label="Left strip")
        _help("Strip whitespace from the left side")
        dpg.add_checkbox(tag="rstrip_transform", label="Right strip")
        _help("Strip whitespace from the right side")
        dpg.add_checkbox(tag="strip_transform", label="Full strip")
        _help("Strip whitespace from both sides")
        dpg.add_checkbox(tag="capitalize_first_letter_transform", label="Capitalize first letter")
        _help("Capitalize the first letter of each token")
        dpg.add_checkbox(tag="alphanumeric_transform", label="Alphanumeric only")
        _help("Keep only letters and numbers")
        dpg.add_checkbox(tag="alpha_transform", label="Alphabetic only")
        _help("Keep only letters")
        dpg.add_checkbox(tag="digit_transform", label="Digits only")
        _help("Keep only numbers")
        dpg.add_checkbox(tag="ascii_transform", label="ASCII only")
        _help("Filter out non-ASCII/unicode characters")
        dpg.add_checkbox(tag="filter_blank_outputs", label="Filter blanks")
        _help("Remove tokens that become empty after transforms")

    # --- Help / Readme Window ---
    readme_pos = (screen_width // 4, screen_height // 4)
    with dpg.window(tag="readme_window", label="Quick Start Guide",
                    no_close=False, width=w, height=h, pos=readme_pos, show=True):
        dpg.add_text("Usage Tips:", color=[120, 200, 255])
        dpg.add_text("The first time you run CTGS, model downloads may take several minutes. "
                      "Check the Model Settings window for loading status.", bullet=True)
        dpg.add_text("Right-click in the text box to see AI-suggested continuations "
                      "with your constraints applied.", bullet=True)
        dpg.add_text("F1 refreshes token predictions (same as 'Predict New Tokens' button).", bullet=True)
        dpg.add_text("F2 inserts a single token directly (same as 'Generate Tokens' with count=1).", bullet=True)
        dpg.add_text("Use Text Transforms to normalize tokens before filtering. "
                      "'Full strip' + 'Filter blanks' is recommended as a baseline.", bullet=True)
        dpg.add_text("The Constraint Strength slider lets you use soft penalties "
                      "instead of hard bans, preventing vocabulary crippling.", bullet=True)
        dpg.add_text("Progressive Relaxation automatically expands the candidate pool "
                      "when constraints are too strict.", bullet=True)
        dpg.add_text("Backtracking lets multi-token generation recover from dead ends "
                      "by trying alternative tokens at previous steps.", bullet=True)
        dpg.add_separator()
        dpg.add_text("Windows can be rearranged, docked, and tiled. "
                      "Hover over green (?) icons for context help.")
        dpg.add_text("Close this window with the X button.")

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
