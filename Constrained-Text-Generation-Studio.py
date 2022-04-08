import dearpygui.dearpygui as dpg
import re
from unittest import result
import string
from collections import Counter
import torch
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoModelForQuestionAnswering,
                          AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2Tokenizer, LogitsProcessor, LogitsProcessorList,
                          pipeline, top_k_top_p_filtering)

from abydos.phonetic import DoubleMetaphone, Eudex
from huggingface_hub import hf_hub_download
import fasttext
from scipy import spatial
import random
import time as time


##TODO: Allow the user to decide to regenerate new predictions with a right click?
##TODO: Use "tab" key to generate new predictions
##TODO: Detect when a model is loaded and display a nice checkmark, and then also a nice thing if GPU is working too! 

starttime = time.time()

fasttext_model = fasttext.load_model(hf_hub_download("osanseviero/fasttext_embedding", "model.bin"))


tokenizer = ""
model = ""
pe = DoubleMetaphone()


def add_and_load_image(image_path, parent=None):
    width, height, channels, data = dpg.load_image(image_path)

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)

    if parent is None:
        return dpg.add_image(texture_id)
    else:
        return dpg.add_image(texture_id, parent=parent)

def _help(message):
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)

def load_model(the_name):
    global tokenizer
    global model
    if the_name == "load_model":
        model_name = dpg.get_value("model_name")
    else:
        model_name = the_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


########--------------------------------------------------------------#### Filters ########--------------------------------------------------------------####

def all_letters_included(word, string_list):
    #any(c in letters for c in word)
    if all(c in word[0] for c in string_list):
        return True
    else:
        return False

def any_letters_included(word, string_list):
    #any(c in letters for c in word)
    if any(c in string_list for c in word[0]):
        return True
    else:
        return False

def all_letters_not_included(word, string_list):
    #any(c in letters for c in word)
    if all(c not in word[0] for c in string_list):
        return True
    else:
        return False

def any_letters_not_included(word, string_list):
    ## "e.g. not words with both B and T in them!"
    if any(c not in string_list for c in word[0]):
        return True
    else:
        return False

def equal_to_length(word, word_length):
    if len(word[0]) == word_length:
        return True
    else:
        return False

def greater_than_length(word, word_length):
    if len(word[0]) > word_length:
        return True
    else:
        return False

def less_than_length(word, word_length):
    if len(word[0]) < word_length:
        return True
    else:
        return False

def ends_with(word, ending_string,  start = None, end = None):
    if word[0].endswith(ending_string, start, end):
        return True
    else:
        return False

def starts_with(word, starting_string, start = None, end = None):
    if word[0].startswith(starting_string, start, end):
        return True
    else:
        return False

def string_in_position(word, a_string_list, position_index_list):
    for idx, string in enumerate(a_string_list):
        try:
            if word[0][position_index_list[idx]] != string:
                return False
        except:
            return False
    return True

def phonetic_matching(word, phonetic_matching_string):
    if pe.encode(word[0]) == pe.encode(phonetic_matching_string):
        return True
    else:
        return False


def semantic_matching(word, semantic_string, semantic_threshold):
    word_word_vector = fasttext_model.get_word_vector(word[0])
    semantic_string_vector = fasttext_model.get_word_vector(semantic_string)
    similarity = 1 - spatial.distance.cosine(word_word_vector, semantic_string_vector)
    if similarity >= semantic_threshold:
        return True
    else: 
        return False


def palindrome(word):
    the_string = word[0]
    if the_string == the_string[::-1]:
        return True
    else:
        return False

def partial_anagram(word, a_string):
    ###Like a full anagram but allows anagramic substrings
    if Counter(word[0]) - Counter(a_string):
         return False
    return True

def full_anagram(word, a_string):
    ### Only true anagrams!
    if Counter(word[0]) == Counter(a_string):
        return True
    else:
        return False

def isogram(word, count = 1):
    ##Allow user to optionally specify list of characters to isolate
    for char in word[0]:
        if word[0].count(char) > count:
            return False
    return True

def reverse_isogram(word, count = 1):
    ##Allow user to optionally specify list of characters to isolate
    for char in word[0]:
        if word[0].count(char) < count:
            return False
    return True


"""

Ideas 

1. Longest repeating and non overlapping substring in a string
2. List of strings to consume from, like a pangram but you specify the alphabet
3. Edit Distance of strings, user supplies string and max distance and it returns strings with that distance. I think this is fast enough to calculate for this to work!

Leetcode stuff like this

Find longest palindroming substring
def countSubstrings(self, s: str) -> int:
        @cache
        def isPalindrome(i, j):
            return i >= j or s[i] == s[j] and isPalindrome(i + 1, j - 1)
        return sum(isPalindrome(i, j) for i in range(len(s)) for j in range(i, len(s)))

"""

########--------------------------------------------------------------####          ########--------------------------------------------------------------####


lipogram_naughty_word_list = []
weak_lipogram_naughty_word_list = []
reverse_lipogram_nice_word_list = []
weak_reverse_lipogram_nice_word_list = []
string_in_positon_list = []
string_in_positon_index_list = []
starts_with_string = ""
ends_with_string = ""
phonetic_matching_string = ""
semantic_matching_string = ""
semantic_distance_threshold = 0.0
constrained_length = 0
constrained_gt_length = 0
constrained_lt_length = 0
palindrome_enabled = False
anagram_string = ""
partial_anagram_string = ""
isogram_count = 0
reverse_isogram_count = 0



def get_next_word_without_e(sequence):
    all_letters_filtered_list = []
    #print(tokenizer)
    input_ids = tokenizer.encode(sequence, return_tensors="pt")
    # get logits of last hidden state
    next_token_candidates_logits = model(input_ids)[0][:, -1, :]
    temperature = dpg.get_value("temperature")
    if temperature != 1.0:
        next_token_candidates_logits = next_token_candidates_logits / temperature
    # filter
    top_p = dpg.get_value("top_p")
    top_k = dpg.get_value("top_k")
    if (top_p > 0 and top_k > 0):
        filtered_next_token_candidates_logits = top_k_top_p_filtering(next_token_candidates_logits, top_k=top_k, top_p=top_p)
    elif top_p > 0:
        filtered_next_token_candidates_logits = top_k_top_p_filtering(next_token_candidates_logits, top_p=top_p)
    elif top_k > 0:
        filtered_next_token_candidates_logits = top_k_top_p_filtering(next_token_candidates_logits, top_k=top_k)
    else:
        filtered_next_token_candidates_logits = next_token_candidates_logits
    # sample and get a probability distribution
    probs = F.softmax(filtered_next_token_candidates_logits, dim=-1).sort(descending = True)
    #next_token_candidates = torch.multinomial(probs, num_samples=number_of_tokens_to_sample) ## 10000 random samples
    #print(next_token_candidates)
    word_list = []
    #print(probs[0][0][0].item())
        #print(probs[1])## the indicies, probs[0] is the probabilities
    for iter, candidate in enumerate(probs[1][0]):
        probability = probs[0][0][iter].item()
        resulting_string = tokenizer.decode(candidate) #skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ##TODO: Consider implementing transforms inspired by stuff in the itertools/moreitertools libraries
        if dpg.get_value("upper_case_transform"):
            resulting_string = resulting_string.upper()
        if dpg.get_value("lower_case_transform"):
            resulting_string = resulting_string.lower()
        if dpg.get_value("replace_spaces"):
            resulting_string = resulting_string.replace(' ', '')
        if dpg.get_value("lstrip_transform"):
            resulting_string = resulting_string.lstrip()
        if dpg.get_value("rstrip_transform"):
            resulting_string = resulting_string.rstrip()
        if dpg.get_value("strip_transform"):
            resulting_string = resulting_string.strip()
        if dpg.get_value("capitalize_first_letter_transform"):
            resulting_string = resulting_string.capitalize()
        if dpg.get_value("alpha_numaric_transform"):
            resulting_string = ''.join(ch for ch in resulting_string if ch.isalnum())
        if dpg.get_value("alpha_transform"):
            resulting_string = ''.join(ch for ch in resulting_string if ch.isalpha())
        if dpg.get_value("digit_transform"):
            resulting_string = ''.join(ch for ch in resulting_string if ch.isdigit())
        if dpg.get_value("ascii_transform"):
            resulting_string = ''.join(ch for ch in resulting_string if ch.isascii())
        if dpg.get_value("filter_blank_outputs"):
            if resulting_string == "":
                continue
        word_list.append((resulting_string, probability))

    #all_letters_filtered_list = [word for word in word_list if all_letters_not_included(word=word, string_list = lipogram_naughty_word_list)]

    for word in word_list:
        return_word = True
        if len(lipogram_naughty_word_list) > 0:
            if not all_letters_not_included(word=word, string_list = lipogram_naughty_word_list):
                return_word = False
        if len(weak_lipogram_naughty_word_list) > 0:
            if not any_letters_not_included(word=word, string_list = weak_lipogram_naughty_word_list):
                return_word = False
        if len(reverse_lipogram_nice_word_list) > 0:
            if not all_letters_included(word=word, string_list = reverse_lipogram_nice_word_list):
                return_word = False 
        if len(weak_reverse_lipogram_nice_word_list) > 0:
            if not any_letters_included(word=word, string_list = weak_reverse_lipogram_nice_word_list):
                return_word = False
        if len(string_in_positon_list) > 0:
            if not string_in_position(word=word, a_string_list=string_in_positon_list, position_index_list=string_in_positon_index_list):
                return_word = False
        if len(starts_with_string) > 0:
            if not starts_with(word=word, starting_string = starts_with_string):
                return_word = False
        if len(ends_with_string) > 0:
            if not ends_with(word=word, ending_string = ends_with_string):
                return_word = False
        if constrained_length > 0:
            if not equal_to_length(word=word, word_length=constrained_length):
                return_word = False
        if constrained_gt_length > 0:
            if not greater_than_length(word=word, word_length=constrained_gt_length):
                return_word = False
        if constrained_lt_length > 0:
            if not less_than_length(word=word, word_length=constrained_lt_length):
                return_word = False
        if palindrome_enabled == True:
            if not palindrome(word=word):
                return_word = False
        if len(phonetic_matching_string) > 0:
            if not phonetic_matching(word=word, phonetic_matching_string=phonetic_matching_string):
                return_word = False
        if len(semantic_matching_string) > 0:
            if not semantic_matching(word=word, semantic_string = semantic_matching_string, semantic_threshold = semantic_distance_threshold):
                return_word = False
        if len(anagram_string) > 0:
            if not full_anagram(word=word, a_string = anagram_string):
                return_word = False
        if len(partial_anagram_string) > 0:
            if not partial_anagram(word=word, a_string = partial_anagram_string):
                return_word = False
        if isogram_count >= 1:
            if not isogram(word = word, count = isogram_count):
                return_word = False
        if reverse_isogram_count >= 1:
            if not reverse_isogram(word = word, count = reverse_isogram_count):
                return_word = False
        if return_word == True:
            all_letters_filtered_list.append(word)
        

    #all_letters_filtered_list = [word for word in word_list if all_letters_not_included(word=word, starting_string= "EN")]
    #list(filter(all_letters_included, word_list))
    #print(probs)
    #print(all_letters_filtered_list[0:50])
    #print(probs)


                
    return all_letters_filtered_list




def generate_tokens_callback():
    number_of_tokens = dpg.get_value("num_tokens_to_generate")
    i = 0
    while i <= number_of_tokens:
        string_input = dpg.get_value("input_string")
        generated_output = get_next_word_without_e(string_input)
        probability_weights = list(zip(*generated_output))[1]
        returned_word = random.choices(generated_output, weights = probability_weights, k = 1)[0][0]
        print(returned_word)
        new_string = string_input + returned_word
        new_value = dpg.set_value("input_string", new_string)
        i = i+1

def add_generated_word_callback(sender, app_data, user_data):
    current_value = dpg.get_value("input_string")
    new_string = current_value + str(user_data)
    new_value = dpg.set_value("input_string", new_string)
    edit_string_callback()

def edit_string_callback():
    string_input = dpg.get_value("input_string")
    returned_words = get_next_word_without_e(string_input)
    #print(returned_words)
    with dpg.popup(parent = "input_string"):
        dpg.add_text("Options")
        dpg.add_separator()
        if len(returned_words) >= 1:
            for word in returned_words:
                dpg.add_selectable(label=word, user_data = word[0], callback = add_generated_word_callback)
        else:
            dpg.add_text("No results with the current filters")
            #print(dpg.get_value(word))
            #print(dpg.get_value("yum"))
    #dpg.log_debug(value)





def lipogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Lipogram Options")
    else:
        dpg.hide_item("Lipogram Options")

def load_naughty_strings_callback():
    global lipogram_naughty_word_list
    string_input = dpg.get_value("lipogram_word_list")
    lipogram_naughty_word_list = string_input.split(" ")
    if not dpg.get_value("naughty_applied"):
        dpg.add_text(tag = "naughty_applied", default_value= "Naughty Strings Applied!", parent = "Lipogram Options")
        dpg.add_text(tag = "naughty_filter" , default_value = "Naughty Strings Filter: " + string_input, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "naughty_filter", value = "Naughty Strings Filter: " + string_input)
    edit_string_callback()

def weak_lipogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Weak Lipogram Options")
    else:
        dpg.hide_item("Weak Lipogram Options")

def load_weak_naughty_strings_callback():
    global weak_lipogram_naughty_word_list
    string_input = dpg.get_value("weak_lipogram_word_list")
    weak_lipogram_naughty_word_list = string_input.split(" ")
    if not dpg.get_value("weak_naughty_applied"):
        dpg.add_text(tag = "weak_naughty_applied", default_value= "Weak Naughty Strings Applied!", parent = "Weak Lipogram Options")
        dpg.add_text(tag = "weak_naughty_filter" , default_value = "Weak Naughty Strings Filter: " + string_input, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "weak_naughty_filter", value = "Weak Naughty Strings Filter: " + string_input)
    edit_string_callback()

def reverse_lipogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Reverse Lipogram Options")
    else:
        dpg.hide_item("Reverse Lipogram Options")

def load_reverse_naughty_strings_callback():
    global reverse_lipogram_nice_word_list
    string_input = dpg.get_value("reverse_lipogram_word_list")
    reverse_lipogram_nice_word_list = string_input.split(" ")
    if not dpg.get_value("reverse_nice_applied"):
        dpg.add_text(tag = "reverse_nice_applied", default_value= "Nice Strings Applied!", parent = "Reverse Lipogram Options")
        dpg.add_text(tag = "reverse_nice_filter" , default_value = "Nice Strings Filter: " + string_input, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "reverse_nice_filter", value = "Nice Strings Filter: " + string_input)
    edit_string_callback()

def weak_reverse_lipogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Weak Reverse Lipogram Options")
    else:
        dpg.hide_item("Weak Reverse Lipogram Options")

def load_weak_reverse_naughty_strings_callback():
    global weak_reverse_lipogram_nice_word_list
    string_input = dpg.get_value("weak_reverse_lipogram_word_list")
    weak_reverse_lipogram_nice_word_list = string_input.split(" ")
    if not dpg.get_value("weak_reverse_nice_applied"):
        dpg.add_text(tag = "weak_reverse_nice_applied", default_value= "Weak Nice Strings Applied!", parent = "Weak Reverse Lipogram Options")
        dpg.add_text(tag = "weak_reverse_nice_filter" , default_value = "Weak Nice Strings Filter: " + string_input, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "weak_reverse_nice_filter", value = "Weak Nice Strings Filter: " + string_input)
    edit_string_callback()

def string_position_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Letter Position Options")
    else:
        dpg.hide_item("Letter Position Options")

def load_string_positon_callback():
    global string_in_positon_list
    global string_in_positon_index_list
    string_input = dpg.get_value("string_for_position")
    int_input = dpg.get_value("string_position_int")
    string_in_positon_list = string_input.split(" ")
    string_in_positon_index_list = int_input.split(" ")
    string_in_positon_index_list = [int(i) for i in string_in_positon_index_list]
    if not dpg.get_value("string_postion_applied"):
        dpg.add_text(tag = "string_postion_applied", default_value= "Strings in Position Applied!", parent = "Letter Position Options")
        dpg.add_text(tag = "string_postion_filter" , default_value = "Strings in Position Applied:  " + string_input + " " + str(int_input), parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_postion_filter", value = "Strings in Position Filter: " + string_input + " " + str(int_input))
    edit_string_callback()



def string_starts_with_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Starting String Options")
    else:
        dpg.hide_item("Starting String Options")

def load_string_starts_with_callback():
    global starts_with_string
    starts_with_string = dpg.get_value("string_start_word")
    if not dpg.get_value("string_starts_with_applied"):
        dpg.add_text(tag = "string_starts_with_applied", default_value= "Starting String Applied!", parent = "Starting String Options")
        dpg.add_text(tag = "string_starts_with_filter" , default_value = "Starting String Applied!  " + starts_with_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_starts_with_filter", value = "Starting String Filter: " + starts_with_string)
    edit_string_callback()

def string_ends_with_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Ending String Options")
    else:
        dpg.hide_item("Ending String Options")

def load_string_ends_with_callback():
    global ends_with_string
    ends_with_string = dpg.get_value("string_end_word")
    if not dpg.get_value("string_ends_with_applied"):
        dpg.add_text(tag = "string_ends_with_applied", default_value= "Ending String Applied!", parent = "Ending String Options")
        dpg.add_text(tag = "string_ends_with_filter" , default_value = "Ending String Applied!  " + ends_with_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_ends_with_filter", value = "Ending String Filter: " + ends_with_string)
    edit_string_callback()

def string_length_constrained_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Length Constrained Options")
    else:
        dpg.hide_item("Length Constrained Options")

def load_string_length_constrained_callback():
    global constrained_length
    constrained_length = dpg.get_value("length_constrained_number")
    constrained_length_str = str(constrained_length)
    if not dpg.get_value("string_length_constrained_applied"):
        dpg.add_text(tag = "string_length_constrained_applied", default_value= "String Length Constraint Applied!", parent = "Length Constrained Options")
        dpg.add_text(tag = "string_length_constrained_filter" , default_value = "String Length Constraint Applied!  " + constrained_length_str, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_length_constrained_filter", value = "String Length Constraint Filter: " + constrained_length_str)
    edit_string_callback()

def string_length_gt_constrained_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Length Greater Than Options")
    else:
        dpg.hide_item("Length Greater Than Options")

def load_string_length_gt_constrained_callback():
    global constrained_gt_length
    constrained_gt_length = dpg.get_value("length_gt_constrained_number")
    constrained_gt_length_str = str(constrained_gt_length)
    if not dpg.get_value("string_length_gt_constrained_applied"):
        dpg.add_text(tag = "string_length_gt_constrained_applied", default_value= "String Length Greater Than Constraint Applied!", parent = "Length Greater Than Options")
        dpg.add_text(tag = "string_length_gt_constrained_filter" , default_value = "String Length Greater Than Constraint Applied!  " + constrained_gt_length_str, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_length_gt_constrained_filter", value = "String Length Greater Than Constraint Filter: " + constrained_gt_length_str)
    edit_string_callback()


def string_length_lt_constrained_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Length Lesser Than Options")
    else:
        dpg.hide_item("Length Lesser Than Options")

def load_string_length_lt_constrained_callback():
    global constrained_lt_length
    constrained_lt_length = dpg.get_value("length_lt_constrained_number")
    constrained_lt_length_str = str(constrained_lt_length)
    if not dpg.get_value("string_length_lt_constrained_applied"):
        dpg.add_text(tag = "string_length_lt_constrained_applied", default_value= "String Length Lesser Than Constraint Applied!", parent = "Length Lesser Than Options")
        dpg.add_text(tag = "string_length_lt_constrained_filter" , default_value = "String Length Lesser Than Constraint Applied!  " + constrained_lt_length_str, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "string_length_lt_constrained_filter", value = "String Length Lesser Than Constraint Filter: " + constrained_lt_length_str)
    edit_string_callback()


def phonetic_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Phonetic Options")
    else:
        dpg.hide_item("Phonetic Options")

def load_phonetic_callback():
    global phonetic_matching_string
    phonetic_matching_string = dpg.get_value("phonetic_word")
    if not dpg.get_value("phonetic_applied"):
        dpg.add_text(tag = "phonetic_applied", default_value= "Phonetic Constraint Applied!", parent = "Phonetic Options")
        dpg.add_text(tag = "phonetic_filter" , default_value = "Phonetic Constraint Applied!  " + phonetic_matching_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "phonetic_filter", value = "Phonetic Constraint Filter: " + phonetic_matching_string)
    edit_string_callback()

def semantic_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Semantic Options")
    else:
        dpg.hide_item("Semantic Options")

def load_semantic_callback():
    global semantic_matching_string
    global semantic_distance_threshold
    semantic_matching_string = dpg.get_value("semantic_word")
    semantic_distance_threshold = dpg.get_value("semantic_distance")
    semantic_distance_threshold_string = str(semantic_distance_threshold)
    if not dpg.get_value("semantic_applied"):
        dpg.add_text(tag = "semantic_applied", default_value= "Semantic Constraint Applied!", parent = "Semantic Options")
        dpg.add_text(tag = "semantic_filter" , default_value = "Semantic Constraint Applied!  " + semantic_matching_string + " Minimum Similarity: " + semantic_distance_threshold_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "semantic_filter", value = "Semantic Constraint Filter: " + semantic_matching_string)
    edit_string_callback()


def palindrome_callback(sender, app_data, user_data):
    global palindrome_enabled
    if app_data == True:
        dpg.show_item("Palindrome Options")
        palindrome_enabled = True
    else:
        dpg.hide_item("Palindrome Options")
        palindrome_enabled = False

def load_palindrome_callback():
    global palindrome_enabled
    if not dpg.get_value("palindrome_applied"):
        dpg.add_text(tag = "palindrome_applied", default_value= "String Palindromeic Constraint Applied!", parent = "Palindrome Options")
        dpg.add_text(tag = "palindrome_filter" , default_value = "String Palindromeic Constraint Applied!  ",  parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "palindrome_filter", value = "String Palindromeic Constraint Applied! ")
    edit_string_callback()

def anagram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Anagram Options")
    else:
        dpg.hide_item("Anagram Options")

def load_anagram_callback():
    global anagram_string
    anagram_string = dpg.get_value("anagram_string")
    if not dpg.get_value("anagram_applied"):
        dpg.add_text(tag = "anagram_applied", default_value= "Anagram Applied!", parent = "Anagram Options")
        dpg.add_text(tag = "anagram_filter" , default_value = "Anagram Applied!  " + anagram_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "anagram_filter", value = "Anagram Filter: " + anagram_string)
    edit_string_callback()

def partial_anagram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Partial Anagram Options")
    else:
        dpg.hide_item("Partial Anagram Options")

def load_partial_anagram_callback():
    global partial_anagram_string
    partial_anagram_string = dpg.get_value("partial_anagram_string")
    if not dpg.get_value("partial_anagram_applied"):
        dpg.add_text(tag = "partial_anagram_applied", default_value= "Partial Anagram Applied!", parent = "Partial Anagram Options")
        dpg.add_text(tag = "partial_anagram_filter" , default_value = "Partial Anagram Applied!  " + partial_anagram_string, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "partial_anagram_filter", value = "Partial Anagram Filter: " + partial_anagram_string)
    edit_string_callback()

def isogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Isogram Options")
    else:
        dpg.hide_item("Isogram Options")

def load_isogram_callback():
    global isogram_count
    isogram_count = dpg.get_value("isogram_number")
    isogram_count_str = str(isogram_count)
    if not dpg.get_value("isogram_applied"):
        dpg.add_text(tag = "isogram_applied", default_value= "Isogram Applied!", parent = "Isogram Options")
        dpg.add_text(tag = "isogram_filter" , default_value = "Isogram Applied!  " + isogram_count_str, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "isogram_filter", value = "Isogram Filter: " + isogram_count_str)
    edit_string_callback()

def reverse_isogram_callback(sender, app_data, user_data):
    if app_data == True:
        dpg.show_item("Reverse Isogram Options")
    else:
        dpg.hide_item("Reverse Isogram Options")

def load_reverse_isogram_callback():
    global reverse_isogram_count
    reverse_isogram_count = dpg.get_value("reverse_isogram_number")
    reverse_isogram_count_str = str(reverse_isogram_count)
    if not dpg.get_value("reverse_isogram_applied"):
        dpg.add_text(tag = "reverse_isogram_applied", default_value= "Reverse Isogram Applied!", parent = "Reverse Isogram Options")
        dpg.add_text(tag = "reverse_isogram_filter" , default_value = "Reverse Isogram Applied!  " + reverse_isogram_count_str, parent = "main_window", before = "lipogram")
    else:
        dpg.set_value(item = "reverse_isogram_filter", value = "Reverse Isogram Filter: " + reverse_isogram_count_str)
    edit_string_callback()


def turn_filters_off_callback():
    global lipogram_naughty_word_list
    global weak_lipogram_naughty_word_list
    global reverse_lipogram_nice_word_list
    global weak_reverse_lipogram_nice_word_list
    global string_in_positon_list
    global string_in_positon_index_list
    global starts_with_string
    global ends_with_string
    global phonetic_matching_string
    global semantic_matching_string
    global semantic_distance_threshold
    global constrained_length
    global constrained_gt_length
    global constrained_lt_length
    global palindrome_enabled
    global anagram_string
    global partial_anagram_string
    global isogram_count
    global reverse_isogram_count 

    lipogram_naughty_word_list = []
    weak_lipogram_naughty_word_list = []
    reverse_lipogram_nice_word_list = []
    weak_reverse_lipogram_nice_word_list = []
    string_in_positon_list = []
    string_in_positon_index_list = []
    starts_with_string = ""
    ends_with_string = ""
    phonetic_matching_string = ""
    semantic_matching_string = ""
    semantic_distance_threshold = 0.0
    constrained_length = 0
    constrained_gt_length = 0
    constrained_lt_length = 0
    palindrome_enabled = False
    anagram_string = ""
    partial_anagram_string = ""
    isogram_count = 0
    reverse_isogram_count = 0
    if dpg.get_value("partial_anagram_applied"):
        dpg.delete_item("partial_anagram_applied")
    if dpg.get_value("partial_anagram_filter"):
        dpg.delete_item("partial_anagram_filter")
    dpg.set_value("partial_anagram_string", "")
    #TODO: Finish this


dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()
#dpg.configure_app(docking=True, dock_space = True)

dpg.enable_docking(dock_space=True)

with dpg.font_registry():
    # Download font here: https://fonts.google.com/specimen/Open+Sans
    font = dpg.add_font("OpenSans-VariableFont_wdth,wght.ttf", 15, tag="ttf-font"
    )

dpg.bind_font(font)


"""
Doesn't work... 

def edited_call(sender, app_data, user_data):
    global starttime
    newtime = time.time()
    if newtime - starttime > 2:
        print(dpg.get_item_)
        print(dpg.get_item_state("input_string"))
        edit_string_callback()
        starttime = newtime
"""



#edit_string_callback("This is an example")

with dpg.window(tag = "main_window", label="CTGS - Contrained Text Generation Studio") as window:
    dpg.add_text("Main Settings")
    with dpg.window(tag = "load_model_window", label = "Model Settings", pos = (700, 200)) as model_window:
        dpg.add_text("Enter the name of the pre-trained model from transformers that we are using for Text Generation")
        dpg.add_text("This will download a new model, so it may take awhile or even break if the model is too large")
        dpg.add_input_text(tag = "model_name", width = 500, height = 500, default_value="distilgpt2", label = "Huggingface Model Name")
        dpg.add_button(tag="load_model", label="load_model", callback=load_model)
    with dpg.window(tag="Filter Options", show = True, pos = (700, 300)) as filter_options:
        dpg.add_text("Select which filters you want to enable")
        dpg.add_text("List of enabled filters: ")
        dpg.add_checkbox(tag="lipogram", label = "All Strings Banned", callback=lipogram_callback)
        _help("A lipogram is when a particular letter or group of strings is avoided.")
        with dpg.child_window(tag="Lipogram Options", show = False, height = 100, width = 600) as lipogram_selection_window:
            dpg.add_text("Add banned letters or strings seperated by a space!")
            dpg.add_input_text(tag = "lipogram_word_list", width = 500, height = 500, label = "Banned Strings")
            dpg.add_button(tag="lipogram_button", label="Load Banned Strings", callback=load_naughty_strings_callback)
        dpg.add_checkbox(tag="weak_lipogram", label = "Any Strings Banned", callback=weak_lipogram_callback)
        _help("A weak lipogram is when at least one of a particular letter or group of strings is avoided.")
        with dpg.child_window(tag="Weak Lipogram Options", show = False, height = 100, width = 600) as weak_lipogram_selection_window:
            dpg.add_text("Add banned letters or strings seperated by a space!")
            dpg.add_input_text(tag = "weak_lipogram_word_list", width = 500, height = 500, label = "Banned Strings")
            dpg.add_button(tag="weak_lipogram_button", label="Load Banned Strings", callback = load_weak_naughty_strings_callback)
        dpg.add_checkbox(tag="reverse_lipogram", label = "All Strings Required", callback=reverse_lipogram_callback)
        _help("A reverse lipogram is when a particular letter or group of strings is forced.")
        with dpg.child_window(tag="Reverse Lipogram Options", show = False, height = 100, width = 600) as reverse_lipogram_selection_window:
            dpg.add_text("Add forced letters or strings seperated by a space!")
            dpg.add_input_text(tag = "reverse_lipogram_word_list", width = 500, height = 500, label = "Forced Strings")
            dpg.add_button(tag="reverse_lipogram_button", label="Load Forced Strings", callback = load_reverse_naughty_strings_callback)
        dpg.add_checkbox(tag="weak_reverse_lipogram", label = "Any Strings Required", callback=weak_reverse_lipogram_callback)
        _help("A weak reverse lipogram is when at least one of a particular letter or group of strings is forced.")


        with dpg.child_window(tag="Weak Reverse Lipogram Options", show = False, height = 100, width = 600) as weak_reverse_selection_window:
            dpg.add_text("Add forced letters or strings seperated by a space!")
            dpg.add_input_text(tag = "weak_reverse_lipogram_word_list", width = 500, height = 500, label = "Forced Strings")
            dpg.add_button(tag="weak_reverse_lipogram_button", label="Load Forced Strings", callback = load_weak_reverse_naughty_strings_callback)

        dpg.add_checkbox(tag="string_position", label = "String In Position", callback = string_position_callback)
        _help("This allows one to force a particular letter in a particular position of a string\n" "NOTE: It's recommended to combine this with whitespace stripping")

        with dpg.child_window(tag="Letter Position Options", show = False, height = 130, width = 600) as letter_position_selection_window:
            dpg.add_text("Add the position that you want to force a particular letter to appear at! Give a list of characters seperated by a space")
            dpg.add_input_text(tag = "string_for_position", width = 500, height = 500, label = "List of characters")
            dpg.add_text("Corresponding list of indexes for each character. Must be the same length as the list of characters")
            dpg.add_input_text(tag = "string_position_int", width = 500, height = 500, label = "List of indexes")
            dpg.add_button(tag="string_position_button", label="Load Strings", callback = load_string_positon_callback)

        dpg.add_checkbox(tag="string_starts", label = "String Starts With", callback = string_starts_with_callback)
        _help("This allows one to guarantee that the string will start with a particular set of letters")

        with dpg.child_window(tag="Starting String Options", show = False, height = 100, width = 600) as starting_string_selection_window:
            dpg.add_text("Add the string that the word should start with")
            dpg.add_input_text(tag = "string_start_word", width = 500, height = 500, label = "String for word to start with")
            dpg.add_button(tag="string_start_button", label="Load Starting String", callback=load_string_starts_with_callback)

        dpg.add_checkbox(tag="string_ends", label = "String Ends With", callback = string_ends_with_callback)
        _help("This allows one to guarantee that the string will end with a particular set of letters")
        with dpg.child_window(tag="Ending String Options", show = False, height = 100, width = 600) as ending_string_selection_window:
            dpg.add_text("Add the string that the word should end with")
            dpg.add_input_text(tag = "string_end_word", width = 500, height = 500, label = "String for word to end with")
            dpg.add_button(tag="string_end_button", label="Load Ending String", callback=load_string_ends_with_callback)

        dpg.add_checkbox(tag="length_constrained", label = "String Length Equal To", callback = string_length_constrained_callback)
        _help("This allows one to guarantee that the string will be of a particular length\n" "NOTE: It's recommended to combine this filter with whitespace stripping")
        with dpg.child_window(tag="Length Constrained Options", show = False, height = 100, width = 600) as length_constrained_selection_window:
            dpg.add_text("Specify the length that you want your strings to be constrained to")
            dpg.add_input_int(tag = "length_constrained_number", label = "Number to constrain the length with")
            dpg.add_button(tag="length_constrained_button", label="Load Length Constrained String", callback=load_string_length_constrained_callback)

        dpg.add_checkbox(tag="length_gt", label = "String Length Greater Than", callback = string_length_gt_constrained_callback)
        _help("This allows one to guarantee that the string will be longar than a particular length\n" "NOTE: It's recommended to combine this filter with whitespace stripping")

        with dpg.child_window(tag="Length Greater Than Options", show = False, height = 100, width = 600) as length_gt_selection_window:
            dpg.add_text("Specify the length that you want your strings to be greater than")
            dpg.add_input_int(tag = "length_gt_constrained_number", label = "Number to constrain the length to be greater than")
            dpg.add_button(tag="length_gt_constrained_button", label="Load Length Constrained String", callback=load_string_length_gt_constrained_callback)


        dpg.add_checkbox(tag="length_lt", label = "String Length Lesser Than", callback = string_length_lt_constrained_callback)
        _help("This allows one to guarantee that the string will be shorter than particular length\n" "NOTE: It's recommended to combine this filter with whitespace stripping")

        with dpg.child_window(tag="Length Lesser Than Options", show = False, height = 100, width = 600) as length_lt_selection_window:
            dpg.add_text("Specify the length that you want your strings to be lesser than")
            dpg.add_input_int(tag = "length_lt_constrained_number", label = "Number to constrain the length to be lesser than")
            dpg.add_button(tag="length_lt_constrained_button", label="Load Length Constrained String", callback=load_string_length_lt_constrained_callback)


        dpg.add_checkbox(tag="phonetic", label = "Phonetic Matching", callback = phonetic_callback)
        _help("This uses the double metaphone algorithm to phonetically match the output to a particular string")

        with dpg.child_window(tag="Phonetic Options", show = False, height = 100, width = 600) as phonetic_selection_window:
            dpg.add_text("Specify the word you want to phonetically match against")
            dpg.add_input_text(tag = "phonetic_word", width = 500, height = 500, label = "String to match Phonetically")
            dpg.add_button(tag="phonetic_constrained_button", label="Load Phonetically Matching Strings", callback=load_phonetic_callback)


        dpg.add_checkbox(tag="semantic", label = "Semantic Matching", callback = semantic_callback)
        _help("This uses fasttext word vectors to return strings which are semantically similar to the provided string")

        with dpg.child_window(tag="Semantic Options", show = False, height = 100, width = 600) as semantic_selection_window:
            dpg.add_text("Specify the word you want to semantically match against")
            dpg.add_input_text(tag = "semantic_word", width = 500, height = 500, label = "String to match Semantically")
            dpg.add_input_float(tag = "semantic_distance", label = "Similarity that the word has to be higher than")
            dpg.add_button(tag="semantic_constrained_button", label="Load Semantic Matching Strings", callback=load_semantic_callback)

        dpg.add_checkbox(tag="palindrome", label = "Palindrome", callback = palindrome_callback)
        _help("A palindrome is a string which reads the same backward as forward, such as madam or racecar")

        with dpg.child_window(tag="Palindrome Options", show = False, height = 100, width = 600) as palindrome_selection_window:
            dpg.add_text("Press the button to force all generated strings to be palindromes!")
            dpg.add_button(tag="palindrome_button_enabled", label="Load Palindromic String", callback=load_palindrome_callback)



        dpg.add_checkbox(tag="anagram", label = "Anagram", callback = anagram_callback)
        _help("An anagram is a string formed by rearranging the letters of a different string")
        with dpg.child_window(tag="Anagram Options", show = False, height = 100, width = 600) as anagram_selection_window:
            dpg.add_text("Specify the string that you want generated strings to be anagrams of!")
            dpg.add_input_text(tag = "anagram_string", width = 500, height = 500, label = "Anagram String")
            dpg.add_button(tag="anagram_button", label="Load Anagramic String", callback=load_anagram_callback)

        dpg.add_checkbox(tag="partial_anagram", label = "Partial Anagram", callback = partial_anagram_callback)
        _help("A partial anagram is a string constructed by rearranging somr or all of the letters of a different string")

        with dpg.child_window(tag="Partial Anagram Options", show = False, height = 100, width = 600) as partial_anagram_selection_window:
            dpg.add_text("Specify the string that you want generated strings to be partial anagrams of!")
            dpg.add_input_text(tag = "partial_anagram_string", width = 500, height = 500, label = "Partial Anagram String")
            dpg.add_button(tag="partial_anagram_button", label="Load Partial Anagramic String", callback=load_partial_anagram_callback)

        dpg.add_checkbox(tag="isogram", label = "Isogram", callback = isogram_callback)
        _help("An isogram is a string in which none of its characters appear more than the provided number of times")

        with dpg.child_window(tag="Isogram Options", show = False, height = 100, width = 600) as isogram_window:
            dpg.add_text("Specify the number of times characters are allowed to repeat")
            dpg.add_input_int(tag = "isogram_number", label = "Number of times characters can repeat", default_value = 1)
            dpg.add_button(tag="isogram_button", label="Load Isogramic String", callback=load_isogram_callback)

        dpg.add_checkbox(tag="reverse_isogram", label = "Reverse Isogram", callback = reverse_isogram_callback)
        _help("A reverse isogram is a string in which all of its characters appear more than the provided number of times")
    
        with dpg.child_window(tag="Reverse Isogram Options", show = False, height = 100, width = 600) as reverse_isogram_window:
            dpg.add_text("Specify the number of times that characters must repeat")
            dpg.add_input_int(tag = "reverse_isogram_number", label = "Number of times characters must repeat", default_value = 1)
            dpg.add_button(tag="reverse_isogram_button", label="Load Reverse Isogramic String", callback=load_reverse_isogram_callback)
        
        dpg.add_button(tag="remove_filters", label = "Reset Filters", callback = turn_filters_off_callback)

    dpg.add_input_text(tag = "input_string", width = 500, height = 500, multiline=True, default_value = "Type something here!")

    dpg.add_button(label="Predict New Tokens", callback=edit_string_callback)
    dpg.add_button(label="AI generate some tokens", callback=generate_tokens_callback)
    dpg.add_input_int(tag = "top_p", label = "Number of tokens for top P sampling", default_value = 0)
    dpg.add_input_int(tag = "top_k", label = "Number of tokens for top K sampling", default_value = 0)
    dpg.add_input_float(tag = "temperature", label = "Temperature", default_value = 1.0)
    dpg.add_input_int(tag = "num_tokens_to_generate", label = "Number of tokens to generate", default_value = 20)
    
    with dpg.window(tag="Transforms", show = True, label = "Text Transforms", pos = (700, 1000)) as pre_filter_options:
        dpg.add_text("These are text transforms which will apply to all tokens *before* the actual filters are applied")
        dpg.add_text("Using these transforms will frequently assist with increasing the vocabulary that is available after a filter is applied")
        dpg.add_text("These transforms can also prevent generation of undesierable characters")
        dpg.add_checkbox(tag="upper_case_transform", label = "Uppercase transform")
        _help("Use this to cause all text to be UPPERCASED LIKE THIS")
        dpg.add_checkbox(tag="lower_case_transform", label = "Lowercase transform")
        _help("Use this to force all tokens to be lowercased like this")
        dpg.add_checkbox(tag="replace_spaces", label = "Remove spaces")
        _help("Use this to force all tokens to not have spaces likethis")
        dpg.add_checkbox(tag = "lstrip_transform", label = "Left side strip")
        _help("Use this to stip spaces from the left of the text")
        dpg.add_checkbox(tag = "rstrip_transform", label = "Right side strip")
        _help("Use this to stip spaces from the right of the text")
        dpg.add_checkbox(tag = "strip_transform", label = "Full strip")
        _help("Use this to stip spaces from both sides of the text")
        dpg.add_checkbox(tag = "capitalize_first_letter_transform", label = "Capitalize the first letter")
        _help("Use this to force all tokens to have their first letter capitalized Like This")
        dpg.add_checkbox(tag = "alpha_numaric_transform", label = "Force tokens to be alpha numaric")
        _help("Use this to force all tokens to be alphanumaric, meaning only using the alphabet or numbers")
        dpg.add_checkbox(tag = "alpha_transform", label = "Force tokens to be alphaic")
        _help("Use this to force all tokens to be alphaic, meaning only using the alphabet")
        dpg.add_checkbox(tag = "digit_transform", label = "Force tokens to be digits")
        dpg.add_checkbox(tag = "ascii_transform", label = "Force tokens to be ascii characters")
        dpg.add_checkbox(tag = "filter_blank_outputs", label = "Filter blank outputs")

load_model(the_name="distilgpt2")

edit_string_callback()
dpg.set_global_font_scale(1.0)

dpg.show_viewport()
#dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()

