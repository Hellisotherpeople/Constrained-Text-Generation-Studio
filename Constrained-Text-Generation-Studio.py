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

tokenizer = ""
model = ""



def load_model(the_name):
    global tokenizer
    global model
    if not the_name:
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
        location_in_word = position_index_list[idx]
        if len(word[0]) > location_in_word:
            if word[0][location_in_word] != string:
                    return False
    return True

def palindrome(word):
    the_string = word[0]
    if the_string == the_string[::-1]:
        return True
    else:
        return False

def partial_anagram(word, a_string):
    if Counter(word[0]) - Counter(a_string):
         return False
    return True

def full_anagram(word, a_string):
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

temperature = 1.0
number_of_tokens_to_sample = 25000
replace_spaces = False
selection_window = False
upper_case_transform = False
lower_case_transform = True
lstrip_transform = False 
rstrip_transform = False
strip_transform = False
capitalize_first_letter_transform = False
alpha_numaric_transform = False 
alpha_transform = True
digit_transform = False 
ascii_transform = False
filter_blank_outputs = True

lipogram_naughty_word_list = []
weak_lipogram_naughty_word_list = []
reverse_lipogram_nice_word_list = []
weak_reverse_lipogram_nice_word_list = []
string_in_positon_list = []
string_in_positon_index_list = []
starts_with_string = ""
ends_with_string = ""
constrained_length = 0
constrained_gt_length = 0
constrained_lt_length = 0
palindrome_enabled = False




def get_next_word_without_e(sequence):
    all_letters_filtered_list = []
    #print(tokenizer)
    input_ids = tokenizer.encode(sequence, return_tensors="pt")
    # get logits of last hidden state
    next_token_candidates_logits = model(input_ids)[0][:, -1, :]
    if temperature != 1.0:
        next_token_candidates_logits = next_token_candidates_logits / temperature
    # filter
    filtered_next_token_candidates_logits = top_k_top_p_filtering(next_token_candidates_logits, top_k=number_of_tokens_to_sample, top_p=number_of_tokens_to_sample)
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
        if upper_case_transform:
            resulting_string = resulting_string.upper()
        if lower_case_transform:
            resulting_string = resulting_string.lower()
        if replace_spaces == True:
            resulting_string = resulting_string.replace(' ', '')
        if lstrip_transform:
            resulting_string = resulting_string.lstrip()
        if rstrip_transform:
            resulting_string = resulting_string.rstrip()
        if strip_transform:
            resulting_string = resulting_string.strip()
        if capitalize_first_letter_transform:
            resulting_string = resulting_string.capitalize()
        if alpha_numaric_transform:
            resulting_string = ''.join(ch for ch in resulting_string if ch.isalnum())
        if alpha_transform:
            resulting_string = ''.join(ch for ch in resulting_string if ch.isalpha())
        if digit_transform:
            resulting_string = ''.join(ch for ch in resulting_string if ch.isdigit())
        if ascii_transform:
            resulting_string = ''.join(ch for ch in resulting_string if ch.isascii())
        if filter_blank_outputs:
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
        if return_word == True:
            all_letters_filtered_list.append(word)
        

    #all_letters_filtered_list = [word for word in word_list if all_letters_not_included(word=word, starting_string= "EN")]
    #list(filter(all_letters_included, word_list))
    #print(probs)
    #print(all_letters_filtered_list[0:50])
    #print(probs)


                
    return all_letters_filtered_list


def add_generated_word_callback(sender, app_data, user_data):
    current_value = dpg.get_value("string")
    new_string = current_value + str(user_data)
    new_value = dpg.set_value("string", new_string)
    edit_string_callback()

def edit_string_callback():
    string_input = dpg.get_value("string")
    returned_words = get_next_word_without_e(string_input)
    #print(returned_words)
    with dpg.popup(parent = "string"):
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

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()


with dpg.font_registry():
    # Download font here: https://fonts.google.com/specimen/Open+Sans
    font = dpg.add_font("OpenSans-VariableFont_wdth,wght.ttf", 15, tag="ttf-font"
    )

dpg.bind_font(font)

load_model(the_name="distilgpt2")
#edit_string_callback("This is an example")

with dpg.window(tag = "main_window", label="CTGS - Contrained Text Generation Studio") as window:
    dpg.add_text("Main Settings")
    dpg.add_text("Enter the name of the pre-trained model from transformers that we are using for Text Generation")
    dpg.add_text("This will download a new model, so it may take awhile or even break if the model is too large")
    dpg.add_input_text(tag = "model_name", width = 500, height = 500, default_value="distilgpt2", label = "Huggingface Model Name")
    dpg.add_button(tag="load_model", label="load_model", callback=load_model)
    dpg.add_text("Select which filters you want to enable")
    dpg.add_text("List of enabled filters: ")
    dpg.add_checkbox(tag="lipogram", label = "All Strings Banned", callback=lipogram_callback)

    with dpg.child_window(tag="Lipogram Options", show = False, height = 100, width = 600) as lipogram_selection_window:
        dpg.add_text("Add naughty letters or strings seperated by a space!")
        dpg.add_input_text(tag = "lipogram_word_list", width = 500, height = 500, label = "Banned Strings")
        dpg.add_button(tag="lipogram_button", label="Load Banned Strings", callback=load_naughty_strings_callback)

    dpg.add_checkbox(tag="weak_lipogram", label = "Any Strings Banned", callback=weak_lipogram_callback)

    with dpg.child_window(tag="Weak Lipogram Options", show = False, height = 100, width = 600) as weak_lipogram_selection_window:
        dpg.add_text("Add naughty letters or strings seperated by a space!")
        dpg.add_input_text(tag = "weak_lipogram_word_list", width = 500, height = 500, label = "Banned Strings")
        dpg.add_button(tag="weak_lipogram_button", label="Load Banned Strings", callback = load_weak_naughty_strings_callback)


    dpg.add_checkbox(tag="reverse_lipogram", label = "All Strings Required", callback=reverse_lipogram_callback)

    with dpg.child_window(tag="Reverse Lipogram Options", show = False, height = 100, width = 600) as reverse_lipogram_selection_window:
        dpg.add_text("Add nice letters or strings seperated by a space!")
        dpg.add_input_text(tag = "reverse_lipogram_word_list", width = 500, height = 500, label = "Forced Strings")
        dpg.add_button(tag="reverse_lipogram_button", label="Load Forced Strings", callback = load_reverse_naughty_strings_callback)

    dpg.add_checkbox(tag="weak_reverse_lipogram", label = "Any Strings Required", callback=weak_reverse_lipogram_callback)

    with dpg.child_window(tag="Weak Reverse Lipogram Options", show = False, height = 100, width = 600) as weak_reverse_selection_window:
        dpg.add_text("Add nice letters or strings seperated by a space!")
        dpg.add_input_text(tag = "weak_reverse_lipogram_word_list", width = 500, height = 500, label = "Forced Strings")
        dpg.add_button(tag="weak_reverse_lipogram_button", label="Load Forced Strings", callback = load_weak_reverse_naughty_strings_callback)

    dpg.add_checkbox(tag="string_position", label = "String In Position", callback = string_position_callback)

    with dpg.child_window(tag="Letter Position Options", show = False, height = 130, width = 600) as letter_position_selection_window:
        dpg.add_text("WARNING: This is a bit slow!")
        dpg.add_text("Add the position that you want to force a particular letter to appear at! Give a list of characters seperated by a space")
        dpg.add_input_text(tag = "string_for_position", width = 500, height = 500, label = "List of characters")
        dpg.add_text("Corresponding list of indexes for each character. Must be the same length as the list of characters")
        dpg.add_input_text(tag = "string_position_int", width = 500, height = 500, label = "List of indexes")
        dpg.add_button(tag="string_position_button", label="Load Strings", callback = load_string_positon_callback)

    dpg.add_checkbox(tag="string_starts", label = "String Starts With", callback = string_starts_with_callback)

    with dpg.child_window(tag="Starting String Options", show = False, height = 100, width = 600) as starting_string_selection_window:
        dpg.add_text("Add the string that the word should start with")
        dpg.add_input_text(tag = "string_start_word", width = 500, height = 500, label = "String for word to start with")
        dpg.add_button(tag="string_start_button", label="Load Starting String", callback=load_string_starts_with_callback)

    dpg.add_checkbox(tag="string_ends", label = "String Ends With", callback = string_ends_with_callback)

    with dpg.child_window(tag="Ending String Options", show = False, height = 100, width = 600) as ending_string_selection_window:
        dpg.add_text("Add the string that the word should end with")
        dpg.add_input_text(tag = "string_end_word", width = 500, height = 500, label = "String for word to end with")
        dpg.add_button(tag="string_end_button", label="Load Ending String", callback=load_string_ends_with_callback)

    dpg.add_checkbox(tag="length_constrained", label = "String Length Equal To", callback = string_length_constrained_callback)

    with dpg.child_window(tag="Length Constrained Options", show = False, height = 100, width = 600) as length_constrained_selection_window:
        dpg.add_text("Specify the length that you want your strings to be constrained to")
        dpg.add_input_int(tag = "length_constrained_number", label = "Number to constrain the length with")
        dpg.add_button(tag="length_constrained_button", label="Load Length Constrained String", callback=load_string_length_constrained_callback)

    dpg.add_checkbox(tag="length_gt", label = "String Length Greater Than", callback = string_length_gt_constrained_callback)

    with dpg.child_window(tag="Length Greater Than Options", show = False, height = 100, width = 600) as length_gt_selection_window:
        dpg.add_text("Specify the length that you want your strings to be greater than")
        dpg.add_input_int(tag = "length_gt_constrained_number", label = "Number to constrain the length to be greater than")
        dpg.add_button(tag="length_gt_constrained_button", label="Load Length Constrained String", callback=load_string_length_gt_constrained_callback)


    dpg.add_checkbox(tag="length_lt", label = "String Length Lesser Than", callback = string_length_lt_constrained_callback)

    with dpg.child_window(tag="Length Lesser Than Options", show = False, height = 100, width = 600) as length_lt_selection_window:
        dpg.add_text("Specify the length that you want your strings to be lesser than")
        dpg.add_input_int(tag = "length_lt_constrained_number", label = "Number to constrain the length to be lesser than")
        dpg.add_button(tag="length_lt_constrained_button", label="Load Length Constrained String", callback=load_string_length_lt_constrained_callback)

    dpg.add_checkbox(tag="palindrome_button", label = "Palindrome", callback = palindrome_callback)

    with dpg.child_window(tag="Palindrome Options", show = False, height = 100, width = 600) as palindrome_selection_window:
        dpg.add_text("Press the button to force all generated strings to be palindromes!")
        dpg.add_button(tag="palindrome_button_enabled", label="Load Palindromic String", callback=load_palindrome_callback)



    dpg.add_checkbox(tag="anagram_button", label = "Anagram")
    dpg.add_checkbox(tag="partial_anagram_button", label = "Partial Anagram")
    dpg.add_checkbox(tag="isogram_button", label = "Isogram")
    dpg.add_checkbox(tag="reverse_isogram_button", label = "Reverse Isogram")
    dpg.add_input_text(tag = "string", width = 500, height = 500, multiline=True, default_value = "Type something here!")
    dpg.add_button(label="Predict New Tokens", callback=edit_string_callback)
edit_string_callback()
dpg.set_global_font_scale(1.0)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

