# CS 410 (Text Information Systems) Fall 2018 Project 
# Dilruba Hawk and Zeed Jarrah
# Build My Battlestation

import sys
from bs4 import BeautifulSoup
import urllib3
import re
import nltk
import requests
import time
import operator

# Info: Tokenizes and and labels each word in the file with a Part-Of-Speech tag
# Input: A line of input from the document
# Output Part-Of-Speech tagged words
def preprocess_text(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

# Info: We create a chunk parser to identify a POS pattern to match AMD CPUs
# Input: Tokenized text data containing a part list used to build the computer
# Output: Dictionary containing CPUs as key with number of instances the CPU appears in the text
def amd_processor_entities(tokens):
    cpu = ""
    cpu_dict = dict()
    # AMD cpu pattern: <NNP><NNP><CD><CD><CD><NNP><CD>
    pattern = "NP: {<NNP><NNP><CD><CD><CD><NNP><CD>}"
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(tokens)
    for sub_tree in cs.subtrees():
        if sub_tree.label() == "NP":
            tree = sub_tree.flatten()
            if tree[0][0] == "AMD":
                cpu = entity_parser(tree)
                try:
                    cpu_dict[cpu] += 1
                except:
                    cpu_dict[cpu] = 1
    
    return cpu_dict

# Info: We create a chunk parser to identify a POS pattern to match Intel CPUs
# Input: Tokenized text data containing a part list used to build the computer
# Output: Dictionary containing CPUs as key with number of instances the CPU appears in the text
def intel_processor_entities(tokens):
    cpu = ""
    cpu_dict = dict()
    # Intel cpu pattern: <NNP><NNP><JJ><CD><NNP>
    pattern = "NP: {<NNP><NNP><JJ><CD><NNP>}"
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(tokens)
    for sub_tree in cs.subtrees():
        if sub_tree.label() == "NP":
            tree = sub_tree.flatten()
            if tree[0][0] == "Intel":
                cpu = entity_parser(tree)
                try:
                    cpu_dict[cpu] += 1
                except:
                    cpu_dict[cpu] = 1
    return cpu_dict

# Info: Composed of two entity recognizers that look for AMD and Intel cpus that are merged to one dictionary
# Input: Tokenized, POS tagged text data
# Output: Dictionary containing recognized cpus in build and repeats
def cpu_entity_recognizer(tokens):
    cpu_dict = dict()
    amd_cpus = amd_processor_entities(tokens)
    intel_cpus = intel_processor_entities(tokens)

    cpu_dict = {**amd_cpus, **intel_cpus}

    return cpu_dict

# Info: We create a chunk parser to identify a POS pattern to match Graphical Processing Units (GPUs)
# Input: Tokenized text data containing a part list used to build the computer
# Output: Dictionary containing GPUs as key with number of instances the GPU appears in the text
def gpu_entity_recognizer(tokens):
    gpu = ""
    gpu_dict = dict()
    pattern = "NP: {<NNP><NNP><NNP><CD><CD><NNP>?}"
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(tokens)
    for sub_tree in cs.subtrees(): 
        if sub_tree.label() == "NP":
            tree = sub_tree.flatten()
            gpu = entity_parser(tree)
        try:
            gpu_dict[gpu] += 1
        except:
            gpu_dict[gpu] = 1
    
    gpu_dict.pop('') # Removing key '' from dictionary
    return gpu_dict

# Info: Concatenates the words found in a labeled subtree.
# Input: Takes a flattened version of a tree with all non-root, non-terminals removed
# Output: Returns a concatenated Noun Phrase (NP) likely to be a computer part  
def entity_parser(tree):
    entity = ""
    for i in range(0, len(tree)):
        entity += " " + tree[i][0]
        entity = entity.lstrip()

    return entity

# Info: Returns the two most frequent occuring CPU and GPU pair with an alternative choice
# Input: Two sorted dictionaries (Assumes dictionaries are sorted in DESCENDING order)
# Output: Returns the top two CPUs and GPUs from the dictionaries
def display_cpu_gpu_options(cpu, gpu):
    choice = list()
    choice.append((cpu[0][0], gpu[0][0]))
    choice.append((cpu[1][0], gpu[1][0]))
    return choice

# Info: Creating a Name-Entity Recognizer (NER) to classify various components of a custom-built computer.
def main():
    if len(sys.argv) != 2:
        print("Usage: {} missing text file".format(sys.argv[0]))
        sys.exit(1)

    doc = sys.argv[1]    
    doc = open(doc, 'r')
    text = doc.read()
    doc.close()
    
    sent = preprocess_text(text)

    results = gpu_entity_recognizer(sent)
    print("\nGPU entity test: \n")
    sort_gpu = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    print(sort_gpu)

    results = cpu_entity_recognizer(sent)
    print("\nCPU entity test: \n")
    sort_cpu = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
    print(sort_cpu)

    options = display_cpu_gpu_options(sort_cpu, sort_gpu)
    print("\n\nResult: ")
    print(options[0])

    print("\n\n Alternative Choice: ")
    print(options[1])


if __name__ == "__main__":
    main()
