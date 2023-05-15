import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import tldextract
import re
import nltk
from math import log


# nltk.download('punkt')
# # Load the grammar file
# grammar = '''
#   S -> NP VP
#   NP -> DT NN
#   VP -> VBZ PP
#   PP -> IN NP
#   DT -> the
#   NN -> [a-zA-Z]+
#   VBZ -> is
#   IN -> of
# '''
# grammar_checker = nltk.RegexpParser(grammar)


def save_model(model, filename):
    with open(f'./Model/{filename}.obj', 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(f'./Model/{filename}.obj', 'rb') as f:
        return pickle.load(f)

def word_frequency(text):
  frequencies = {}
  for word in text.split():
    if word not in frequencies:
      frequencies[word] = 0
    frequencies[word] += 1

  return frequencies

def getFeatures(url):
  features = {}

  # Extract the length of the URL.
  features["length"] = len(url)

  # Extract the number of parameters in the URL.
  parameters = re.findall(r"\?(.*)", url)
  features["number_of_parameters"] = len(parameters) if parameters else 0

#   extracted_url = tldextract.extract(url)
  # # Extract the domain name of the URL.
  # domain_name = extracted_url.registered_domain
  # features["domain_name"] = domain_name

  # # Extract the top-level domain of the URL.
  # top_level_domain = extracted_url.suffix
  # features["top_level_domain"] = top_level_domain

  # Extract the number of digits in the URL.
  number_of_digits = len(re.findall(r"\d", url))
  features["number_of_digits"] = number_of_digits

  # Extract the number of hyphens in the URL.
  number_of_hyphens = len(re.findall(r"-", url))
  features["number_of_hyphens"] = number_of_hyphens

  # Extract the number of underscores in the URL.
  number_of_underscores = len(re.findall(r"_", url))
  features["number_of_underscores"] = number_of_underscores

  # Extract the number of special characters in the URL.
  number_of_special_characters = len(re.findall(r"[^a-zA-Z0-9-_]", url))
  features["number_of_special_characters"] = number_of_special_characters

  # Extract the number of words in the URL.
  words = re.findall(r"\w+", url)
#   features["number_of_words"] = len(words) if words else 0

  # Extract the average length of the words in the URL.
#   if words:
#     average_word_length = sum(len(word) for word in words) / len(words)
#     features["average_word_length"] = average_word_length
#   else:
#     features["average_word_length"] = 0

#   Extract the entropy of the URL.
  entropy = 0
  for word in words:
    entropy += -sum(val * log(val) for val in word_frequency(word).values())
  features["entropy"] = entropy

  #   # Check if the URL contains any grammatical errors.
  # try:
  #   tree = grammar_checker.tree(url)
  #   if not tree.check():
  #     features["contains_grammatical_error"] = True
  # except Exception as e:
  #   print(e)
  #   features["contains_grammatical_error"] = False

# Extract the use of social engineering techniques
  is_social_engineering = re.search(r'free|gift|prize|win|click|here|now', url) is not None
  features['is_social_engineering'] =  int(is_social_engineering)

  return features