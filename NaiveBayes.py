import numpy as np
from math import sqrt
from math import exp
from math import pi

def separate_by_class(dataset):
    """
    Using the function `separate_by_class`, we calculate the probability of data by
    the class they belong to, the so-called base rate.
    """
    
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated
 
def summarize_dataset(dataset):
 	"""
    Using the function `summarize_dataset`, we calculate mean, standard deviation
    and count for each column in a dataset
    """
    
    summaries = [(std.mean(column), np.std(coulmn), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
 

def summarize_by_class(dataset):
    """
    Using the function `summarize_by_class` we calculate the statistics for each
    row.
    """
    
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries
 

def calculate_probability(x, mean, stdev):
  	"""
    Using the function `calculate_probability`, we calculate Gaussian probability
    distribution function for x.
    """
    
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent
  

def calculate_class_probabilities(summaries, row):
  	"""
    Using the function `calculate_class_probabilities`, we calculate probabilities
    of predicting each class for a given row.
    """
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in
