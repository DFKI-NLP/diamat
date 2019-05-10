'''
Created on 7 May 2019

@author: elav01
'''
import sys
from server import jsonl_to_explanation, cache


def get_explanations(jsonl_url):
    explanations = []
    with open(jsonl_url, 'r') as fin:
        for line in fin:
            explanations.append(jsonl_to_explanation(line))
    return explanations

def get_classification_accuracy(jsonl_url):
    explanations = get_explanations(jsonl_url)
    success_machine = 0
    for explanation in explanations:
        score_machine = explanation.predictions['machine']
        if score_machine > 0:
            success_machine+=1
        
    return 1.0 * success_machine / len(explanations)
if __name__ == '__main__':
    json_url = sys.argv[1]
    print(get_classification_accuracy(json_url))
    