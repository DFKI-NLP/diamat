'''
Created on 7 May 2019

@author: elav01
'''
import sys
import os
import numpy as np 
from tools.score import get_classification_accuracy, get_explanations
        

def compare_systems(jsonl_urls):
    results = []
    for jsonl_url in jsonl_urls:
        result = {}
        try:
            name = os.path.basename(jsonl_url).split('.')[1]
        except:
            name = os.path.basename(jsonl_url)
            
        #result['accuracy'] = get_classification_accuracy(jsonl_url)
        
        explanations = get_explanations(jsonl_url)
        #softmax_sentences = [e.softmax['machine'] for e in explanations]
        #result['softmax_sentences'] = softmax_sentences
        #result['softmax_sum'] = np.sum(softmax_sentences)
        #result['softmax_prod'] = np.prod(softmax_sentences)
        #result['softmax_avg'] = np.average(softmax_sentences)
        
        sum_contributions = 0
        for explanation in explanations:
            seq = explanation.machine
            sum_contributions += np.sum(seq.scores)
        result['sum_contributions'] = sum_contributions
        
        #logit_sentences = [e.predictions['machine'] for e in explanations]
        #result['logit_sum'] = np.sum(logit_sentences)
        #result['logit_sum_pos'] = np.sum([l for l in logit_sentences if l>0])
        #result['logit_prod'] = np.prod(logit_sentences)
        #result['logit_avg'] = np.average(logit_sentences)
        #result['logit_avg_pos'] = np.average([l for l in logit_sentences if l>0])
        
        results.append((result, name))
    
    for key in results[0][0].keys():
        print(key)
        print("-----")
        for result, name in sorted(results, key=lambda x: x[0][key]):
            print(name, result[key])
        print("")
        
    print("Sentences: ", len(explanations)) 
    print("Systems: ", len(results))        

if __name__ == '__main__':
    jsonl_urls = sys.argv[1:]
    compare_systems(jsonl_urls)