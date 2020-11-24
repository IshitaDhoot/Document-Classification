import math
import os
from collections import Counter 

def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    f = open(filepath, "r")
    line = f.readline().strip()
    while line:
        if vocab.count(line) == 0:
            if None not in bow.keys():
                bow[None] = 1
            else:
                bow[None] += 1
            line = f.readline().strip()
            continue
        if line not in bow.keys():
            bow[line] = 1
        else:
            bow[line] += 1
        line = f.readline().strip()
    f.close()
    
    
    return bow

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    for root, dirs, files in os.walk(directory):
     for file in files:
         dataset.append({'label': os.path.basename(root), 'bow':create_bow(vocab,root+'/'+file )})
         
    return dataset

def removeElements(lst, k): 
    counted = Counter(lst) 
    return [el for el in lst if counted[el] >= k] 

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    vocab = []
    temp = []
    
    for root, dirs, files in os.walk(directory):
      for file in files:
        with open(os.path.join(root, file), "r") as auto:
            for i in auto.read().splitlines():
                temp.append(i)
        auto.close()

    vocab = removeElements(temp, cutoff)
            
    vocab = sorted(list(dict.fromkeys(vocab)))
    return vocab

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    documentsCount = 0
    label2016 = 0
    label2020 = 0
    for i in training_data:
        if list(training_data[documentsCount].values())[0] == '2016':
            label2016 = label2016+1
        if list(training_data[documentsCount].values())[0] == '2020':
            label2020 = label2020+1
        documentsCount = documentsCount + 1
            
    logprob['2020'] = math.log((label2020 + smooth))-math.log((documentsCount+2))
    logprob['2016'] = math.log((label2016 + smooth))-math.log((documentsCount+2))

    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    
    smooth = 1 # smoothing factor
    word_prob = {}
    totalWords = 0
    noneWords = 0
    
    for i in vocab:
        word_prob[i] = 0
    
    for i in training_data:
        if i.get('label') == label:
            for j in i.get('bow'):
                totalWords = totalWords + i.get('bow').get(j)
                if j == None:
                    noneWords = noneWords + i.get('bow').get(j)
                    continue
                word_prob[j] = word_prob[j] + i.get('bow').get(j)
                
    
    for i in vocab:
        word_prob[i] = math.log(word_prob[i] + smooth*1) - math.log(totalWords + smooth*(len(vocab)+1))
        word_prob[None] = math.log(noneWords + smooth*1) - math.log(totalWords + smooth*(len(vocab)+1))
    
    return word_prob

    
##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    training_data = load_training_data(create_vocabulary(training_directory, cutoff), training_directory)
    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)
    retval['log prior'] = prior(training_data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(create_vocabulary(training_directory, cutoff), training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(create_vocabulary(training_directory, cutoff), training_data, '2020')

    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    
    P2016 = model['log prior']['2016']
    P2020 = model['log prior']['2020']
    bow = create_bow(model['vocabulary'], filepath)
    pWordGivenLabel2016 = 0
    pWordGivenLabel2020 = 0
    
    for key in model['log p(w|y=2016)']:
        if key in bow:
            pWordGivenLabel2016 += model['log p(w|y=2016)'][key]*bow[key]
    
    for key in model['log p(w|y=2020)']:
        if key in bow:
            pWordGivenLabel2020 += model['log p(w|y=2020)'][key]*bow[key]
        
    final2016 = P2016 + pWordGivenLabel2016
    final2020 = P2020 + pWordGivenLabel2020
  
    retval['log p(y=2020|x)'] = final2020
    retval['log p(y=2016|x)'] = final2016
    if final2020>final2016:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    
    return retval

model = train('./corpus/training/', 2)
print(classify(model, './corpus/test/2016/0.txt'))