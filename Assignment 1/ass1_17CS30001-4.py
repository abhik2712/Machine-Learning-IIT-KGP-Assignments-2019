#Roll No: 17CS30001
#Name : Abhik Naskar
#Assignment Number - 1 : Decision Trees
#compile by command on terminal : python ass1_17CS30001.py

import pandas as pd
import math

def entropy(probs):
    import math
    return sum([-prob * math.log(prob, 2) for prob in probs])

def entropy_list(a):
    from collections import Counter
    cnt = Counter(x for x in a)
    total_size = len(a)*1.0
    probs = [x/total_size for x in cnt.values()]
    return entropy(probs)
    
##Information Gain for the attributes

def information_gain(data, attribute_name, target_name, trace = 0):
    data_split = data.groupby(attribute_name)
    
    number_of_observations = len(data.index)*1.0
    data_agg_ent = data_split.agg({target_name : [entropy_list, lambda x: len(x)/number_of_observations] })[target_name]
    
    data_agg_ent.columns = ['Entropy', 'PropObservations']
    
    new_entropy = sum( data_agg_ent['Entropy'] * data_agg_ent['PropObservations'] )
    old_entropy = entropy_list(data[target_name])
    return old_entropy - new_entropy

def id3(data, target_name, attribute_names, default_class=None):
    from collections import Counter
    
    cnt = Counter(x for x in data[target_name]) ##check yes or no
    ##check if the split is homogeneous
    if len(cnt) == 1:
        return next(iter(cnt))
    
    ##check if the split is empty or not
    ## if empty return a default value
    elif data.empty or (not attribute_names):
        return 'no'            ##return None or empty
    
    ##Else perform the algorithm
    else:
        default_class = max(cnt.keys()) #max of yes or no
        #computing the information gain of the attributes
        gain = [information_gain(data, attr, target_name, default_class) for attr in attribute_names]
        
        index_of_max = gain.index(max(gain)) #index of best attribute
        best_attribute = attribute_names[index_of_max]
        
        # Create an empty tree, to be populated in a moment
        tree = {best_attribute:{}} # Iniiate the tree with best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attribute]
        #print(data.groupby(best_attribute))
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in data.groupby(best_attribute):
            subtree = id3(data_subset,
                        target_name,
                        remaining_attribute_names,
                        default_class)
            tree[best_attribute][attr_val] = subtree
        return tree

if __name__ == '__main__':
    data = pd.read_csv("/Users/abhiknaskar/Downloads/data1_19.csv")
    attribute_names = list(data.columns)
    print("List of Attributes:", attribute_names) 
    attribute_names.remove('survived') #Remove the class attribute
    print("Predicting Attributes:", attribute_names)
    from pprint import pprint
    tree = id3(data,'survived',attribute_names)
    print("\n\nThe Resultant Decision Tree is :\n")
    # print(tree)
    # print("\n")
    pprint(tree)
    attribute = next(iter(tree))
    print("Best Attribute :\n",attribute)
    print("Tree Keys:\n",tree[attribute].keys())

