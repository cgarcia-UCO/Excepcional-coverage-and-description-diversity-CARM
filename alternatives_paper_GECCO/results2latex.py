import os.path

from numpy.core import unicode
from numpy.ma import arange
import sys
import pandas as pd
from scipy.io import arff
import numpy as np
import io
from scipy.io.arff._arffread import ParseArffError
import warnings
from collections import Counter
import re
from xml.dom.minidom import parse, parseString
from xml.parsers.expat import ExpatError
#warnings.filterwarnings("error")
sys.path.insert(1, './python_code')
from utilities.representation.assoc_rules import RuleSet, Rule


# Global variable for compatibility with utilities.representation.assoc_rules package
class Dummy_class:
    pass
params = {'FITNESS_FUNCTION': Dummy_class()}
params['FITNESS_FUNCTION'].training_exp = None



def read_feature_types(filename):
    """
    This function reads the types of the features
    """

    types = {}
    translation = {}
    counter = 0
    with open(filename, "r", encoding='utf-8', errors="ignore") as f:
        l = f.readline()
    
        while l:
            if l[0:10] == "@attribute":
                words = l.split()
                nameAttribute = words[1].strip()

                if len(words) < 3 and '{' in l and '}' in l:
                    new_words = words[1].split('{')
                    words[1] = new_words[0]
                    words.append('{'+new_words[1])
                    nameAttribute = words[1].strip()

                try:
                    if (words[2][0] == '{'):
                        types[nameAttribute] = "categorical"
                        translation[nameAttribute] = counter
                        counter +=1
                    elif (words[2].startswith('real') or words[2].startswith('integer')):
                        types[nameAttribute] = "real"
                        translation[nameAttribute] = counter
                        counter +=1
                    else:
                        raise Exception("Unrecognized line: "+ str(words[2]))
                except IndexError:
                    print(words)
                    raise
            else:
                pass
    
            l = f.readline()
            pass
    
    return types, translation

def read_cat_feat_values(filename):
    values = {}

    with open(filename, 'rb') as f:

        for l in f:
            try:
                l = l.decode("utf-8")

                if l.startswith('@attribute '):
                    feature = re.search('@attribute ([^\{]+)', l).group(1)
                    these_values = re.findall('\'[^\']+\'', l)
                    if len(these_values) == 0:
                        these_values = re.search('\{([^\}]*)\}', l).group(1)
                        these_values = these_values.split(',')
                        these_values = [i.replace(',','').strip() for i in these_values]
                    values[feature] = these_values
            except UnicodeDecodeError:
                pass
            except Exception:
                print(filename,'\n',l)
                raise

    return values


def read_num_feat_intervals(filename):
    
    intervals = {}
 
    with open(filename, "r") as f:
        l = f.readline()
    
        while l:
            words = l.split(":")
    
            head = words[0]
    
            if (head[0:9] == "Cut point"):
                indexAttribute = head.split("attribute ")
                indexAttribute = int(indexAttribute[1])
                cuts = []
                cuts.append(float(words[1]))
                l = f.readline()
                words = l.split(":")
                head = words[0]
    
                while (head[0:9] == "Cut point"):
                    cuts.append(float(words[1]))
                    l = f.readline()
                    words = l.split(":")
                    head = words[0]
    
                intervals[indexAttribute] = cuts
    
            elif (head[0:9] == "Number of"):
                pass
    
            else:
                print("Unrecognized line: ", head[0:9])
    
            l = f.readline()
            
    return intervals


def read_feature_translation(filename):
    """
    This function reads the translation of each feature

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    translation = {}
    with open(filename, "r", encoding='utf-8', errors="ignore") as f:
        l = f.readline()
    
        while l and len(l) <= 0:
            l = f.readline()
    
        while l:
            if l[0:11] == "@Number of ":
                words = l.split("Variable ")
                indexVar = words[1].split(":")
                indexVar = int(indexVar[0])
                l = f.readline()
                nameVar = l.split(":")
                nameVar = nameVar[0]
                translation[nameVar] = indexVar
            else:
                pass
    
            l = f.readline()
            while l and len(l) <= 0:
                l = f.readline()
                
    return translation

def read_interval(string):
    left = re.search('\[[^,]+,', string)[0].strip()
    left = left[1:len(left) - 1]
    right = re.search(',[^\]]+\]', string)[0][1:].strip()
    right = right[:len(right) - 1]
    return left, right

def read_condition(string):
    if '>=' in string:
        words = string.split(' >= ')
        feature, op, value = words[0], '>=', words[1]
    elif '<=' in string:
        words = string.split(' <= ')
        feature, op, value = words[0], '<=', words[1]
    elif '<>' in string:
        words = string.split(' <> ')
        feature, op, value = words[0], '!=', words[1]
    elif '=' in string:
        words = string.split(' = ')
        feature, op, value = words[0], '==', words[1]
    elif '<' in string:
        words = string.split(' < ')
        feature, op, value = words[0], '<', words[1]
    elif '>' in string:
        words = string.split(' > ')
        feature, op, value = words[0], '>', words[1]
    else:
        raise Exception('weird condition: '+string)

    return feature, op, value

def read_MOPNAR_NICGAR_QARCIP_rules(filename, goal_feature, goal_value, ruleset, ruleset_strs, numConditions, dataset_name):
    dataset = dataset_name
    params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:,-1]
    params['POSITIVE_CLASS'] = goal_value
    params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)
    rule_set = RuleSet(params)
    try:
        document = parse(filename)
    except ExpatError:
        with open(filename, 'r') as f:
            content = ''.join(f.readlines())
            content = content.replace('<100_ml', '&lt;100_ml')
            with io.StringIO(content) as f2:
                document = parse(f2)
    rules = document.getElementsByTagName('rule')

    for i in rules:
        antecedents = i.getElementsByTagName('antecedents')
        consequents = i.getElementsByTagName('consequents')

        for j_consequent in consequents:
            conditions_consequent = j_consequent.getElementsByTagName('attribute')
            for k_consequent in conditions_consequent:
                feature = k_consequent.getAttribute('name').strip()
                value = k_consequent.getAttribute('value').strip()
                if feature == goal_feature and (value == goal_value or
                                                ('NOT' in value and
                                                 'NOT '+goal_value not in value and
                                                 len(set(dataset.iloc[:,-1])) == 2)):  # In some occasions, these algorithms set consequent as NOT X, and I am using binary datasets

                    for j in antecedents:
                        conditions = j.getElementsByTagName('attribute')
                        antecedent = []
                        for k in conditions:
                            value = k.getAttribute('value').replace('&lt;','<')
                            feature = k.getAttribute('name')
                            if '[' in value:
                                try:
                                    left, right = read_interval(value)

                                except Exception:
                                    print('ERROR CON: ', value, end='')
                                    raise

                                if 'NOT' in value:
                                    condition = ('~', ((feature,'>=',left),(feature,'<=',right)))  # NOT representation
                                    antecedent.append(condition)
                                else:
                                    condition = (feature, '>=', left)
                                    antecedent.append(condition)
                                    condition = (feature, '<=', right)
                                    antecedent.append(condition)

                            else:
                                if "'" not in value:
                                    value = "'"+value.strip()+"'"
                                if 'NOT' in value:
                                    condition = (feature, '!=', value.strip())
                                else:
                                    condition = (feature, '==', value.strip())
                                antecedent.append(condition)

                        if str(antecedent) not in ruleset_strs and "Miss" not in str(antecedent):
                            ruleset.append(antecedent)
                            new_rule = Rule(params)
                            new_rule.set_consequent(goal_value)
                            for i_conds in antecedent:
                                new_rule.add_condition('('+get_str_for_evaluation(i_conds,dataset,'x')+')')
                            rule_set.add_rule(new_rule)
                            ruleset_strs.add(str(antecedent))
                            numConditions.append(len(conditions))

                else:
                    pass
    return numConditions, ruleset, ruleset_strs, rule_set


def read_CN2_SD_rules(filename, intervals, cat_values, goal_feature, goal_value, ruleset, ruleset_strs, numConditions, translation, types,
                        original_rules = None, dataset_name = None):
    dataset = dataset_name
    params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:,-1]
    params['POSITIVE_CLASS'] = goal_value
    params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)
    rule_set = RuleSet(params)
    with open(filename, "r", encoding='utf-8', errors="ignore") as f:
        for l in f:
            if not l.startswith('Rule'):
                continue

            consequent = re.search('.* THEN '+goal_feature+' -> ([^\s]+)', l).group(1)

            if consequent == goal_value:
                rule = re.search('Rule [^\s]+ IF  (.+) THEN', l).group(1)
                validRule = True
                conditions = rule.split(' AND ')
                cadena = []

                for cond in conditions:
                    feature, op, value = read_condition(cond)
                    indexFeature = translation[feature]
                    value = value.replace("'", "").replace("_"," ")

                    if types[feature] == "categorical":
                        value = round(float(value))
                        if op == '==':
                            concrete_value = cat_values[feature][value].replace("_"," ")
                            cond = (feature, op, "'"+concrete_value.replace("'","").strip()+"'")
                        elif op == '<>' or op == '!=':
                            try:
                                concrete_value = cat_values[feature][value].replace("_", " ")
                            except IndexError:
                                print(feature, value, cat_values)
                                raise
                            cond = (feature, '!=', "'"+concrete_value.replace("'","").strip()+"'")
                        elif op == '<=':
                            cond = ('|',[])
                            for i in reversed(range(value+1)):
                                concrete_value = cat_values[feature][i].replace("_", " ")
                                cond[1].append((feature,'==',"'"+concrete_value.replace("'","").strip()+"'"))
                        elif op == '>=':
                            cond = ('|', [])
                            for i in range(value,len(cat_values[feature])):
                                concrete_value = cat_values[feature][i].replace("_", " ")
                                cond[1].append((feature, '==', "'"+concrete_value.replace("'","").strip()+"'"))
                        elif op == '>':
                            cond = ('|', [])
                            for i in range(value+1, len(cat_values[feature])):
                                concrete_value = cat_values[feature][i].replace("_", " ")
                                cond[1].append((feature, '==', "'"+concrete_value.replace("'","").strip()+"'"))
                        elif op == '<':
                            cond = ('|', [])
                            for i in reversed(range(value)):
                                concrete_value = cat_values[feature][i].replace("_", " ")
                                cond[1].append((feature, '==', "'"+concrete_value.replace("'","").strip()+"'"))
                        else:
                            print('weird operator', op)
                            raise Exception

                        cadena.append(cond)
                    elif types[feature] == "real":
                        value = round(float(value))
                        if op == '==':
                            if int(value) == 0:
                                cadena.append((feature, "<=", intervals[indexFeature][0]))
                                if intervals[indexFeature][0] < 0:
                                    validRule = False
                            elif int(value) == len(intervals[indexFeature]):
                                cadena.append((feature, ">=", intervals[indexFeature][len(intervals[indexFeature]) - 1]))
                            else:
                                try:
                                    cadena.append((feature, ">=", intervals[indexFeature][int(value) - 1]))
                                    cadena.append((feature, "<=", intervals[indexFeature][int(value)]))
                                    if intervals[indexFeature][int(value) - 1] < 0:
                                        validRule = False
                                except:
                                    print(feature, "(", len(intervals[indexFeature]), "): ", int(value))
                                    raise

                        elif op == '<>' or op == '!=':
                            if int(value) == 0:
                                cadena.append((feature, ">=", intervals[indexFeature][0]))
                            elif int(value) == len(intervals[indexFeature]):
                                cadena.append((feature, "<=", intervals[indexFeature][len(intervals[indexFeature]) - 1]))
                            else:
                                try:
                                    left = intervals[indexFeature][int(value) - 1]
                                    right = intervals[indexFeature][int(value)]
                                    cond = ('~', ((feature,'>=',left),(feature,'<=',right)))
                                    cadena.append(cond)
                                except:
                                    print(feature, "(", len(intervals[indexFeature]), "): ", int(value))
                                    raise

                        elif op == '>=':
                            if int(value) == 0:
                                pass
                            else:
                                cadena.append((feature, ">=", intervals[indexFeature][int(value)-1]))
                        elif op == '<=':
                            if int(value) > len(intervals[indexFeature]):
                                pass
                            else:
                                cadena.append((feature, "<=", intervals[indexFeature][int(value)]))
                        elif op == '>':
                            if int(value) == 0:
                                pass
                            elif int(value) >= len(intervals[indexFeature]):
                                raise Exception('Condition out of its bounds: '+str(cond)+'('+str(value)+') : '+str(intervals[indexFeature]))
                            else:
                                cadena.append((feature, ">", intervals[indexFeature][int(value)]))
                        elif op == '<':
                            if int(value) == 0:
                                raise Exception('Condition out of its bounds: '+str(cond)+'('+str(value)+') : '+str(intervals[indexFeature]))
                            elif int(value) >= len(intervals[indexFeature]):
                                pass
                            else:
                                cadena.append((feature, "<=", intervals[indexFeature][int(value)-1]))
                        else:
                            print('weird numeric op', op)
                            sys.exit(1)
                    else:
                        print("Feature en regla no encontrada: ", feature)

                if validRule:
                    if str(cadena) not in ruleset_strs and "Miss" not in cadena:
                        original_rules.append(rule)
                        new_rule = Rule(params)
                        new_rule.set_consequent(goal_value)
                        for i_conds in cadena:
                            new_rule.add_condition('(' + get_str_for_evaluation(i_conds, dataset, 'x') + ')')
                        rule_set.add_rule(new_rule)
                        ruleset.append(cadena)
                        ruleset_strs.add(str(cadena))
                        numConditions.append(len(conditions))

    return numConditions, ruleset, ruleset_strs, original_rules, rule_set


def get_str_for_evaluation(condition, dataset, name_dataset):
    if condition[0] == '~':
        condition_str = '(~ ((dataset[\'' + str(condition[1][0][0]) + '\']' \
                        + str(condition[1][0][1]) + str(condition[1][0][2]) + ')' \
                        + ' & ' \
                        + '(dataset[\'' + str(condition[1][1][0]) + '\']' \
                        + str(condition[1][1][1]) + str(condition[1][1][2]) + ')' \
                        + '))'
    elif condition[0] == '|':
        first_cond = condition[1][0]
        condition_str = '((dataset[\''+str(first_cond[0])+ '\']' + str(first_cond[1]) + str(first_cond[2]) + ') '
        for i in condition[1][1:]:
            condition_str += ' | (dataset[\'' + str(i[0]) + '\']' + str(i[1]) + str(i[2]) + ')'
        condition_str += ')'
    else:
        condition_str = 'dataset[\'' + str(condition[0]) + '\']' + condition[1] + str(condition[2])
        try:
            _ = eval(condition_str)
        except (NameError, SyntaxError):
            condition_str = 'dataset[\'' + str(condition[0]) + '\']' + condition[1] + "'" + str(condition[2]) + "'"
    return condition_str.replace('dataset', name_dataset)

def read_CBA2CPAR_rules(filename, intervals, goal_value, ruleset, ruleset_strs,
                        numConditions, translation, types,
                        original_rules = None, dataset_name = None):
    dataset = dataset_name
    params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:,-1]
    params['POSITIVE_CLASS'] = goal_value
    params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)
    rule_set = RuleSet(params)
    with open(filename, "r", encoding='utf-8', errors="ignore") as f:
        for l in f:
            words = l.split(":")

            if len(words) < 3:
                pass
            elif words[2].strip() == goal_value:
                rule = words[1]
                validRule = True
                conditions = rule.split(' AND ')
                cadena = []

                for cond in conditions:
                    parts = cond.split(' IS ')
                    feature = parts[0].strip()
                    indexFeature = translation[feature]-1
                    value = parts[1].strip()
                    value = value.replace("'", "").replace("_"," ")

                    if types[feature] == "categorical":
                        cadena.append((feature, "==", "'"+str(value)+"'"))
                    elif types[feature] == "real":
                        if int(value) == 0:
                            cadena.append((feature, "<=", intervals[indexFeature][0]))
                            if intervals[indexFeature][0] < 0:
                                validRule = False
                        elif int(value) == len(intervals[indexFeature]):
                            cadena.append((feature, ">=", intervals[indexFeature][len(intervals[indexFeature]) - 1]))
                        else:
                            try:
                                cadena.append((feature, ">=", intervals[indexFeature][int(value) - 1]))
                                cadena.append((feature, "<=", intervals[indexFeature][int(value)]))
                                if intervals[indexFeature][int(value) - 1] < 0:
                                    validRule = False
                            except:
                                print(feature, "(", len(intervals[indexFeature]), "): ", int(value))
                                raise

                    else:
                        print("Feature en regla no encontrada: ", feature)

                if validRule:
                    if str(cadena) not in ruleset_strs and "Miss" not in cadena:
                        original_rules.append(rule)
                        ruleset.append(cadena)
                        new_rule = Rule(params)
                        new_rule.set_consequent(goal_value)
                        for i_conds in cadena:
                            j_conds = list(i_conds)
                            j_conds[0] = '(x[\''+j_conds[0]+'\']'
                            new_rule.add_condition("".join(map(lambda x: str(x),j_conds))+')')
                        rule_set.add_rule(new_rule)
                        ruleset_strs.add(str(cadena))
                        numConditions.append(len(conditions))

    return numConditions, ruleset, ruleset_strs, original_rules, rule_set


def results_CBA2_etal_2latex(dataset_name, goal_value, algname, whole_dataset):
    dataset = whole_dataset
    header = 'alternatives_paper_GECCO/test_CBA_CPAR_FARCHD_final/'
    numfolds = 1
    intervalFiles = []
    indexNameTranslation = []
    comprobationFiles = []
    rulesFiles = []

    for i in range(numfolds):
        filename = header + "datasets/KMeans-MV.UniformFrequency-D." + dataset_name + "/result" + str(i + 1) + "s0e0.txt"
        if not os.path.exists(filename):
            filename = header + "datasets/KMeans-MV." + dataset_name + "/result" + str(i + 1) + "s0e0.txt"
        intervalFiles.append(filename)
        indexNameTranslation.append(header +
            "results/KMeans-MV.UniformFrequency-D."+algname+
            '-C.'+dataset_name+'/'+
            "result" +
            "0e0.txt")
        filename = header + "datasets/" + dataset_name + "/" + dataset_name + "-tra.dat"
        if not os.path.exists(filename):
            filename = header + "datasets/" + dataset_name + "/" + dataset_name + ".dat"
        comprobationFiles.append(filename)
        rulesFiles.append(
            header + 
            "results/KMeans-MV.UniformFrequency-D."+algname+
            '-C.'+dataset_name+'/'+
            "result" + str(i) + "e1.txt")


    folds = len(rulesFiles)
    ruleset = []
    original_rules = []
    ruleset_strs = set()
    numConditions = []

    # For each fold
    for iFiles in range(folds):
        types,_ = read_feature_types(comprobationFiles[iFiles])

        # Read the translation of the feature and its index
        translation = read_feature_translation(indexNameTranslation[iFiles])

        # Read the intervals of numeric features and values for categorical features
        intervals = read_num_feat_intervals(intervalFiles[iFiles])

        # Read rules
        numConditions, ruleset, ruleset_strs,\
            original_rules, as_a_ruleset = read_CBA2CPAR_rules(rulesFiles[iFiles], intervals, goal_value,
                                                 ruleset, ruleset_strs, numConditions, translation, types,
                                                 original_rules, whole_dataset)

    return as_a_ruleset

def results_FARCHD_2latex(dataset_name, goal_value, whole_dataset):

    dataset = whole_dataset
    params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:,-1]
    params['POSITIVE_CLASS'] = goal_value
    params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)
    rule_set = RuleSet(params)

    folds = 1
    numConditions = []
    ruleset = []
    ruleset_strs = set()
    original_rules = []

    header = 'alternatives_paper_GECCO/test_CBA_CPAR_FARCHD_final/'
    comprobationFiles = []
    filename = header + "datasets/" + dataset_name + "/" + dataset_name + "-tra.dat"
    if not os.path.exists(filename):
        filename = header + "datasets/" + dataset_name + "/" + dataset_name + ".dat"
    comprobationFiles.append(filename)
    
    # For each fold
    for iFiles in range(folds):
        # Read the type of each feature
        types,_ = read_feature_types(comprobationFiles[iFiles])

    files_head = 'alternatives_paper_GECCO/test_CBA_CPAR_FARCHD_final/results/KMeans-MV.Fuzzy-FARCHD-C.'+\
                 dataset_name+'/'

    for iFiles in range(folds):

        # Read fuzzy labels
        labelfile = files_head+"result"+str(iFiles)+"s0e0.txt"
        f = open(labelfile, "r", encoding='utf-8', errors="ignore")
        l = f.readline()

        intervals = {}

        while l:
            words = l.split(":")

            if (len(words) > 1):
                if (len(words[1].strip()) <= 0):
                    featureName = words[0]
                    l = f.readline()
                    intervals[featureName] = {}

                    while (l and len(l.strip()) > 0):
                        values = l.split(": ")
                        label = values[0]
                        label = label.split("_")
                        label = label[1].split("(")
                        label = label[0]
                        values = values[1].replace("(","").replace(")","")
                        values = values.split(',')
                        intervals[featureName][label] = {}
                        intervals[featureName][label]["min"] = float(values[0])
                        intervals[featureName][label]["medium"] = float(values[1])
                        intervals[featureName][label]["max"] = float(values[2])
                        intervals[featureName][label]["First05"] = (float(values[0]) + float(values[1])) / 2
                        intervals[featureName][label]["Second05"] = (float(values[1]) + float(values[2])) / 2
                        l = f.readline()

            l = f.readline()

        f.close()

        # Read categorical features
        catValuesFile = files_head+"result"+str(iFiles)+"s0.tra"
        f = open(catValuesFile, "r", encoding='utf-8', errors="ignore")
        traduccion = {}

        for l in f:
            words = l.split()

            if (words[0] == "@attribute"):

                if (len(words) <= 2): # It is a categorical feature
                    words = words[1].split('{')
                    featureName = words[0]
                    values = words[1].split(',')
                    values[len(values)-1] = values[len(values)-1].replace('}','')

                    for i in arange(len(values)):
                        values[i] = values[i].replace("'","")
                        values[i] = values[i].replace("_"," ")

                    traduccion[featureName] = values
                else: # It is a numerical feature
                    words = words[1].split()
                    featureName = words[0]

        #Read rules
        f.close()
        rulesFile = files_head+"result"+str(iFiles)+"s0e1.txt"
        f = open(rulesFile, "r", encoding='utf-8', errors="ignore")

        for l in f:
            words = l.split(":")

            if (len(words) > 2):
                consequent = words[2].split()

                if (consequent[0] == goal_value):
                    rule = words[1]
                    validRule = True
                    conditions = rule.split(' AND ')
                    cadena = []

                    for cond in conditions:
                        parts = cond.split(' IS ')
                        feature = parts[0].strip()
                        value = parts[1].split('(')
                        value = value[0].split('_')
                        value = value[1]

                        if (feature in traduccion):
                            cadena.append((feature, "==", "'"+str(traduccion[feature][int(value)])+"'"))
                        else:
                            cadena.append((feature, '>=', str(round(intervals[feature][value]["First05"],7))))
                            cadena.append((feature, '<=', str(round(intervals[feature][value]["Second05"], 7))))
                            if intervals[feature][value]["Second05"] < 0:
                                validRule = False

                    if validRule:
                        if str(cadena) not in ruleset_strs and "Miss" not in str(cadena):
                            original_rules.append(rule)
                            ruleset.append(cadena)
                            new_rule = Rule(params)
                            new_rule.set_consequent(goal_value)
                            for i_conds in cadena:
                                new_rule.add_condition('('+get_str_for_evaluation(i_conds,dataset,'x')+')')
                            rule_set.add_rule(new_rule)
                            ruleset_strs.add(str(cadena))
                            numConditions.append(len(conditions))
        f.close()

    return rule_set

def results_MOPNAR_NICGAR_QARCIP_2latex(i_alg, rules_file_dataset, goal_feature, goal_value, whole_dataset):
    header = 'alternatives_paper_GECCO/test_MOPNAR_NICGAR_QARCIP_final/'
    file = header+'results/KMeans-MV.' + i_alg + '-A.' + rules_file_dataset + '/result0s0.tra'

    ruleset = []
    ruleset_strs = set()
    numConditions = []

    numConditions, ruleset, ruleset_strs, rule_set = read_MOPNAR_NICGAR_QARCIP_rules(file, goal_feature,
                                                                           goal_value, ruleset,
                                                                           ruleset_strs, numConditions, whole_dataset)

    return rule_set

def results2latex_CN2_SD(dataset_name, goal_feature, goal_value, algname, alg_conf, whole_dataset):
    dataset = whole_dataset
    header = 'alternatives_paper_GECCO/'

    numfolds = 1
    intervalFiles = []
    indexNameTranslation = []
    comprobationFiles = []
    rulesFiles = []
    trainingFiles = []

    for i in range(numfolds):
        intervalFiles.append(header+
                             "test_SubgroupDisc_final/"
                             "datasets/UniformFrequency-D." +
                             dataset_name + "/result" + str(i + 1) + "s0e0.txt")
        indexNameTranslation.append(header+
            "test_SubgroupDisc_final/datasets/UniformFrequency-D."+dataset_name+'/'+
            "result1s" + str(i) + "e0.txt")
        comprobationFiles.append(header+
                                 "test_SubgroupDisc_final/"
                                 "datasets/" + dataset_name + "/" + dataset_name +
                                 ".dat")
        filename = header+\
            "test_SubgroupDisc_final/results/"+str(alg_conf)+"-UniformFrequency-D."+algname+\
            '.'+dataset_name+'/'+\
            "result" + str(i) + "e0.txt"
        if not os.path.exists(filename):
            filename = header + \
                       "test_SubgroupDisc_final/results/" + "UniformFrequency-D." + algname + \
                       '.' + dataset_name + '/' + \
                       "result" + str(i) + "e0.txt"
        rulesFiles.append(filename)

        filename=header + 'test_SubgroupDisc_final/results/' + str(alg_conf) + '-UniformFrequency-D.' + algname +\
                 '.' + dataset_name + '/' + 'result' + str(i) + '.tra'
        if not os.path.exists(filename):
            filename=header + 'test_SubgroupDisc_final/results/' + 'UniformFrequency-D.' + algname +\
                     '.' + dataset_name + '/' + 'result' + str(i) + '.tra'
        trainingFiles.append(filename)

    folds = len(rulesFiles)
    ruleset = []
    original_rules = []
    ruleset_strs = set()
    numConditions = []

    # For each fold
    for iFiles in range(folds):
        
        # Read the type of each feature
        types, translation = read_feature_types(comprobationFiles[iFiles])

        # Read the intervals of numerical features and values for categorical features
        cat_values = read_cat_feat_values(trainingFiles[iFiles])
        intervals = read_num_feat_intervals(intervalFiles[iFiles])

        # Read rules
        numConditions, ruleset, ruleset_strs,\
            original_rules, rule_set = read_CN2_SD_rules(rulesFiles[iFiles], intervals, cat_values,
                                                         goal_feature, goal_value,
                                                         ruleset, ruleset_strs, numConditions,
                                                         translation, types,
                                                         original_rules, whole_dataset)

    return rule_set

