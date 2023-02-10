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

def read_apriori_condition(string):
    string = string.replace('OD280_OD315', 'OD280/OD315')

    try:
        feature, value = string.split('=')
    except ValueError:
        words = string.split('=')
        feature = words[0]
        value = '='.join(words[1:])


    if '[' == value.strip()[0] or '(' in value.strip()[0]:
        try:
            left, right = value.replace('[','').replace(']','').replace('(','').replace(')','').split(',')
        except ValueError:
            print('VALUE: ', value, string)
            raise
        result = ('&', [])

        if '[' in value:
            result[1].append((feature, '>=', float(left)))
        elif '(' in value:
            result[1].append((feature, '>', float(left)))
        else:
            raise Exception('Weird error')

        if ']' in value:
            result[1].append((feature, '<=', float(right)))
        elif ')' in value:
            result[1].append((feature, '<', float(right)))
        else:
            raise Exception('Weird error')

        return result
    else:
        return (feature, '==', '"'+value.strip()+'"')


def read_heuops_condition(string):
    def remove_spaces(string):
        string = string.strip()
        first_char = string[0]

        if first_char == '"' or first_char == "'":
            string = first_char + string.replace(first_char,'').strip() + first_char

        return string

    string = string.replace('OD280_OD315','OD280/OD315')

    if '>=' in string:
        words = string.split('>=')
        feature, op, value = remove_spaces(words[0]), '>=', remove_spaces(words[1])
        result = (feature, op, value)
    elif '<=' in string:
        words = string.split('<=')
        feature, op, value = remove_spaces(words[0]), '<=', remove_spaces(words[1])
        result = (feature, op, value)
    elif '<>' in string:
        words = string.split('<>')
        feature, op, value = remove_spaces(words[0]), '!=', remove_spaces(words[1])
        result = (feature, op, value)
    elif '==' in string:
        words = string.split('=')
        feature, op, value = remove_spaces(words[0]), '==', remove_spaces(words[1])
        result = (feature, op, value)
    elif '%in%' in string:
        words = string.split('%in%')
        feature, op, value = remove_spaces(words[0]), ' in ', words[1].strip().replace('c','').\
            replace('(','').replace(')','')
        result = ('|',[])
        values = value.split(',')
        result[1].append((feature, '==',remove_spaces(values[0])))
        for i_value in values[1:]:
            result[1].append((feature, '==',remove_spaces(i_value)))
    elif '=' in string:
        words = string.split('=')
        feature, op, value = remove_spaces(words[0]), '==', remove_spaces(words[1])
        result = (feature, op, value)
    elif '<' in string:
        words = string.split('<')
        feature, op, value = remove_spaces(words[0]), '<', remove_spaces(words[1])
        result = (feature, op, value)
    elif '>' in string:
        words = string.split('>')
        feature, op, value = remove_spaces(words[0]), '>', remove_spaces(words[1])
        result = (feature, op, value)
    else:
        raise Exception('weird condition: '+string)

    return result

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
    elif condition[0] == '&':
        first_cond = condition[1][0]
        condition_str = '((dataset[\'' + str(first_cond[0]) + '\']' + str(first_cond[1]) + str(first_cond[2]) + ') '
        for i in condition[1][1:]:
            condition_str += ' & (dataset[\'' + str(i[0]) + '\']' + str(i[1]) + str(i[2]) + ')'
        condition_str += ')'
    else:
        try:
            condition_str = 'dataset[\'' + str(condition[0]) + '\']' + str(condition[1]) + str(condition[2])
        except IndexError:
            print('Error:', condition)
            raise
        try:
            _ = eval(condition_str)
        except (NameError, SyntaxError):
            condition_str = 'dataset[\'' + str(condition[0]) + '\']' + condition[1] + "'" + str(condition[2]) + "'"
        except TypeError:
            print(condition_str)
            print(eval('dataset[\''+str(condition[0])+'\']'))
            raise
    return condition_str.replace('dataset', name_dataset)

def read_rules_apriori_from_file(filename, min_conf, dataset, goal_value, params):
    params['CACHE'] = False
    rule_set = RuleSet(params)

    with open(filename) as f:
        ruleset_str = []

        for line in f:

            if  'No rules' in line:
                continue

            set_conditions = []
            conditions = line.split(' => ')[0].replace('{','').replace('}','')
            conditions = re.findall('[^,]+=[^,\[]+|[^,]+=\[[^,]+,[^,]+\]|[^,]+=\[[^,]+,[^,]+\)|[^,]+=\([^,]+,[^,]+\]|[^,]+=\([^,]+,[^,]+\)', conditions)
            result = read_apriori_condition(conditions[0].strip())
            set_conditions.append(result)

            for i_cond in conditions[1:]:
                result = read_apriori_condition(i_cond.strip())
                set_conditions.append(result)

            if str(set_conditions) not in ruleset_str:
                new_rule = Rule(params)
                new_rule.set_consequent(goal_value)
                for i_cond in set_conditions:
                    try:
                        new_rule.add_condition('('+get_str_for_evaluation(i_cond, dataset, 'x')+ ')')
                    except KeyError:
                        print('ERROR')
                        print(i_cond)
                        print(get_str_for_evaluation(i_cond, dataset, 'x'))
                        raise

                if new_rule.get_confidence() < (min_conf-0.005):
                    pass
                else:
                    rule_set.add_rule(new_rule)
                    ruleset_str.append(str(set_conditions))

    return rule_set

def read_rules_heuops_from_file(filename, dataset, goal_value):
    params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:,-1]
    params['POSITIVE_CLASS'] = goal_value
    params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)
    rule_set = RuleSet(params)

    with open(filename) as f:

        ruleset_str = []

        for line in f:

            if 'No rules' in line:
                continue

            set_conditions = []
            conditions = line.split('&')
            result = read_heuops_condition(conditions[0].strip())
            set_conditions.append(result)
            for i_cond in conditions[1:]:
                result = read_heuops_condition(i_cond.strip())
                set_conditions.append(result)

            if str(set_conditions) not in ruleset_str:
                new_rule = Rule(params)
                new_rule.set_consequent(goal_value)
                for i_cond in set_conditions:
                    try:
                        new_rule.add_condition('(' + get_str_for_evaluation(i_cond, dataset, 'x')+ ')')
                    except KeyError:
                        print('ERROR')
                        print(i_cond)
                        print(get_str_for_evaluation(i_cond, dataset, 'x'))
                        raise
                rule_set.add_rule(new_rule)
                ruleset_str.append(str(set_conditions))

    return rule_set




if __name__ == '__main__':
    print(read_rules_heuops_from_file('apriori_heuristics/results/rules_Datasets_australian_TrueFalse_forR.arff.csv'))
