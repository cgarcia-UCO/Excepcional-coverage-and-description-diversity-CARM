"""
This script is used to compute the performance metrics of the methods in the GECCO submission
from the rules produced by the algorithms.

This script also generate the graphs used in the submission
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import io
import re
import warnings
from os import path
from scipy.io.arff._arffread import ParseArffError
from scipy.io import arff
from zipfile import ZipFile
from tqdm import tqdm
import pickle
sys.path.insert(1, '.')
from alternatives_paper_GECCO.results2latex import results_CBA2_etal_2latex, results_FARCHD_2latex,\
    results_MOPNAR_NICGAR_QARCIP_2latex, results2latex_CN2_SD
from apriori_heuristics.results2latex import read_rules_heuops_from_file, read_rules_apriori_from_file
sys.path.insert(2, './python_code')
from utilities.representation.assoc_rules import RuleSet, Rule
from collections import Counter
import contextlib

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

# For printing in output with tqdm
class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, *args, **kwargs):
        kwargs['file'] = self.file
        kwargs['end'] = ''
        for i in args:
            tqdm.write(str(i), **kwargs)

    def __eq__(self, other):
        return other is self.file

    def flush(self):
        pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


# Global variable for compatibility with utilities.representation.assoc_rules package
class Dummy_class:
    pass

np.seterr(all='raise')
warnings.filterwarnings("error")

def develop_coordinates(tuples):
    result = []
    for i in tuples:
        for _ in range(i[1]):
            result.append(i[0])

    return result

def read_arff(file, binarize, params):
    """
    Read an arff dataset and returns the input variables and output ones

    :param file: Path to the file to be read
    :return: The parsed data contained in the dataset file in a tuple (input, output, metadata)
       . input contains the in features, and output the target, assumed it was the last one.
       . metadata contains information of the features (if categorical, numerical...)
    """

    try:  # arff.arffread fails in case the file has some special characters.
        data, metadata = arff.loadarff(file)
    except (UnicodeEncodeError, ParseArffError):
        with open(file, 'r') as f:
            content = ''.join(f.readlines())
            content = content.replace('á', 'a')
            content = content.replace('é', 'e')
            content = content.replace('í', 'i')
            content = content.replace('ó', 'o')
            content = content.replace('ú', 'u')
            content = content.replace('ñ', 'n')
            content = re.sub(r'@inputs .*', '', content)
            content = re.sub(r'@outputs .*', '', content)
            content = re.sub(r'@output .*', '', content)
            with io.StringIO(content) as f2:
                data_metadata = arff.loadarff(f2)
                data, metadata = data_metadata

    data = pd.DataFrame(data)

    for i, column in zip(data.dtypes, data.columns):

        if i == object:
            data[column] = data[column].str.decode('utf-8')
            missing = data[column] == '?'
            data.loc[missing, column] = np.nan
    num_in_features = data.shape[1] - 1
    input_data = data.iloc[:, :num_in_features]
    output = data.iloc[:, num_in_features]

    if binarize:
        counts = Counter(output)
        num_training_samples = len(output)
        new_training_out = np.full((num_training_samples,), False)
        min_positive_class = 0.1 * num_training_samples
        new_testing_out = None

        for i in reversed(counts.most_common()):
            new_training_out = new_training_out | ( output == i[0] )

            if sum(new_training_out) >= min_positive_class:
                break

        output = np.asarray(list(map(lambda x: str(x), new_training_out)))
        data.iloc[:,-1] = output

    return data, input_data, output, metadata

def identity(x):
    return x

def percent(x):
    return 100 * x

def remove_first(x):
    result = '\{'
    num_commas = 0
    for i in x:
        if num_commas >= 2 and i != '.' and i != ']':
            result +=i
        elif num_commas >= 2 and i == ']':
            result +='\}'
        if i == ',':
            num_commas += 1
    return result

def add_metrics(data, algorithm, dataset, metrics):

    if metrics['num_rules'] <= 0:
        data[algorithm][dataset]['num_rules'].append(0)
        return data

    for i in metrics:
        data[algorithm][dataset][i].append(metrics[i])

    return data

def compute_rules_metrics(ruleset, dataset, goal_value):
    rules_metrics = {}

    rules_metrics['num_rules'] = len(ruleset.rules)

    if rules_metrics['num_rules'] <= 0:
        return rules_metrics

    confidences = ruleset.get_confidence()
    supports = ruleset.get_support(dataset)
    feat_freq = ruleset.get_freq_attributes(dataset)
    uncovered = np.sum(ruleset.uncovered_target_patterns) /\
                np.sum((dataset.iloc[:,-1] == goal_value) |
                       ((dataset.iloc[:,-1] == goal_value.replace("'",''))))
    significances = ruleset.get_significances()
    mean_significance = np.mean(significances)
    mean_lift = np.mean(ruleset.get_lift_values())

    numConditions = develop_coordinates(ruleset.get_lengths())
    rules_metrics['num_conditions_mean'] = np.mean(numConditions)
    rules_metrics['num_conditions_std'] = np.std(numConditions)
    rules_metrics['coverage_min'] = np.min(supports) * 100
    rules_metrics['coverage_mean'] = np.mean(supports) * 100
    rules_metrics['global_coverage'] = (1-uncovered) * 100
    rules_metrics['confidence_min'] = 100 * np.min(confidences)
    rules_metrics['confidence_mean'] = 100 * np.mean(confidences)
    rules_metrics['ratio_used_features'] = ruleset.get_ratio_used_attributes()
    rules_metrics['max_feat_freq'] = np.max(feat_freq)
    rules_metrics['mean_significance'] = mean_significance
    rules_metrics['mean_lift'] = mean_lift
    wracc_values = ruleset.get_wracc_values()
    try:
        rules_metrics['mean_wracc'] = np.nanmean(wracc_values)
    except RuntimeWarning:
        for i in ruleset.rules:
            print(i)
        print(ruleset.get_wracc_values())
        raise
    return rules_metrics


def print_mean_std(values, key):
    try:
        metric = [np.mean(values[i][key]) for i in values if key in values[i] and len(values[i][key]) > 0]
        if len(metric) > 0:
            print('{:>3.0f}'.format(np.mean(metric)), '+-',
                  '{:<5.1f}'.format(np.std(metric)), sep='', end=' ')
        else:
            print('No values', end='')
        print(' & ', end=' ')
    except (FloatingPointError, TypeError):
        # print(values)
        raise
    except KeyError:
        raise

def boxplot(data, key, xlabels, new_order, logscale, suffix_graphfile=''):

    titles = {'num_rules': 'Number of rules (not many, not few)',
              'num_conditions_mean': 'Mean length of the rules (not long)',
              'num_conditions_std': 'Std(length of the rules)',
              'coverage_min': 'Min individual coverage (max)',
              'coverage_mean': 'Mean individual coverage (max)',
              'global_coverage': 'Overall coverage (max)',
              'confidence_min': 'Min confidence (max)',
              'confidence_mean': 'Mean confidence (max)',
              'ratio_used_features': 'Features coverage (max)',
              'max_feat_freq': 'Maximal feature dominance (min)',
              'mean_significance': 'Significance (max)',
              'mean_wracc': 'Unusualness (max)',
              'mean_lift': 'Lift (max)'}
    y_lims = {'num_conditions_mean': [0,10],
              'coverage_mean': [0,62],
              'confidence_mean': [40,100],
              'mean_lift': [0,5],
              'mean_significance': [0,300]}
    figsize = (2, 2)


    if suffix_graphfile == '_apriori':
        y_lims['num_rules']=[1,3000]
        y_lims['coverage_mean']=[0,30]

    try:
        current_data = [[np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
                         for i in data[i_method]]
                        for i_method in data]

    except RuntimeWarning:
        print('METRIC:', key, '\n', [[data[i_method][i][key] for i in data[i_method]]
                        for i_method in data], '\n')
        raise

    try:
        new_current_data = np.array(current_data).T
    except np.VisibleDeprecationWarning:
        print('METRIC:',key,'\n',current_data)
        raise

    current_data = new_current_data

    try:
        current_data = pd.DataFrame(current_data,
                                    columns=xlabels)
    except (ValueError,RuntimeWarning):
        print('METRIC:',key,'\n',current_data,'\n')
        raise

    if key in ['num_rules', 'global_coverage']:
        current_data[np.isnan(current_data)] = 0

    current_data = current_data[new_order]
    fig, ax = plt.subplots()

    if logscale:
        ax.set_yscale('log')
        ax.set_ylabel('log scale')

    if suffix_graphfile == '_apriori':
        plt.subplots_adjust(bottom=0.33, left=0.15)
    else:
        if key == 'num_rules':
            plt.subplots_adjust(bottom=0.33, left=0.15)
        else:
            plt.subplots_adjust(bottom=0.33)

    if key in y_lims:
        ax.set_ylim(y_lims[key])

    current_data.boxplot(ax=ax, grid=True,
                         # rot=-60,
                         figsize=figsize, boxprops={'color':'#1f77b4','linewidth':2},
                         medianprops={'linewidth':2, 'color':'#2ca02c'},
                         whiskerprops={'linewidth':2, 'color':'#1f77b4'},
                         capprops={'linewidth':2, 'color':'#1f77b4'})
    proposal_q1 = np.percentile(current_data['Proposal'], 25)
    proposal_q3 = np.percentile(current_data['Proposal'], 75)
    iqr = proposal_q3 - proposal_q1
    plt.fill_between(np.arange(len(xlabels))+0.5, np.repeat(proposal_q3, len(xlabels)),
                     np.repeat(proposal_q1, len(xlabels)), alpha=0.25, color='#1f77b4')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-60, ha="left", rotation_mode="anchor")
    plt.title(titles[key])

    if logscale:
        fig.savefig(str(key) + suffix_graphfile + '_log.pdf')
    else:
        fig.savefig(str(key) + suffix_graphfile + '.pdf')

    plt.close()

def print_global_metrics(metrics):
    print_mean_std(metrics, 'num_rules')

    print_mean_std(metrics, 'num_conditions_mean')

    print_mean_std(metrics, 'coverage_min')

    print_mean_std(metrics, 'coverage_mean')

    print_mean_std(metrics, 'global_coverage')

    print_mean_std(metrics, 'confidence_min')
    print_mean_std(metrics, 'confidence_mean')
    print_mean_std(metrics, 'ratio_used_features')
    print_mean_std(metrics, 'max_feat_freq')
    print_mean_std(metrics, 'mean_significance')
    print_mean_std(metrics, 'mean_wracc')
    print_mean_std(metrics, 'mean_lift')
    print('\\\\')

def print_tab_end():
    table_end = '</table>'

    print(table_end)

def print_tab_head():
    row_head = '<tr>' # ''
    row_end = '</tr>\n' # '\\\\\n'
    cell_head = '<th> ' # ''
    cell_end = ' </th> ' # ' &'
    table_head = '<table>'

    print(table_head)

    print(row_head,cell_head,'Alg',cell_end,
          cell_head, '{:>5}'.format('#rules'), cell_end,
          cell_head, '{:>13}'.format('conds'), cell_end,
          cell_head, '{:>13}'.format('I.cov:minmean'), cell_end,
          cell_head, '{:>6}'.format('G.cov'), cell_end,
          cell_head, '{:>13}'.format('Conf:minmean'), cell_end,
          cell_head, '{:>5}'.format('%attrs'), cell_end,
          cell_head, '{:>5}'.format('Attr.dom'), cell_end,
          cell_head, '{:>5}'.format('Sign'), cell_end,
          cell_head, '{:>5}'.format('WRAcc'), cell_end,
          cell_head, '{:>5}'.format('Lift'), cell_end, row_end,
          sep='', end='')

def print_metrics(alg_name, rules_metrics):

    row_head = '<tr>' # ''
    row_end = '</tr>\n' # '\\\\\n'
    cell_head = '<td> ' # ''
    cell_end = ' </td> ' # ' &'

    print(row_head, end='')
    print(cell_head, end='')
    print('{:>14} & '.format(alg_name), end='')
    print(cell_end, end='')
    print(cell_head, end='')

    if rules_metrics['num_rules'][0] <= 0:
        print('No rules', cell_end, row_end, end='', sep='')
        return

    print(cell_head, end='')
    print('{:>5.0f}'.format(np.mean(rules_metrics['num_rules'])), end=' ')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.0f}".format(np.mean(rules_metrics['num_conditions_mean'])),'+-',
          "{:<5.1f}".format(np.mean(rules_metrics['num_conditions_std'])), sep='', end='')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.0f}".format(np.mean(rules_metrics['coverage_min'])), ':',
          "{:<5.0f}".format(np.mean(rules_metrics['coverage_mean'])), sep='', end=' ')
    print(cell_end, end='')
    print(cell_head, end='')
    print('{:>6.1f}'.format(np.mean(rules_metrics['global_coverage'])), end='')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.0f}".format(np.mean(rules_metrics['confidence_min'])), ":",
          "{:<5.0f}".format(np.mean(rules_metrics['confidence_mean'])), sep='', end=' ')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.2f}".format(np.mean(rules_metrics['ratio_used_features'])), end='')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.0f}".format(np.mean(rules_metrics['max_feat_freq']) * 100), end='')
    print(cell_end, end='')
    print(cell_head, end='')
    print('{:>5.2f}'.format(np.mean(rules_metrics['mean_significance'])), end='')
    print(cell_end, end='')
    print(cell_head, end='')
    if np.isnan(np.nanmean(rules_metrics['mean_wracc'])):
        print(rules_metrics['mean_wracc'], end='')
        raise Exception
    else:
        print('{:>5.2f}'.format(np.nanmean(rules_metrics['mean_wracc'])), end='')
    print(cell_end, end='')
    print(cell_head, end='')
    if np.isnan(np.nanmean(rules_metrics['mean_lift'])):
        print(rules_metrics['mean_lift'], end='')
        raise Exception
    else:
        print('{:>5.2f}'.format(np.nanmean(rules_metrics['mean_lift'])), end='')
    print(cell_end, end='')
    print(row_end, end='')


def results_best_conf_proposal(dominance, name_dataset, data, name_dataset_v2, params, dataset):
    goal_value = params['POSITIVE_CLASS']
    rule_set = RuleSet(params)

    def read_rules(text):
        lines = text.split('\n')
        rules = RuleSet(params)

        for i in lines:
            if i == '':
                continue
            conditions = i.split(' => ')[0].split('&')
            new_rule = Rule(params)
            new_rule.set_consequent(goal_value)
            for i_conds in conditions:
                try:
                    new_rule.add_condition(i_conds)
                except SyntaxError:
                    print('{',i_conds,'}:',conditions,':',i,'\n',lines)
                    raise
            rules.add_rule(new_rule)
        return rules

    try:
        with ZipFile(path.join('best_conf',name_dataset+'.zip'), 'r') as zipObj:
            files = [i for i in zipObj.namelist() if 'assoc_rules_'+dominance+'.txt' in i]
            for i in files:
                text = zipObj.read(i).decode(encoding="utf-8")
                rules = read_rules(text)
                metrics = compute_rules_metrics(rules, dataset, goal_value)
                data = add_metrics(data, dominance.upper(), name_dataset_v2, metrics)
            print_metrics(dominance.upper(), data[dominance.upper()][name_dataset_v2])
    except FileNotFoundError:
        print('No files')
        raise

    return data

def rules_to_metrics(methods, metrics_v2):
    params = {'FITNESS_FUNCTION': Dummy_class()}
    params['FITNESS_FUNCTION'].training_exp = None
    params['MIN_TARGET_CLASS_RATIO'] = 0.1
    datasets = ['australian', 'balance', 'car','chess','german','glass','heart','hepatitis','iris','tic-tac-toe',
                'wine', 'wisconsin',
                'ionosphere', 'pima', 'vowel', 'data.arff']

    data = {i:{} for i in methods}

    # INITIATE A CONTAINER FOR THE RESULTS
    for i in data:
        for j in datasets:
            data[i][j] = {}
            for k in metrics_v2:
                data[i][j][k] = []

    #######################
    # Parse the rules per method and print the metrics
    ########################

    with open('results_per_dataset.html', 'w') as f:
        with contextlib.redirect_stdout(f):
            with nostdout():
                with tqdm(total=len(datasets) * len(methods)) as pbar:
                    try:
                        for i_dataset_CBA_andcia, i_dataset_Heu_ops, i_dataset_proposals,\
                                i_dataset_v2, goal_feature, goal_value, whole_dataset,\
                                rules_file_dataset_MOPNAR_andcia, binarize in\
                                    zip(['australian_truefalse_c','balance_truefalse_c','car_truefalse_c','chess_truefalse_c',
                                         'german_truefalse_c','glass_truefalse_c','heart_truefalse_c','hepatitis_truefalse_c',
                                         'iris_truefalse_c','tic-tac-toe_truefalse_c','wine_truefalse_c','wisconsin_truefalse_c',
                                         'ionos_imba', 'pima_imba', 'vowel0_imba','complicaciones_filtered'],
                                        ['australian_truefalse_c', 'balance_truefalse_c', 'car_truefalse_c', 'chess_truefalse_c',
                                         'german_truefalse_c', 'glass_truefalse_c', 'heart_truefalse_c', 'hepatitis_truefalse_c',
                                         'iris_truefalse_c', 'tic-tac-toe_truefalse_c', 'wine_truefalse_c', 'wisconsin_truefalse_c',
                                         'ionosphere.arff.csv', 'pima.arff.csv', 'vowel0_forR.arff.csv', 'data.arff.csv'],
                                        ['australian', 'balance', 'car', 'chess',
                                         'german', 'glass', 'heart', 'hepatitis',
                                         'iris', 'tic-tac-toe', 'wine', 'wisconsin',
                                         'ionos_imba', 'pima_imba', 'vowel0_imba', 'complicaciones_filtered'],
                                        datasets,
                                        ['Class','Balance_scale','Acceptability','Class','Customer','TypeGlass','Class','Class','Class',
                                         'Class','Class','Class',
                                         'class','class','Class','COMPLICACIONES'],
                                        ['True','True','True','True','True','True','True','True','True','True','True','True',
                                         'b', 'tested_positive', 'positive', "'Si'"],
                                        ['datasets/australian.arff',
                                         'datasets/balance.arff',
                                         'datasets/car_TrueFalse.arff',
                                         'datasets/chess_TrueFalse.arff',
                                         'datasets/german_TrueFalse.arff',
                                         'datasets/glass_TrueFalse.arff',
                                         'datasets/heart_TrueFalse.arff',
                                         'datasets/hepatitis_TrueFalse.arff',
                                         'datasets/iris_TrueFalse.arff',
                                         'datasets/tic-tac-toe_TrueFalse.arff',
                                         'datasets/wine_TrueFalse.arff',
                                         'datasets/wisconsin_TrueFalse.arff',
                                         'datasets/ionosphere.arff',
                                         'datasets/pima.arff',
                                         'datasets/vowel0.arff',
                                         'datasets/data.arff'],
                                        ['australian_truefalse',
                                         'balance_truefalse',
                                         'car_truefalse',
                                         'chess_truefalse',
                                         'german_truefalse',
                                         'glass_truefalse',
                                         'heart_truefalse',
                                         'hepatitis_truefalse',
                                         'iris_truefalse',
                                         'tic-tac-toe_truefalse',
                                         'wine_truefalse',
                                         'wisconsin_truefalse',
                                         'assoc_ionosphere',
                                         'assoc_pima',
                                         'assoc_vowel0',
                                         'association_complicaciones_filtered'],
                                        [True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         True,
                                         False,
                                         False,
                                         False,
                                         False]):

                            print('\n--------------------------------')
                            print(i_dataset_v2)
                            try:
                                dataset, _, _, _ = read_arff(whole_dataset, binarize, params)
                            except FileNotFoundError:
                                print('Dataset', whole_dataset, 'is not present. Particularly, we had no permission to distribute the colorectal dataset')
                                continue
                            params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:, -1]
                            params['POSITIVE_CLASS'] = goal_value
                            params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)

                            print_tab_head()

                            #### Apriori
                            for i_conf, i_support in zip([0.7, 0.8, 0.9], [0.05, 0.025, 0.0125]):
                                try:
                                    rules = read_rules_apriori_from_file('apriori_heuristics/results/rules_apriori_' +
                                                                         str(i_conf) + '_' + str(i_support) + 'Datasets_' +
                                                                         i_dataset_Heu_ops.replace('truefalse_c',
                                                                                                   'TrueFalse_forR.arff.csv'),
                                                                         i_conf, dataset, goal_value, params)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, 'Apriori_' + str(i_conf)+'_'+str(i_support), i_dataset_v2, rules_metrics)
                                    print_metrics('Apriori_' + str(i_conf)+'_'+str(i_support),
                                                  data['Apriori_' + str(i_conf)+'_'+str(i_support)][i_dataset_v2])

                                except FileNotFoundError:
                                    rules = None
                                    raise Exception('Apriori & No files apriori '+str(i_conf)+' '+
                                                    i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                                pbar.update(1)

                            #### Heuristic Operators
                            try:
                                rules = read_rules_heuops_from_file('apriori_heuristics/results/rules_Datasets_' +
                                                             i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'),
                                                             dataset, goal_value)
                                rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                data = add_metrics(data, 'Heu.Ops', i_dataset_v2, rules_metrics)
                                print_metrics('Heu.Ops', data['Heu.Ops'][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('Heu.Ops & No files for:', 'apriori_heuristics/results/rules_Datasets_' +
                                                             i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                            pbar.update(1)

                            for i_alg, alg_name in zip(['CBA2', 'CPAR'], ['CBA2', 'CPAR']):
                                try:
                                    rules = results_CBA2_etal_2latex(i_dataset_CBA_andcia, goal_value, i_alg, dataset)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)
                                    print_metrics(alg_name, data[i_alg][i_dataset_v2])
                                except FileNotFoundError:
                                    rules = None
                                    print('No files')
                                    raise
                                pbar.update(1)

                            try:
                                rules = results_FARCHD_2latex(i_dataset_CBA_andcia, goal_value, dataset)
                                rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                data = add_metrics(data, 'FARCHD', i_dataset_v2, rules_metrics)
                                print_metrics('FARCHD', data['FARCHD'][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('No files')
                                raise
                            pbar.update(1)

                            for i_alg in ['NICGAR','MOPNAR', 'QAR_CIP_NSGAII']:
                                try:
                                    rules = results_MOPNAR_NICGAR_QARCIP_2latex(i_alg, rules_file_dataset_MOPNAR_andcia, goal_feature,
                                                                                goal_value, dataset)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)
                                    print_metrics(i_alg, data[i_alg][i_dataset_v2])
                                except FileNotFoundError:
                                    rules = None
                                    print('No files')
                                    raise
                                pbar.update(1)

                            for i_conf in range(1):
                                try:
                                    rules = results2latex_CN2_SD(i_dataset_CBA_andcia,goal_feature, goal_value,'CN2-SD', i_conf, dataset)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, 'CN2_SD', i_dataset_v2, rules_metrics)
                                    print_metrics('CN2_SD', data['CN2_SD'][i_dataset_v2])
                                except FileNotFoundError:
                                    rules = None
                                    print('No files')
                                    raise
                                pbar.update(1)

                            # print('UNCOMMENT PROPOSAL RESULTS')
                            data = results_best_conf_proposal('nd2', i_dataset_proposals, data, i_dataset_v2, params, dataset)
                            pbar.update(1)

                            print_tab_end()
                            sys.stdout.flush()

                        if path.exists('metrics.pickle'):
                            import shutil
                            shutil.copyfile('metrics.pickle', 'metrics_previous.pickle')

                        with open('metrics.pickle', 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        return data

                    except KeyboardInterrupt:
                        print('Process interrupted')
                        sys.exit(0)

def datasets_properties():
    file_dataset = {'australian':'datasets/australian.arff',
                    'balance':'datasets/balance.arff',
                    'car':'datasets/car.arff',
                    'chess':'datasets/chess.arff',
                    'german':'datasets/german.arff',
                    'glass':'datasets/glass.arff',
                    'heart':'datasets/heart.arff',
                    'hepatitis':'datasets/hepatitis.arff',
                    'ionosphere':'datasets/ionosphere.arff',
                    'iris':'datasets/iris.arff',
                    'pima':'datasets/pima.arff',
                    'tic-tac-toe':'datasets/tic-tac-toe.arff',
                    'vowel':'datasets/vowel0.arff',
                    'wine':'datasets/wine.arff',
                    'wisconsin':'datasets/wisconsin.arff',
                    'colorrectal':'datasets/data.arff'}
    binarize = {'australian': True,
                'balance': True,
                'car': True,
                'chess': True,
                'german': True,
                'glass': True,
                'heart': True,
                'hepatitis': True,
                'ionosphere': False,
                'iris': True,
                'pima': False,
                'tic-tac-toe': True,
                'vowel': False,
                'wine': True,
                'wisconsin': True,
                'colorrectal': False}

    for i in sorted(file_dataset):
        min_ratio = 0
        if binarize[i]:
            min_ratio = 0.1
        _, data_in, data_out, metadata = read_arff(file_dataset[i], binarize[i], min_ratio)

        num_features = len(list(metadata))
        for i_metadata, key in enumerate(metadata):
            if i_metadata == num_features - 1:
                num_classes = len(metadata[key][1])

        diff_values = []
        count_categorical = 0
        count_numeric = 0
        for _, i_metadata in zip(range(len(list(metadata)) - 1), metadata):
            if metadata[i_metadata][0] == 'numeric':
                count_numeric +=1
            elif metadata[i_metadata][0] == 'nominal':
                count_categorical += 1
                diff_values.append(len(metadata[i_metadata][1]))
            else:
                raise ValueError('Weird feature type: ' + str(metadata[i_metadata]))

        try:
            num_missing = np.sum(np.sum(data_in.isna()))
        except TypeError:
            print(data_in)
            raise

        print(i, '&',
              data_in.shape[0], '&',
              data_in.shape[1], '&',
              count_categorical, '&',
              str(min(diff_values))+'/'+'{:.0f}'.format(np.mean(diff_values))+'/'+str(max(diff_values)) if count_categorical > 0 else '-/-/-', '&',
              num_classes, '&',
              '{:.0f}\\%'.format(min(list(Counter(data_out).values())) / len(data_out) * 100), '\\\\')

def wilcox_heatmap(data, metrics, metric_labels, methods, method_labels):
    warnings.filterwarnings("default")
    from scipy import stats

    min_max={'num_rules':'max',
             'num_conditions_mean':'min',
             'coverage_mean':'max',
             'global_coverage':'max',
             'confidence_mean':'max',
             'ratio_used_features':'max',
             'max_feat_freq':'min',
             'mean_significance':'max',
             'mean_wracc':'max',
             'mean_lift':'max'}

    heatmap_data = []
    proposal = 'ND2'

    for i_metric in metrics:
        heatmap_data.append([])

        data_j = np.array([np.nanmean(data[proposal][i_dataset][i_metric])
                           if i_metric in data[proposal][i_dataset] and
                              len(data[proposal][i_dataset][i_metric]) > 0 else np.nan
                           for i_dataset in data[proposal]])

        for i_dataset in data[proposal]:
            assert i_metric in data[proposal][i_dataset]

        for i_method in methods:

            data_i = np.array([np.nanmean(data[i_method][i_dataset][i_metric])
                               if i_metric in data[i_method][i_dataset] and
                                  len(data[i_method][i_dataset][i_metric]) > 0 else np.nan
                               for i_dataset in data[i_method]])


            try:
                result = stats.wilcoxon(data_i, data_j, nan_policy='omit')
                if (np.nanmean(data_i) < np.nanmean(data_j) and min_max[i_metric] == 'max') or \
                        (np.nanmean(data_i) > np.nanmean(data_j) and min_max[i_metric] == 'min'):
                    heatmap_data[-1].append(result.pvalue)
                else:
                    heatmap_data[-1].append(-1*result.pvalue)
            except (ValueError, TypeError):
                print(data_i)
                print(data_j)
                raise

    heatmap_data = np.asarray(heatmap_data)
    heatmap_data_image = heatmap_data.copy()
    heatmap_data_image[heatmap_data < 0] = -1 - 3*heatmap_data_image[heatmap_data < 0]
    heatmap_data_image[heatmap_data > 0] = 1 - 3*heatmap_data_image[heatmap_data > 0]
    heatmap_data_image[np.abs(heatmap_data) > 0.15] = 0

    fig, ax = plt.subplots(figsize=(4.3,3.6))
    im = ax.imshow(heatmap_data_image, cmap='coolwarm_r')
    ax.set_xticks(np.arange(len(methods)), labels=method_labels, fontsize=8)
    ax.set_yticks(np.arange(len(metrics)), labels=metric_labels, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(metrics)):
        for j in range(len(methods)):
            color = 'black' if abs(heatmap_data[i,j]) > 0.1 else 'w'
            ax.text(j,i, '{:.2f}'.format(heatmap_data[i][j]).replace('0.','.'),
            ha='center', va='center', color=color,
            fontsize='7')
    ax.set_title('Wilcoxon p-values', fontsize=12)
    plt.subplots_adjust(bottom=0.18, left=0.25, right=0.99)
    plt.savefig('heatmap_wilcox.pdf')
    plt.close()
    warnings.filterwarnings("error")


if __name__ == '__main__':

    metrics_v2 = ['num_rules', 'num_conditions_mean', 'num_conditions_std',
                  'coverage_min', 'coverage_mean', 'global_coverage', 'confidence_min', 'confidence_mean',
                  'ratio_used_features', 'max_feat_freq', 'mean_significance', 'mean_wracc', 'mean_lift']
    methods = ['Apriori_0.7_0.05','Apriori_0.8_0.025','Apriori_0.9_0.0125','Heu.Ops','CBA2','CPAR','FARCHD','ND2','NICGAR','MOPNAR', 'QAR_CIP_NSGAII', 'CN2_SD']
    methods = ['Apriori_0.7_0.05', 'Apriori_0.8_0.025', 'Apriori_0.9_0.0125', 'Heu.Ops', 'CBA2', 'CPAR', 'FARCHD',
               'ND2', 'NICGAR', 'MOPNAR', 'QAR_CIP_NSGAII', 'CN2_SD']

    data = None

    if path.exists('metrics_algs_original.pickle'):
        answer = 'p'

        while answer not in ['y', 'n']:
            print('I have detected the file with the performance metrics computed (metrics_algs_original.pickle)')
            print('I can use it to generate the graphs "quickly", but nevertheless,')
            print('do you want to re-compute the metrics from the rules? (This takes hours; y/n)')
            answer = input('')

            if answer == 'y':
                data = rules_to_metrics(methods, metrics_v2)
            elif answer == 'n':
                print('I am using the previously computed performance values (metrics_algs_original.pickle)')
                with open('metrics_algs_original.pickle', 'rb') as handle:
                    data = pickle.load(handle)
            else:
                print('Not valid answer: ', answer)
    else:
        data = rules_to_metrics(methods, metrics_v2)


    print('\n------------------------\nGLOBAL RESULTS')
    print('                 ',
          '{:>11}'.format('#rules'), ' & ',
          '{:>12}'.format('conds'), ' & ',
          '{:>12}'.format('I.min_cov'), ' & ',
          '{:>12}'.format('I.mean_cov'), ' & ',
          '{:>12}'.format('G.coverage'), ' & ',
          '{:>12}'.format('min_conf'), ' & ',
          '{:>12}'.format('mean_conf'), ' & ',
          '{:>12}'.format('%used_attrs'), ' & ',
          '{:>12}'.format('Attr.dom'), ' & ',
          '{:>12}'.format('Significance'), ' & ',
          '{:>12}'.format('WRAcc'), ' & ',
          '{:>12}'.format('Lift'),
          sep='')
    for i in methods:
        print("{:>14}".format(i), '&', end=' ')
        print_global_metrics(data[i])

    print('\n-------------------------\nGENERATING PLOTS')
    logscale_for = ['num_rules']
    xlabels = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','Heu.Ops', 'CBA2', 'CPAR', 'FARCHD', 'Cons.D', 'Proposal',
               'NICGAR', 'MOPNAR', 'QAR-CIP', 'CN2-SD']
    new_order = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
               'NICGAR','MOPNAR','QAR-CIP','CN2-SD','Heu.Ops','Cons.D','Bold.D']
    new_order = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
               'NICGAR','MOPNAR','QAR-CIP','CN2-SD','Heu.Ops','Proposal']
    for i in tqdm(metrics_v2):
        if i in logscale_for:
            boxplot(data, i, xlabels, new_order, logscale=True)
        else:
            boxplot(data, i, xlabels, new_order, logscale=False)

    wilcox_heatmap(data, metrics=['num_conditions_mean', 'ratio_used_features', 'max_feat_freq',
                                  'coverage_mean', 'global_coverage', 'confidence_mean',
                                  'mean_lift', 'mean_significance', 'mean_wracc'],
                   metric_labels=['Rule length', '#Features', 'Feat. dominance',
                                  'Ind. coverage', 'Overall coverage', 'Confidence',
                                  'Lift', 'Significance', 'Unusualness'],
                   methods=['Apriori_0.7_0.05', 'Apriori_0.8_0.025', 'Apriori_0.9_0.0125', 'CBA2', 'CPAR', 'FARCHD',
                    'NICGAR', 'MOPNAR', 'QAR_CIP_NSGAII', 'CN2_SD','Heu.Ops'],
                   method_labels=['Ap(70,10/2)', 'Ap(80,10/4)', 'Ap(90,10/8)', 'CBA2', 'CPAR', 'FARCHD',
                    'NICGAR', 'MOPNAR', 'QAR-CIP', 'CN2-SD','Heu. Ops'])