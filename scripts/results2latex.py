"""
This script is used to compute the performance metrics of the methods
from the rules produced by the algorithms.

This script also generate the graphs used in the submission
"""
import datetime

import matplotlib
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
import matplotlib.transforms as transforms
sys.path.insert(1, '.')
from alternatives_paper.results2latex import results_CBA2_etal_2latex, results_FARCHD_2latex,\
    results_MOPNAR_NICGAR_QARCIP_2latex, results2latex_CN2_SD, results_NMEEF_MESDIF_2latex
from apriori_heuristics.results2latex import read_rules_heuops_from_file, read_rules_apriori_from_file
sys.path.append(os.path.join(os.getcwd(), "python_code"))
#sys.path.insert(2, './python_code')
from utilities.representation.assoc_rules import RuleSet, Rule
from collections import Counter
import contextlib

# For preventing writting in sys.output
class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, *args, **kwargs):
        kwargs['file'] = self.file
        kwargs['end'] = ''
        # for i in args:
        #     tqdm.write(str(i), **kwargs)

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

# IMPORTING OR INSTALLING THE SCOTT-KNOTT STATISTICAL TEST
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r, pandas2ri
    import rpy2.robjects as ro

    r['options'](warn=-1)
    r_installed = True
except:
    r_installed = False
    print('\n**************************************************************')
    print('PACKAGE MISSING (R or rpy2). I will generate the graphs without the Scott-Knott-ESD test results\n')
    print('**************************************************************')
    print('')
    print('Press enter')
    _ = input('')

def import_or_install_ScottKnott(r_installed):
    sk = None

    if r_installed:
        try:
            sk = importr("ScottKnottESD")
        except:
            try:
                sk = importr("reshape2", lib_loc='R_local_site_library')
                sk = importr("effsize", lib_loc='R_local_site_library')
                sk = importr("carData", lib_loc='R_local_site_library')
                sk = importr("car", lib_loc='R_local_site_library')
                sk = importr("ScottKnottESD", lib_loc='R_local_site_library')
            except:
                try:
                    not_installed = []
                    try:
                        import apt
                        cache = apt.Cache()

                        for i_tool in ["cmake", "libcurl4-openssl-dev", "gfortran", "libblas-dev", "liblapack-dev", "libpcre2-dev", "libbz2-dev", "zlib1g-dev", "liblzma-dev", "r-base", "r-base-core"]:
                            try:
                                if not cache[i_tool].is_installed:
                                    not_installed.append(i_tool)
                            except:
                                not_installed.append(i_tool)
                    except:
                        print('\n**************************************************************')
                        print("PACKAGE MISSING: ScottKnottESD. I would need python3-apt installed to check if I can try to install it")
                        not_installed = []
                        raise Exception

                    if len(not_installed) > 0:
                        print('\n**************************************************************')
                        print("PACKAGE MISSING: ScottKnottESD. I can not try to install it because I need, at least, the following packages (sudo apt install ...):")
                        print(not_installed)
                        raise Exception

                    print('\n**************************************************************')
                    print('PACKAGE MISSING (ScottKnottESD). Should I try to install it locally (R_local_site_library)?\n'
                          'This takes time (30 minutes in my case) and may fail. Should I try to install it? (y/n)')
                    char_read = input('')

                    while char_read not in ['y','n']:
                        print('Invalid option:', char_read)
                        print('\n**************************************************************')
                        print('PACKAGE MISSING (ScottKnottESD). Should I try to install it locally (R_local_site_library)?\n'
                              'This takes time (30 minutes in my case) and may fail. Should I try to install it? (y/n)')
                        char_read = input('')

                    if char_read == 'y':
                        try:
                            #installation of scott-knott thourgh python
                            import rpy2.robjects.packages as rpackages
                            from rpy2.robjects.vectors import StrVector  # R vector of strings
                            from rpy2.robjects.constants import TRUE

                            utils = rpackages.importr("utils")
                            #utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

                            # R package names
                            packnames = ["tseries", "forecast", "reshape2", "ScottKnottESD"]

                            # Selectively install what needs to be installed.
                            names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
                            print(f"packages to install: {names_to_install}")

                            if not os.path.exists('R_local_site_library'):
                                os.mkdir('R_local_site_library')

                            if len(names_to_install) > 0:
                                utils.install_packages(StrVector(names_to_install), lib='R_local_site_library', dependencies=TRUE)
                                sk = importr("ScottKnottESD", lib_loc='R_local_site_library')
                        except Exception as e:
                            print('I could not install the ScottKnottESD package.\n')
                            print("I will generate the graphs without the Scott-Knott-ESD test results.\n")
                            print('**************************************************************\n')
                            r_installed = False
                    else:
                        print("I will generate the graphs without the Scott-Knott-ESD test results.\n")
                        print('**************************************************************\n')
                        r_installed = False
                except:
                    print('PACKAGE MISSING (ScottKnottESD). I can not install it')
                    print("I will generate the graphs without the Scott-Knott-ESD test results.")
                    print('**************************************************************\n')
                    r_installed = False

    return sk, r_installed

sk, r_installed = import_or_install_ScottKnott(r_installed)
# END IMPORTING OR INSTALLING THE SCOTT-KNOTT STATISTICAL TEST


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

path_generated_figures='generated_results'

if not os.path.exists(path_generated_figures):
    os.mkdir(path_generated_figures)

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

seeds = [23456781, 23187654, 12345678, 34567812, 45678123, 56781234, 67812345, 78123456, 81234567,
         21345678, 32456781, 43567812, 54678123, 65781234, 76812345, 87123456, 18234567,
         87654321, 76543218, 65432187, 54321876, 43218765, 32187654, 21876543, 18765432,
         78654321, 67543218, 56432187, 45321876, 34218765]

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
        # This happened when none of the rules covered any pattern
        rules_metrics['mean_wracc'] = 0
        # for i in ruleset.rules:
        #     print(i)
        # print(ruleset.get_wracc_values())
        # raise
    return rules_metrics


def compute_ScottKnott_ranks(data, key, max_or_min):
    try:
        current_data = [[np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
                         for i in data[i_method]]
                        for i_method in data]# if i_method not in ['Apriori0.7','Apriori0.8','Apriori0.9']]

    except RuntimeWarning:
        print('METRIC:', key, '\n', [[data[i_method][i][key] for i in data[i_method]]
                        for i_method in data], '\n')
        raise

    try:
        new_current_data = np.array(current_data).T
    except np.VisibleDeprecationWarning:
        print('METRIC:',key,'\n',current_data)
        raise
    # if new_current_data.shape[0] == 0:
    #     print(current_data)
    current_data = new_current_data

    try:
        current_data = pd.DataFrame(current_data,
                                    columns=xlabels)
    except (ValueError,RuntimeWarning):
        print('METRIC:',key,'\n',current_data.shape,'\n')
        intent = [ (i_method,
            [np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
             for i in data[i_method]])
            for i_method in data]
        print(intent,'\n',xlabels, flush=True)
        raise

    # current_data = current_data[new_order]

    if key in ['num_rules', 'global_coverage', 'ratio_used_features']:
        current_data[np.isnan(current_data)] = 0
        # current_data.fillna(0)
        # print(current_data)
        aux_current_data = current_data
    elif key in ['max_feat_freq', 'num_conditions_mean', 'confidence_mean', 'confidence_min', 'mean_lift',
                 'mean_significance', 'mean_wracc', 'coverage_mean']:
        current_data[np.isnan(current_data)] = np.nan
        # current_data.fillna(0)
        # print(current_data)
        aux_current_data = current_data
    else:
        aux_current_data = current_data.copy()
        for i in aux_current_data.columns:
            try:
                new_value = np.nanmean(aux_current_data[i])
            except RuntimeWarning:
                new_value = 0

            if np.isnan(new_value):
                # print('--------------------------------')
                # print(new_value, np.isnan(new_value))
                aux_current_data[i].fillna(new_value, axis='rows', inplace=True)
            else:
                print("WHAAAAAAAAAAAAAAAAAAAAAT??????", key)
                print(aux_current_data[i])
                print(aux_current_data[i].isna())
                sys.exit(1)

    # Non-parametric ScottKnott-ESD
    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(aux_current_data)

    # print('----------------------------------')
    # print(current_data.shape)
    # print(r_from_pd_df)
    # print('----------------------------------')

    r_sk = sk.sk_esd(r_from_pd_df, version='np')
    column_order = list(np.asarray(r_sk[3]) - 1)
    ranks = pd.DataFrame(
        # [r_sk[1].astype("float")], columns=[data.columns[i] for i in column_order]
        [r_sk[1]], columns=[current_data.columns[i] for i in column_order]
    )  # wide format

    if max_or_min == 'min':
        ranks = ranks.max().max() - ranks + 1

    for i in current_data.columns:
        if i not in ranks.columns:
            ranks[i] = np.nan

    # print(ranks, flush=True, file=sys.stderr)
    # sys.exit(1)
    ranks = ranks[current_data.columns]
    return ranks

def boxplot(data, key, xlabels, new_order, max_or_min='max', logscale=False, suffix_graphfile='', proposal_new_label = 'Proposal'):
    warnings.filterwarnings("ignore")
    # pandas2ri.activate()
    # if r_installed:
    #     sk = importr("ScottKnottESD")

    # for i_method in data:
    #     [data[i_method][i][key] for i in data[i_method]]
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
              'mean_significance': [0,300],
              'num_rules': [0.5,1000]}
    figsize = (2, 2)


    if suffix_graphfile == '_apriori':
        y_lims['num_rules']=[1,3000]
        y_lims['coverage_mean']=[0,30]

    ########################3
    # Load the data
    try:
        current_data = [[np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
                         for i in data[i_method]]
                        for i_method in data]# if i_method not in ['Apriori0.7','Apriori0.8','Apriori0.9']]

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

    suppossed_xlabels = data.keys()
    #assert suppossed_xlabels == xlabels, '\n' + str(suppossed_xlabels) + '\n != \n' + str(xlabels)

    try:
        current_data = pd.DataFrame(current_data,
                                    columns=xlabels)
    except (ValueError,RuntimeWarning):
        print('METRIC:',key,'\n',current_data.shape,'\n')
        intent = [ (i_method,
            [np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
             for i in data[i_method]])
            for i_method in data]
        print(intent,'\n',xlabels, flush=True)
        raise

    current_data = current_data[new_order]

    if key in ['num_rules', 'global_coverage', 'ratio_used_features']:
        current_data[np.isnan(current_data)] = 0
        # current_data.fillna(0)
        # print(current_data)
        aux_current_data = current_data
    elif key in ['max_feat_freq', 'num_conditions_mean', 'confidence_mean', 'confidence_min', 'mean_lift',
                 'mean_significance', 'mean_wracc', 'coverage_mean']:
        current_data[np.isnan(current_data)] = np.nan
        # current_data.fillna(0)
        # print(current_data)
        aux_current_data = current_data
    else:
        aux_current_data = current_data.copy()
        for i in aux_current_data.columns:
            try:
                new_value = np.nanmean(aux_current_data[i])
            except RuntimeWarning:
                new_value = 0

            if np.isnan(new_value):
                # print('--------------------------------')
                # print(new_value, np.isnan(new_value))
                aux_current_data[i].fillna(new_value, axis='rows', inplace=True)
            else:
                print("WHAAAAAAAAAAAAAAAAAAAAAT??????", key)
                print(aux_current_data[i])
                print(aux_current_data[i].isna())
                sys.exit(1)
        # sys.exit(1)

    ##################################
    # Non-parametric ScottKnott-ESD
    if r_installed:
        with (ro.default_converter + pandas2ri.converter).context():
            r_from_pd_df = ro.conversion.get_conversion().py2rpy(aux_current_data)

    # print('----------------------------------')
    # print(current_data.shape)
    # print(r_from_pd_df)
    # print('----------------------------------')

    if r_installed:
        r_sk = sk.sk_esd(r_from_pd_df, version='np')
        column_order = list(np.asarray(r_sk[3]) - 1)
        # print(current_data.shape, aux_current_data.shape, len(column_order), '\n', aux_current_data, '\n', column_order)
        ranks = pd.DataFrame(
            # [r_sk[1].astype("float")], columns=[data.columns[i] for i in column_order]
            [r_sk[1]], columns=[current_data.columns[i] for i in column_order]
        )  # wide format

    if r_installed:
        if max_or_min == 'min':
            ranks = ranks.max().max() - ranks + 1
            column_order = column_order[::-1]



    #################################
    # PLOTTING
    fig, ax = plt.subplots()

    if logscale:
        ax.set_yscale('log')
        ax.set_ylabel('log scale')
        ax.set_ylim(max(0.00001,current_data.min().min()), current_data.max().max())

    if suffix_graphfile == '_apriori':
        plt.subplots_adjust(bottom=0.33, left=0.15)
    else:
        if key == 'num_rules':
            plt.subplots_adjust(bottom=0.33, left=0.15)
        else:
            plt.subplots_adjust(bottom=0.33)

    if key in y_lims:
        try:
            ax.set_ylim(y_lims[key])
        except:
            print(key, y_lims[key])
            raise

    # Sort plots according to the Scott-Knott order
    if r_installed:
        copy_column_order = column_order.copy()
        # print(key, '\n', column_order, '\n', sorted(copy_column_order))
        current_data = pd.DataFrame(
            {current_data.columns[i]: current_data.iloc[:,i] for i in column_order}
        )

    current_data.boxplot(ax=ax, grid=True,
                         # showmeans=True,
                         # rot=-60,
                         figsize=figsize, boxprops={'color':'#1f77b4','linewidth':2},
                         medianprops={'linewidth':2, 'color':'#2ca02c'},
                         whiskerprops={'linewidth':2, 'color':'#1f77b4'},
                         capprops={'linewidth':2, 'color':'#1f77b4'})
    # ax.set_xticklabels(new_order, ha='left')

    ###########################
    # PRINT THE SCOTT-KNOTT RANKS
    if r_installed:
        current_pos = 0.25
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        y_lim = ax.get_ylim()
        for i in np.unique(ranks):
            num_algs_sharing_rank = np.sum(np.sum(ranks == i))
            if i >= 10:
                text_correction = -0.25
                fontsize = 16
            else:
                fontsize = 22
                text_correction = 0


            if i % 2 == 0:
                if key != 'num_rules':
                    plt.text((current_pos + (num_algs_sharing_rank-1)/2+text_correction)/ len(column_order), 1.05,
                             str(int(i)), transform=ax.transAxes, alpha=0.5, fontsize=fontsize)
                x_values = np.array([current_pos,current_pos+num_algs_sharing_rank])+0.25
                plt.fill_between(x_values, [1,1],[0,0],
                                 alpha=0.1, color='#000000', transform=trans)
            else:
                if key != 'num_rules':
                    plt.text((current_pos + (num_algs_sharing_rank - 1) / 2 +text_correction) / len(column_order), 1.05,
                             str(int(i)), transform=ax.transAxes, alpha=0.5, fontsize=fontsize)
            current_pos += num_algs_sharing_rank

        ax.set_ylim(y_lim)

    # Band covering the first and third quartile of the proposal
    # if proposal_new_label in current_data:
    #     proposal_q1 = np.percentile(current_data[proposal_new_label], 25)
    #     proposal_q3 = np.percentile(current_data[proposal_new_label], 75)
    #     iqr = proposal_q3 - proposal_q1
    #     aux = current_data[proposal_new_label][(current_data[proposal_new_label] <= (proposal_q3 + 1.5 * iqr)) &
    #                                      (current_data[proposal_new_label] >= (proposal_q1 - 1.5 * iqr))]
    #     plt.fill_between(np.arange(len(xlabels)+1)+0.5, np.repeat(proposal_q3, len(xlabels)+1),
    #                      np.repeat(proposal_q1, len(xlabels)+1), alpha=0.25, color='#1f77b4')
    #     # plt.fill_between(np.arange(len(xlabels))+0.5, np.repeat(max(aux), len(xlabels)),
    #     #                  np.repeat(min(aux), len(xlabels)), alpha=0.1, color='#1f77b4')

    # if key in y_lims:
    #     try:
    #         ax.set_ylim(y_lims[key])
    #     except:
    #         print(key, y_lims[key])
    #         raise

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-60, ha="left", rotation_mode="anchor")

    if proposal_new_label in current_data:
        for ticklabel in plt.gca().get_xticklabels():
            if ticklabel.get_text() == proposal_new_label:
                ticklabel.set_weight('bold')
    else:
        for ticklabel in plt.gca().get_xticklabels():
            if ticklabel.get_text() in  ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)']:
                ticklabel.set_weight('bold')

    # plt.title(titles[key])

    if logscale:
        fig.savefig(path.join(path_generated_figures, str(key) + suffix_graphfile + '_log.pdf'))
    else:
        fig.savefig(path.join(path_generated_figures, str(key) + suffix_graphfile + '.pdf'))

    plt.close()
    warnings.filterwarnings("error")

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

def print_global_metrics(metrics):
    row_head = '<tr>' # ''
    row_end = '</tr>\n' # '\\\\\n'
    cell_head = '<td> ' # ''
    cell_end = ' </td> ' # ' &'

    print(cell_head, len([np.mean(metrics[i_dataset]['coverage_mean'])
                          for i_dataset in metrics
                          if 'coverage_mean' in metrics[i_dataset] and len(metrics[i_dataset]['coverage_mean']) > 0]),
          cell_end)

    # aux_data = {i_dataset: np.mean(metrics[i_dataset]['coverage_mean'])
    #                       for i_dataset in metrics
    #                       if 'coverage_mean' in metrics[i_dataset] and len(metrics[i_dataset]['coverage_mean']) > 0}
    # print(aux_data, file=sys.stderr)

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

def print_mean_std(values, key):
    row_head = '<tr>' # ''
    row_end = '</tr>\n' # '\\\\\n'
    cell_head = '<td> ' # ''
    cell_end = ' </td> ' # ' &'

    try:
        metric = [np.mean(values[i][key]) for i in values if key in values[i] and len(values[i][key]) > 0]
        print(cell_head, end='')
        if len(metric) > 0:
            print('{:>3.0f}'.format(np.mean(metric)), '+-',
                  '{:<5.1f}'.format(np.std(metric)), sep='', end=' ')
        else:
            print('No values', end='')
        print(cell_end, end='')
    except (FloatingPointError, TypeError):
        # print(values)
        raise
    except KeyError:
        raise


def print_metrics(alg_name, rules_metrics):

    row_head = '<tr>' # ''
    row_end = '</tr>\n' # '\\\\\n'
    cell_head = '<td> ' # ''
    cell_end = ' </td> ' # ' &'

    print(row_head, end='')
    print(cell_head, end='')
    print('{:>14}'.format(alg_name.replace('ND2','Proposal')), end='')
    print(cell_end, end='')
    print(cell_head, end='')

    if len(rules_metrics['num_rules']) <= 0:
        print('No rules', cell_end, cell_head, cell_end, cell_head, cell_end, cell_head, 0, cell_end, cell_head, cell_end, cell_head, 0, cell_end, row_end, end='', sep='')
        return

    if np.mean(rules_metrics['num_rules']) <= 0:
        # print('No rules ({:d})'.format(len(rules_metrics['num_rules'])),
        #       cell_end, cell_head, cell_end, cell_head, cell_end,
        #       cell_head, 0, cell_end, cell_head, cell_end,
        #       cell_head, 0, cell_end, row_end, end='', sep='')
        print('No rules',
              cell_end, cell_head, cell_end, cell_head, cell_end,
              cell_head, 0, cell_end, cell_head, cell_end,
              cell_head, 0, cell_end, row_end, end='', sep='')
        return

    print('{:>5.2f}'.format(np.mean(rules_metrics['num_rules'])), end=' ')
    # print('({:d})'.format(len(rules_metrics['num_rules'])), end= ' ')
    print(cell_end, end='')
    print(cell_head, end='')
    print("{:>5.1f}".format(np.mean(rules_metrics['num_conditions_mean'])),'+-',
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
            # print_metrics(dominance.upper(), data[dominance.upper()][name_dataset_v2])
    except FileNotFoundError:
        print('No files')
        raise

    return data

def output_metrics(data, methods, metrics_v2):
    params = {'FITNESS_FUNCTION': Dummy_class()}
    params['FITNESS_FUNCTION'].training_exp = None
    params['MIN_TARGET_CLASS_RATIO'] = 0.1
    datasets = ['australian', 'balance', 'car','chess','german','glass','heart','hepatitis','iris','tic-tac-toe',
                'wine', 'wisconsin',
                'ionosphere', 'pima', 'vowel', 'data.arff']


    #######################
    # Parse the rules per method and print the metrics
    ########################

    with open(os.path.join(path_generated_figures,'results_per_dataset.html'), 'w') as f:
        with contextlib.redirect_stdout(f):
            with tqdm(total=len(datasets) * len(methods)) as pbar:

                print('<html>')
                print('<head>')
                print('<style>')
                print('table, th, td {')
                print('  border: 1px solid black;')
                print('  border-collapse: collapse;')
                print('}')
                print('</style>')
                print('</head>')
                print('<body>')
                try:
                    for i_dataset_CBA_andcia, i_dataset_Heu_ops, i_dataset_proposals,\
                            i_dataset_v2, goal_feature, goal_value, whole_dataset,\
                            rules_file_dataset_MOPNAR_andcia, binarize in \
                            zip(
                                # ['australian_truefalse_c','balance_truefalse_c','car_truefalse_c','chess_truefalse_c',
                                #  'german_truefalse_c','glass_truefalse_c','heart_truefalse_c','hepatitis_truefalse_c',
                                #  'iris_truefalse_c','tic-tac-toe_truefalse_c','wine_truefalse_c','wisconsin_truefalse_c',
                                #  'ionos_imba', 'pima_imba', 'vowel0_imba','complicaciones_filtered'],
                                ['australian_tf_sd', 'balance_tf_sd', 'car_tf_sd',
                                 'chess_tf_sd',
                                 'german_tf_sd', 'glass_tf_sd', 'heart_tf_sd',
                                 'hepatitis_tf_sd',
                                 'iris_tf_sd', 'tic-tac-toe_tf_sd', 'wine_tf_sd',
                                 'wisconsin_tf_sd',
                                 'ionosphere_sd', 'pima_sd', 'vowel0_sd', 'data_sd'],
                                # 'ionos_imba', 'pima_imba', 'vowel0_imba', 'data.arff'],
                                ['australian_truefalse_c', 'balance_truefalse_c', 'car_truefalse_c',
                                 'chess_truefalse_c',
                                 'german_truefalse_c', 'glass_truefalse_c', 'heart_truefalse_c',
                                 'hepatitis_truefalse_c',
                                 'iris_truefalse_c', 'tic-tac-toe_truefalse_c', 'wine_truefalse_c',
                                 'wisconsin_truefalse_c',
                                 'ionosphere.arff.csv', 'pima.arff.csv', 'vowel0_forR.arff.csv', 'data.arff.csv'],
                                ['australian', 'balance', 'car', 'chess',
                                 'german', 'glass', 'heart', 'hepatitis',
                                 'iris', 'tic-tac-toe', 'wine', 'wisconsin',
                                 'ionos_imba', 'pima_imba', 'vowel0_imba', 'complicaciones_filtered'],
                                datasets,
                                ['Class', 'Balance_scale', 'Acceptability', 'Class', 'Customer', 'TypeGlass',
                                 'Class', 'Class', 'Class',
                                 'Class', 'Class', 'Class',
                                 'class', 'class', 'Class', 'COMPLICACIONES'],
                                ['True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True', 'True',
                                 'True', 'True',
                                 'b', 'tested_positive', 'positive', "'Si'"],
                                [os.path.join('datasets', 'australian.arff'),
                                 os.path.join('datasets', 'balance.arff'),
                                 os.path.join('datasets', 'car_TrueFalse.arff'),
                                 os.path.join('datasets', 'chess_TrueFalse.arff'),
                                 os.path.join('datasets', 'german_TrueFalse.arff'),
                                 os.path.join('datasets', 'glass_TrueFalse.arff'),
                                 os.path.join('datasets', 'heart_TrueFalse.arff'),
                                 os.path.join('datasets', 'hepatitis_TrueFalse.arff'),
                                 os.path.join('datasets', 'iris_TrueFalse.arff'),
                                 os.path.join('datasets', 'tic-tac-toe_TrueFalse.arff'),
                                 os.path.join('datasets', 'wine_TrueFalse.arff'),
                                 os.path.join('datasets', 'wisconsin_TrueFalse.arff'),
                                 os.path.join('datasets', 'ionosphere.arff'),
                                 os.path.join('datasets', 'pima.arff'),
                                 os.path.join('datasets', 'vowel0.arff'),
                                 os.path.join('datasets', 'data.arff')],
                                ['australian_tf',  # 'australian_truefalse',
                                 'balance_tf',  # 'balance_truefalse',
                                 'car_tf',  # 'car_truefalse',
                                 'chess_tf',  # 'chess_truefalse',
                                 'german_tf',  # 'german_truefalse',
                                 'glass_tf',  # 'glass_truefalse',
                                 'heart_tf',  # 'heart_truefalse',
                                 'hepatitis_tf',  # 'hepatitis_truefalse',
                                 'iris_tf',  # 'iris_truefalse',
                                 'tic-tac-toe_tf',  # 'tic-tac-toe_truefalse',
                                 'wine_tf',  # 'wine_truefalse',
                                 'wisconsin_tf',  # 'wisconsin_truefalse',
                                 'ionosphere',  # 'assoc_ionosphere',
                                 'pima_assoc',  # 'assoc_pima',
                                 'vowel0',  # 'assoc_vowel0',
                                 'colorrectal'],  # ,'association_complicaciones_filtered'],
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
                                # zip(
                                #     # ['australian_truefalse_c','balance_truefalse_c','car_truefalse_c','chess_truefalse_c',
                                #     #  'german_truefalse_c','glass_truefalse_c','heart_truefalse_c','hepatitis_truefalse_c',
                                #     #  'iris_truefalse_c','tic-tac-toe_truefalse_c','wine_truefalse_c','wisconsin_truefalse_c',
                                #     #  'ionos_imba', 'pima_imba', 'vowel0_imba','complicaciones_filtered'],
                                #     ['australian_tf_sd', 'balance_tf_sd', 'car_tf_sd',
                                #      'chess_tf_sd',
                                #      'german_tf_sd', 'glass_tf_sd', 'heart_tf_sd',
                                #      'hepatitis_tf_sd',
                                #      'iris_tf_sd', 'tic-tac-toe_tf_sd', 'wine_tf_sd',
                                #      'wisconsin_tf_sd',
                                #      'ionos_imba', 'pima_imba', 'vowel0_imba', 'data.arff'],
                                #     ['australian_truefalse_c', 'balance_truefalse_c', 'car_truefalse_c', 'chess_truefalse_c',
                                #      'german_truefalse_c', 'glass_truefalse_c', 'heart_truefalse_c', 'hepatitis_truefalse_c',
                                #      'iris_truefalse_c', 'tic-tac-toe_truefalse_c', 'wine_truefalse_c', 'wisconsin_truefalse_c',
                                #      'ionosphere.arff.csv', 'pima.arff.csv', 'vowel0_forR.arff.csv', 'data.arff.csv'],
                                #     ['australian', 'balance', 'car', 'chess',
                                #      'german', 'glass', 'heart', 'hepatitis',
                                #      'iris', 'tic-tac-toe', 'wine', 'wisconsin',
                                #      'ionos_imba', 'pima_imba', 'vowel0_imba', 'complicaciones_filtered'],
                                #     datasets,
                                #     ['Class','Balance_scale','Acceptability','Class','Customer','TypeGlass','Class','Class','Class',
                                #      'Class','Class','Class',
                                #      'class','class','Class','COMPLICACIONES'],
                                #     ['True','True','True','True','True','True','True','True','True','True','True','True',
                                #      'b', 'tested_positive', 'positive', "'Si'"],
                                #     [os.path.join('datasets','australian.arff'),
                                #      os.path.join('datasets','balance.arff'),
                                #      os.path.join('datasets','car_TrueFalse.arff'),
                                #      os.path.join('datasets','chess_TrueFalse.arff'),
                                #      os.path.join('datasets','german_TrueFalse.arff'),
                                #      os.path.join('datasets','glass_TrueFalse.arff'),
                                #      os.path.join('datasets','heart_TrueFalse.arff'),
                                #      os.path.join('datasets','hepatitis_TrueFalse.arff'),
                                #      os.path.join('datasets','iris_TrueFalse.arff'),
                                #      os.path.join('datasets','tic-tac-toe_TrueFalse.arff'),
                                #      os.path.join('datasets','wine_TrueFalse.arff'),
                                #      os.path.join('datasets','wisconsin_TrueFalse.arff'),
                                #      os.path.join('datasets','ionosphere.arff'),
                                #      os.path.join('datasets','pima.arff'),
                                #      os.path.join('datasets','vowel0.arff'),
                                #      os.path.join('datasets','data.arff')],
                                #     ['australian_truefalse',
                                #      'balance_truefalse',
                                #      'car_truefalse',
                                #      'chess_truefalse',
                                #      'german_truefalse',
                                #      'glass_truefalse',
                                #      'heart_truefalse',
                                #      'hepatitis_truefalse',
                                #      'iris_truefalse',
                                #      'tic-tac-toe_truefalse',
                                #      'wine_truefalse',
                                #      'wisconsin_truefalse',
                                #      'assoc_ionosphere',
                                #      'assoc_pima',
                                #      'assoc_vowel0',
                                #      'association_complicaciones_filtered'],
                                #     [True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      True,
                                #      False,
                                #      False,
                                #      False,
                                #      False]):

                        print('<h2>', i_dataset_v2.replace('data.arff','colorectal'),'</h2>')
                        if len(data['Apriori_' + str(0.7) + '_' + str(0.05)][i_dataset_v2]['num_rules']) <= 0:
                            print('<p>Colorectal dataset is not available</p>')
                            pbar.set_description('')
                            pbar.update(17)
                            continue
                        print_tab_head()

                        #### Apriori
                        for i_conf, i_support in zip([0.7, 0.8, 0.9], [0.05, 0.025, 0.0125]):
                            try:
                                print_metrics('Apriori_' + str(i_conf)+'_'+str(i_support),
                                              data['Apriori_' + str(i_conf)+'_'+str(i_support)][i_dataset_v2])

                            except FileNotFoundError:
                                rules = None
                                raise Exception('Apriori & No files apriori '+str(i_conf)+' '+
                                                i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                            pbar.update(1)

                        #### Heuristic Operators
                        try:
                            print_metrics('Heu.Ops', data['Heu.Ops'][i_dataset_v2])
                        except FileNotFoundError:
                            rules = None
                            print('Heu.Ops & No files for:', os.path.join('apriori_heuristics','results','rules_Datasets_' +
                                                         i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv')))
                        pbar.update(1)

                        for i_alg, alg_name in zip(['CBA2', 'CPAR'], ['CBA2', 'CPAR']):
                            try:
                                print_metrics(alg_name, data[i_alg][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('No files')
                                raise
                            pbar.update(1)

                        try:
                            print_metrics('FARCHD', data['FARCHD'][i_dataset_v2])
                        except FileNotFoundError:
                            rules = None
                            print('No files')
                            raise
                        pbar.update(1)

                        for i_alg in ['NICGAR','MOPNAR', 'QAR_CIP_NSGAII']:
                            try:
                                print_metrics(i_alg, data[i_alg][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('No files')
                                raise
                            pbar.update(1)

                        for i_conf in range(1):
                            try:
                                print_metrics('CN2_SD', data['CN2_SD'][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('No files')
                                raise
                            except IndexError:
                                rules = None
                                #print('CN2_SD', i_dataset_v2, data['CN2_SD'][i_dataset_v2]['num_rules'], file=sys.stderr)
                                print('No files')
                                # raise
                            pbar.update(1)

                        for i_alg in ['SD-Algorithm-SD', 'SDMap-SD','NMEEF', 'MESDIF', 'SDIGA']:
                            try:
                                print_metrics(i_alg, data[i_alg][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('No files')
                                raise
                            except IndexError:
                                print(i_alg, i_dataset_v2, data[i_alg][i_dataset_v2], file=sys.stderr)
                                raise
                            pbar.update(1)

                        print_metrics('nd2'.upper(), data['nd2'.upper()][i_dataset_v2])
                        pbar.update(1)

                        print_tab_end()
                        sys.stdout.flush()

                except KeyboardInterrupt:
                    print('Process interrupted')
                    sys.exit(0)

                print('</body>')
                print('</html>')

    return data

def rules_to_metrics(methods, metrics_v2, data = None):
    params = {'FITNESS_FUNCTION': Dummy_class()}
    params['FITNESS_FUNCTION'].training_exp = None
    params['MIN_TARGET_CLASS_RATIO'] = 0.1
    datasets = ['australian', 'balance', 'car','chess','german','glass','heart','hepatitis','iris','tic-tac-toe',
                'wine', 'wisconsin',
                'ionosphere', 'pima', 'vowel', 'data.arff']

    if data is None:
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

    with open(os.path.join(path_generated_figures,'results_per_dataset.html'), 'w') as f:
        with contextlib.redirect_stdout(f):
            with nostdout():
                with tqdm(total=len(datasets) * len(methods)) as pbar:
                    try:
                        for i_dataset_CBA_andcia, i_dataset_Heu_ops, i_dataset_proposals,\
                                i_dataset_v2, goal_feature, goal_value, whole_dataset,\
                                rules_file_dataset_MOPNAR_andcia, binarize in\
                                    zip(
                                        # ['australian_truefalse_c','balance_truefalse_c','car_truefalse_c','chess_truefalse_c',
                                        #  'german_truefalse_c','glass_truefalse_c','heart_truefalse_c','hepatitis_truefalse_c',
                                        #  'iris_truefalse_c','tic-tac-toe_truefalse_c','wine_truefalse_c','wisconsin_truefalse_c',
                                        #  'ionos_imba', 'pima_imba', 'vowel0_imba','complicaciones_filtered'],
                                        ['australian_tf_sd', 'balance_tf_sd', 'car_tf_sd',
                                         'chess_tf_sd',
                                         'german_tf_sd', 'glass_tf_sd', 'heart_tf_sd',
                                         'hepatitis_tf_sd',
                                         'iris_tf_sd', 'tic-tac-toe_tf_sd', 'wine_tf_sd',
                                         'wisconsin_tf_sd',
                                         'ionosphere_sd', 'pima_sd', 'vowel0_sd', 'data_sd'],
                                         # 'ionos_imba', 'pima_imba', 'vowel0_imba', 'data.arff'],
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
                                        [os.path.join('datasets','australian.arff'),
                                         os.path.join('datasets','balance.arff'),
                                         os.path.join('datasets','car_TrueFalse.arff'),
                                         os.path.join('datasets','chess_TrueFalse.arff'),
                                         os.path.join('datasets','german_TrueFalse.arff'),
                                         os.path.join('datasets','glass_TrueFalse.arff'),
                                         os.path.join('datasets','heart_TrueFalse.arff'),
                                         os.path.join('datasets','hepatitis_TrueFalse.arff'),
                                         os.path.join('datasets','iris_TrueFalse.arff'),
                                         os.path.join('datasets','tic-tac-toe_TrueFalse.arff'),
                                         os.path.join('datasets','wine_TrueFalse.arff'),
                                         os.path.join('datasets','wisconsin_TrueFalse.arff'),
                                         os.path.join('datasets','ionosphere.arff'),
                                         os.path.join('datasets','pima.arff'),
                                         os.path.join('datasets','vowel0.arff'),
                                         os.path.join('datasets','data.arff')],
                                        ['australian_tf',# 'australian_truefalse',
                                         'balance_tf',# 'balance_truefalse',
                                         'car_tf',# 'car_truefalse',
                                         'chess_tf',# 'chess_truefalse',
                                         'german_tf',# 'german_truefalse',
                                         'glass_tf',# 'glass_truefalse',
                                         'heart_tf',# 'heart_truefalse',
                                         'hepatitis_tf',# 'hepatitis_truefalse',
                                         'iris_tf',# 'iris_truefalse',
                                         'tic-tac-toe_tf',# 'tic-tac-toe_truefalse',
                                         'wine_tf',# 'wine_truefalse',
                                         'wisconsin_tf',# 'wisconsin_truefalse',
                                         'ionosphere',# 'assoc_ionosphere',
                                         'pima_assoc',# 'assoc_pima',
                                         'vowel0',# 'assoc_vowel0',
                                         'colorrectal'],# ,'association_complicaciones_filtered'],
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
                                pbar.set_description('')
                                pbar.update(14)
                                continue
                            params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:, -1]
                            params['POSITIVE_CLASS'] = goal_value
                            params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)

                            print_tab_head()

                            #### Apriori
                            for i_conf, i_support in zip([0.7, 0.8, 0.9], [0.05, 0.025, 0.0125]):
                                try:
                                    pbar.set_description(
                                        "{:s} - Ap({:.1f},{:.3f}) - {:s}".format(i_dataset_CBA_andcia, i_conf, i_support,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                    rules = read_rules_apriori_from_file(os.path.join('apriori_heuristics','results_1','rules_apriori_' +
                                                                         str(i_conf) + '_' + str(i_support) + 'Datasets_' +
                                                                         i_dataset_Heu_ops.replace('truefalse_c',
                                                                                                   'TrueFalse_forR.arff.csv')),
                                                                         i_conf, dataset, goal_value, params)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, 'Apriori_' + str(i_conf)+'_'+str(i_support), i_dataset_v2, rules_metrics)
                                    # print_metrics('Apriori_' + str(i_conf)+'_'+str(i_support),
                                    #               data['Apriori_' + str(i_conf)+'_'+str(i_support)][i_dataset_v2])

                                except FileNotFoundError:
                                    rules = None
                                    raise Exception('Apriori & No files apriori '+str(i_conf)+' '+
                                                    i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                                pbar.update(1)

                            #### Heuristic Operators
                            try:
                                results_dirs = [i for i in os.listdir('apriori_heuristics') if i.startswith('results_')]
                                for iii, i_result_dir in enumerate(results_dirs):
                                    pbar.set_description(
                                        "{:s} - Heu.Ops({:d}) - {:s}".format(i_dataset_CBA_andcia,iii,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                    rules = read_rules_heuops_from_file(os.path.join('apriori_heuristics',i_result_dir,'rules_Datasets_' +
                                                                 i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv')),
                                                                 dataset, goal_value)
                                    rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                    data = add_metrics(data, 'Heu.Ops', i_dataset_v2, rules_metrics)
                                    # print_metrics('Heu.Ops', data['Heu.Ops'][i_dataset_v2])
                            except FileNotFoundError:
                                rules = None
                                print('Heu.Ops & No files for:', os.path.join('apriori_heuristics','results','rules_Datasets_' +
                                                             i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv')))
                            pbar.update(1)

                            for i_alg, alg_name in zip(['CBA2', 'CPAR'], ['CBA2', 'CPAR']):
                                counter = 0
                                for i_seed in seeds:
                                    try:
                                        # print(i_alg, i_seed)
                                        pbar.set_description(
                                            "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, i_alg,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                        rules = results_CBA2_etal_2latex(i_dataset_CBA_andcia, goal_value, i_alg, dataset, i_seed)
                                        rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                        data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)
                                        # print_metrics(alg_name, data[i_alg][i_dataset_v2])
                                        counter +=1
                                    except FileNotFoundError:
                                        rules = None
                                        # print('No files', file=sys.stderr)
                                        # raise

                                if counter == 0:
                                    raise FileNotFoundError('CBA')
                                pbar.update(1)

                            if 'FARCHD' in methods:
                                for i_seed in seeds:
                                    try:
                                        pbar.set_description(
                                            "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, "FARCHD",
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                        rules = results_FARCHD_2latex(i_dataset_CBA_andcia, goal_value, dataset, i_seed)#whole_dataset)
                                        rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                        data = add_metrics(data, 'FARCHD', i_dataset_v2, rules_metrics)
                                        # print_metrics('FARCHD', data['FARCHD'][i_dataset_v2])
                                    except FileNotFoundError:
                                        rules = None
                                        # print('No files')
                                        # raise
                                pbar.update(1)

                            for i_alg in ['NICGAR','MOPNAR', 'QAR_CIP_NSGAII']:
                                try:
                                    if i_alg in methods:
                                        for i_seed in seeds:
                                            pbar.set_description(
                                                "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, i_alg,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                            rules = results_MOPNAR_NICGAR_QARCIP_2latex(i_alg, rules_file_dataset_MOPNAR_andcia, goal_feature,
                                                                                        goal_value, dataset, i_seed)#whole_dataset)
                                            rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                            data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)

                                        # print_metrics(i_alg, data[i_alg][i_dataset_v2])
                                        # print(i_alg, '& ', end='')
                                        # print_rules_metric(rules, dataset, goal_value)
                                except FileNotFoundError:
                                    rules = None
                                    print('No files')
                                    raise
                                pbar.update(1)

                            for i_alg in ['CN2_SD', 'SD-Algorithm-SD', 'SDMap-SD']:
                                i_conf = None if i_alg != 'CN2_SD' else 0
                                i_dataset_name = i_dataset_CBA_andcia # if i_alg == 'CN2_SD' else i_dataset_NMEEF_andcia
                                if i_alg in methods:
                                    counter = 0
                                    for i_seed in seeds:
                                        # if i_alg == 'SDMap-SD':
                                        #     print(i_seed, file=sys.stderr)
                                        try:
                                            pbar.set_description(
                                                "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, i_alg,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                            rules = results2latex_CN2_SD(i_dataset_name,goal_feature, goal_value,i_alg.replace('CN2_SD','CN2-SD'), i_conf, dataset, i_seed)#whole_dataset)
                                            rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                            data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)
                                            counter += 1
                                        except FileNotFoundError as e:
                                            # print(i_alg, i_dataset_name, 'No files: REVISAR!!!\n', str(e), file=sys.stderr)
                                            # print(i_alg, i_dataset_name, 'No files: REVISAR!!!')
                                            # raise
                                            pass
                                    # print(i_dataset_v2, file=sys.stderr)
                                    # print(data[i_alg][i_dataset_v2], file=sys.stderr)
                                    # sys.exit(1)
                                    # print_metrics(i_alg, data[i_alg][i_dataset_v2])
                                    # print('CN2_SD & ', sep='', end='')
                                    # print_rules_metric(rules, dataset, goal_value)
                                    if counter == 0 and not (i_alg == 'SD-Algorithm-SD' and i_dataset_v2 == 'data.arff')\
                                            and not (i_alg == 'SDMap-SD' and i_dataset_v2 == 'chess')\
                                            and not (i_alg == 'SDMap-SD' and i_dataset_v2 == 'data.arff'):
                                         raise FileNotFoundError(str(i_alg)+str(i_dataset_v2))
                                pbar.update(1)

                            #### NMEEF and MESDIF
                            for i_alg in ['NMEEF', 'MESDIF', 'SDIGA']:
                                try:
                                    if i_alg in methods:
                                        for i_seed in seeds:
                                            pbar.set_description(
                                                "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, i_alg,
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                                            rules = results_NMEEF_MESDIF_2latex(i_alg,
                                                                        i_dataset_CBA_andcia,#i_dataset_NMEEF_andcia,
                                                                        goal_feature,
                                                                        goal_value, dataset, i_seed)
                                            rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                            data = add_metrics(data, i_alg, i_dataset_v2, rules_metrics)
                                        # print_metrics(i_alg, data[i_alg][i_dataset_v2])
                                except FileNotFoundError:
                                    rules = None
                                    print('No files')
                                    raise
                                pbar.update(1)

                            # print('UNCOMMENT ND1 AND ND2 RESULTS')
                            pbar.set_description(
                                "{:s} - {:s} - {:s}".format(i_dataset_CBA_andcia, "EXDERD",
                                                                                   datetime.datetime.now().strftime("%H:%M:%S")))
                            if 'ND1' in methods:
                                data = results_best_conf_proposal('nd1', i_dataset_proposals, data, i_dataset_v2, params, dataset)
                                pbar.update(1)

                            if 'ND2' in methods:
                                data = results_best_conf_proposal('nd2', i_dataset_proposals, data, i_dataset_v2, params, dataset)
                                pbar.update(1)

                        print_tab_end()
                        sys.stdout.flush()

                        if path.exists('metrics_algs_computed.pickle'):
                            import shutil
                            shutil.copyfile('metrics_algs_computed.pickle', 'metrics_algs_computed_previous.pickle')

                        with open('metrics_algs_computed.pickle', 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        return data

                    except KeyboardInterrupt:
                        print('Process interrupted')
                        sys.exit(0)

def datasets_properties():
    file_dataset = {'australian':os.path.join('datasets','australian.arff'),
                    'balance':os.path.join('datasets','balance.arff'),
                    'car':os.path.join('datasets','car.arff'),
                    'chess':os.path.join('datasets','chess.arff'),
                    'german':os.path.join('datasets','german.arff'),
                    'glass':os.path.join('datasets','glass.arff'),
                    'heart':os.path.join('datasets','heart.arff'),
                    'hepatitis':os.path.join('datasets','hepatitis.arff'),
                    'ionosphere':os.path.join('datasets','ionosphere.arff'),
                    'iris':os.path.join('datasets','iris.arff'),
                    'pima':os.path.join('datasets','pima.arff'),
                    'tic-tac-toe':os.path.join('datasets','tic-tac-toe.arff'),
                    'vowel':os.path.join('datasets','vowel0.arff'),
                    'wine':os.path.join('datasets','wine.arff'),
                    'wisconsin':os.path.join('datasets','wisconsin.arff'),
                    'colorrectal':os.path.join('datasets','data.arff')}
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

def starplot(data, metrics, metric_labels, methods, new_order):

    def radar_factory(num_vars, frame='circle'):
        """
        Create a radar chart with `num_vars` axes.

        This function creates a RadarAxes projection and registers it.

        Parameters
        ----------
        num_vars : int
            Number of variables for radar chart.
        frame : {'circle', 'polygon'}
            Shape of frame surrounding axes.

        """
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        from matplotlib.spines import Spine
        from matplotlib.path import Path
        from matplotlib.transforms import Affine2D
        from matplotlib.patches import Circle, RegularPolygon

        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        class RadarTransform(PolarAxes.PolarTransform):

            def transform_path_non_affine(self, path):
                # Paths with non-unit interpolation steps correspond to gridlines,
                # in which case we force interpolation (to defeat PolarTransform's
                # autoconversion to circular arcs).
                if path._interpolation_steps > 1:
                    path = path.interpolated(num_vars)
                return Path(self.transform(path.vertices), path.codes)

        class RadarAxes(PolarAxes):

            name = 'radar'
            PolarTransform = RadarTransform

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                          radius=.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                  spine_type='circle',
                                  path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta

    star_data = pd.DataFrame(np.nan, columns=metrics, index=methods)

    for i_metric in metrics:
        for i_method in methods:
            star_data.loc[i_method,i_metric] = np.nanmedian([np.mean(data[i_method][i_dataset][i_metric])
                                                        if i_metric in data[i_method][i_dataset] and
                                                           len(data[i_method][i_dataset][i_metric]) > 0 else np.nan
                                                        for i_dataset in data[i_method]])

    ranks_star_data = star_data.rank()

    theta = radar_factory(len(metrics), frame='polygon')

    # spoke_labels = data.pop(0)

    # fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
    #                         subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    plt.subplots_adjust(right=0.5)
    # fig.subplots_adjust(hspace=0.25, right=0.3)

    plt.set_cmap('Accent')
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title('Global', weight='bold', size='medium', position=(0.5,1.1),
                 horizontalalignment='center', verticalalignment='center')

    # Plot the four cases from the example data on separate axes
    for alg_index in ranks_star_data.index:
        ax.plot(theta, ranks_star_data.loc[alg_index,:])#, color=color)
        ax.fill(theta, ranks_star_data.loc[alg_index,:],# facecolor=color,
                alpha=0.25, label='_nolegend_')
    ax.set_varlabels(metric_labels)
    # for ax, (title, case_data) in zip(axs.flat, data):
    #     # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    #     # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #     #              horizontalalignment='center', verticalalignment='center')
    #     for d, color in zip(case_data, colors):
    #         ax.plot(theta, d, color=color)
    #         ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    #     ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    # legend = ax.legend(methods, loc=(0.9, .95), labelspacing=0.1, fontsize='small')
    ax.legend(methods,loc=(1.3,0), fontsize='x-small')
    # labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    # legend = axs[0, 0].legend(labels, loc=(0.9, .95),
    #                           labelspacing=0.1, fontsize='small')
    #
    # fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size='large')
    #
    # plt.show()
    plt.savefig('spider_graph.pdf')
    plt.close()

    # try:
    #     current_data = [[np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(
    #         data[i_method][i][key]) > 0 else np.nan
    #                      for i in data[i_method]]
    #                     for i_method in data]  # if i_method not in ['Apriori0.7','Apriori0.8','Apriori0.9']]
    #
    # except RuntimeWarning:
    #     print('METRIC:', key, '\n', [[data[i_method][i][key] for i in data[i_method]]
    #                                  for i_method in data], '\n')
    #     raise
    #
    # try:
    #     new_current_data = np.array(current_data).T
    # except np.VisibleDeprecationWarning:
    #     print('METRIC:', key, '\n', current_data)
    #     raise
    # # if new_current_data.shape[0] == 0:
    # #     print(current_data)
    # current_data = new_current_data
    #
    # try:
    #     current_data = pd.DataFrame(current_data,
    #                                 columns=xlabels)
    # except (ValueError, RuntimeWarning):
    #     print('METRIC:', key, '\n', current_data, '\n')
    #     raise
    #
    # if key in ['num_rules', 'global_coverage']:  # and suffix_graphfile == '_apriori'
    #     # print(current_data)
    #     current_data[np.isnan(current_data)] = 0
    #     # current_data.fillna(0)
    #     # print(current_data)
    #
    # current_data = current_data[new_order]

def starplot_v2(data, metrics, metric_labels, methods):
    # import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    ticks = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    # text = lambda: "".join(np.random.choice(list("manuladefil"), size=10))
    # labels = [text() for _ in range(len(ticks))]
    labels = metric_labels

    plt.xticks(ticks, labels, size=16)
    for label, rot in zip(ax.get_xticklabels(), ticks):
        label.set_rotation(rot * 180. / np.pi)
        print(label, rot * 180. / np.pi)
        label.set_horizontalalignment("left")
        label.set_rotation_mode("anchor")

    plt.tight_layout()

    star_data = pd.DataFrame(np.nan, columns=metrics, index=methods)

    for i_metric in metrics:
        for i_method in methods:
            star_data.loc[i_method,i_metric] = np.nanmedian([np.mean(data[i_method][i_dataset][i_metric])
                                                        if i_metric in data[i_method][i_dataset] and
                                                           len(data[i_method][i_dataset][i_metric]) > 0 else np.nan
                                                        for i_dataset in data[i_method]])

    ranks_star_data = star_data.rank()

    plt.set_cmap('Accent')
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    # ax.set_title('Global', weight='bold', size='medium', position=(0.5,1.1),
    #              horizontalalignment='center', verticalalignment='center')

    # Plot the four cases from the example data on separate axes
    for alg_index in ranks_star_data.index:
        ax.plot(ranks_star_data.loc[alg_index,:])
        # ax.plot(theta, ranks_star_data.loc[alg_index,:])#, color=color)
        # ax.fill(theta, ranks_star_data.loc[alg_index,:],# facecolor=color,
        #         alpha=0.25, label='_nolegend_')
    # ax.set_varlabels(metric_labels)

    # add legend relative to top-left plot
    ax.legend(methods,loc=(1.3,0), fontsize='x-small')
    plt.savefig('spider_graph.pdf')
    plt.close()

def starplot_v3(data, metrics, metric_labels, methods):
    # import numpy as np
    import matplotlib.pyplot as plt

    # GET the data
    star_data = pd.DataFrame(np.nan, columns=metrics, index=methods)

    for i_metric in metrics:
        for i_method in methods:
            star_data.loc[i_method,i_metric] = np.nanmedian([np.mean(data[i_method][i_dataset][i_metric])
                                                        if i_metric in data[i_method][i_dataset] and
                                                           len(data[i_method][i_dataset][i_metric]) > 0 else np.nan
                                                        for i_dataset in data[i_method]])

    ranks_star_data = star_data.rank()


    r = np.arange(0, 1, 1/len(metric_labels))
    theta = 2 * np.pi * r

    ax = plt.subplot(111, projection='polar')

    for alg_index in ranks_star_data.index:
        ax.plot(theta,ranks_star_data.loc[alg_index,:].values)
        ax.fill(theta,ranks_star_data.loc[alg_index, :].values,  # facecolor=color,
                alpha=0.25, label='_nolegend_')

    # ax.plot(theta, r)
    # ax.set_rmax(2)
    ax.set_rticks([])
    ticks = np.linspace(0, 360, len(metric_labels)+1)
    ax.set_xticks(np.deg2rad(ticks))
    ticklabels = metric_labels
    ticklabels.append('')
    ax.set_xticklabels(ticklabels, fontsize=10)

    plt.gcf().canvas.draw()
    angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    angles = np.rad2deg(angles)
    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x, y = label.get_position()
        lab = ax.text(x, y-0.2, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va())
        print(label.get_text())
        lab.set_rotation(angle)
        # label.set_horizontalalignment("right")
        # label.set_rotation_mode("anchor")
        labels.append(lab)
    ax.set_xticklabels([])
    plt.savefig('spider_graph.pdf')
    plt.close()

def wilcox_heatmap(data, metrics, metric_labels, methods, method_labels):
    warnings.filterwarnings("ignore")
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
    heatmap_data_image[np.abs(heatmap_data) > 0.10] = 0

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
    plt.savefig(path.join(path_generated_figures,'heatmap_wilcox.pdf'))
    plt.close()
    warnings.filterwarnings("error")


if __name__ == '__main__':
    # datasets_properties()

    metrics_v2 = ['num_rules', 'num_conditions_mean', 'num_conditions_std',
                  'coverage_min',
                  'coverage_mean', 'global_coverage', 'confidence_min', 'confidence_mean',
                  'ratio_used_features', 'max_feat_freq', 'mean_significance', 'mean_wracc', 'mean_lift']
    methods = ['Apriori_0.7_0.05','Apriori_0.8_0.025','Apriori_0.9_0.0125','Heu.Ops','CBA2','CPAR','FARCHD','ND1','ND2','NICGAR','MOPNAR', 'QAR_CIP_NSGAII', 'CN2_SD']
    methods = ['Apriori_0.7_0.05', 'Apriori_0.8_0.025', 'Apriori_0.9_0.0125', 'Heu.Ops', 'CBA2', 'CPAR', 'FARCHD',
               'ND2', 'NICGAR', 'MOPNAR', 'QAR_CIP_NSGAII', 'CN2_SD', 'NMEEF', 'MESDIF', 'SDIGA','SDMap-SD','SD-Algorithm-SD']

    data = None

    if path.exists('metrics_algs_original.pickle'):
        answer = 'p'

        while answer not in ['1', '2', '3']:
            print('\n**********************************************************')
            print('This software is able to read association rules from different methods and produce graphs and tables.')
            print()
            print('This software should have been provided together with files:')
            print(' - metrics_algs_original.pickle: File with performance metrics already computed, as used in the submission')
            print(' - metrics_algs_computed.pickle: File with performance metrics computed from the available data (colorrectal dataset is not available)')
            print('')
            print('Please, choose one of the following options (1/2/3):')
            print('1: Generate the graphs from metrics_algs_original.pickle (quickly)')
            print('2: Generate the graphs from metrics_algs_computed.pickle (quickly)')
            print('3: Recompute the performance metrics into metrics_algs_computed.pickle from the rules produced by the methods (takes time, 2 hours and 50 minutes in my case)')
            print('')
            print('PLEASE, notice that the software always writes to the directory generated_results, regardless of the option chosen.')
            answer = input('Please, choose one of the previous options (1/2/3):\n')
            # answer = '3' # remove

            if answer == '3':
                # with open('metrics_algs_original.pickle', 'rb') as handle:
                #     data = pickle.load(handle)
                data = rules_to_metrics(methods, metrics_v2, data)
            elif answer == '1':
                print('I am using the previously computed original performance values (metrics_algs_original.pickle)')
                with open('metrics_algs_original.pickle', 'rb') as handle:
                    data = pickle.load(handle)
            elif answer == '2':
                print('I am using the previously computed performance values (metrics_algs_computed.pickle)')
                with open('metrics_algs_computed.pickle', 'rb') as handle:
                    data = pickle.load(handle)
            else:
                print('Not valid answer: ', answer)
    else:
        data = rules_to_metrics(methods, metrics_v2)

    output_metrics(data, methods, metrics_v2)

    with open(os.path.join(path_generated_figures,'results_per_dataset.html'), 'a') as f:
        with contextlib.redirect_stdout(f):

            print('<h2>GLOBAL RESULTS</h2>')
            print('<table>')
            print('<tr>')
            print('<td>Alg.</td>',
                  '<td>{:>11}'.format('#Datasets with rules'), '</td>',
                  '<td>{:>11}'.format('#rules'), '</td>',
                  '<td>{:>12}'.format('conds'), '</td>',
                  '<td>{:>12}'.format('I.min_cov'), '</td>',
                  '<td>{:>12}'.format('I.mean_cov'), '</td>',
                  '<td>{:>12}'.format('G.coverage'), '</td>',
                  '<td>{:>12}'.format('min_conf'), '</td>',
                  '<td>{:>12}'.format('mean_conf'), '</td>',
                  '<td>{:>12}'.format('%used_attrs'), '</td>',
                  '<td>{:>12}'.format('Attr.dom'), '</td>',
                  '<td>{:>12}'.format('Significance'), '</td>',
                  '<td>{:>12}'.format('WRAcc'), '</td>',
                  '<td>{:>12}'.format('Lift'), '</td>',
                  sep='')
            for i in methods:
                print("<tr><td>{:>14}".format(i.replace('ND2','Proposal')), '</td>', end=' ')
                # print(i, file=sys.stderr)
                print_global_metrics(data[i])
                print('</tr>')

    print('\n-------------------------\nGENERATING PLOTS')
    logscale_for = ['num_rules']
    xlabels = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','Heu.Ops', 'CBA2', 'CPAR', 'FARCHD', 'Cons.D', 'Proposal',
               'NICGAR', 'MOPNAR', 'QAR-CIP', 'CN2-SD', 'NMEEF', 'MESDIF', 'SDIGA','SDMap-SD','SD-Algorithm-SD']
    xlabels = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','Heu.Ops', 'CBA2', 'CPAR', 'FARCHD', 'EXDERD',
               'NICGAR', 'MOPNAR', 'QAR-CIP', 'CN2', 'NMEEF', 'MESDIF', 'SDIGA','SDMap','SD-Alg']
    new_order = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
               'NICGAR','MOPNAR','QAR-CIP','CN2','Heu.Ops','Cons.D','Bold.D']
    new_order = ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
               'NICGAR','MOPNAR','QAR-CIP','CN2','NMEEF', 'MESDIF', 'SDIGA','SDMap','SD-Alg','Heu.Ops','Proposal']
    new_order = {'num_rules': ['Ap(90,10/8)','Ap(80,10/4)','Heu.Ops','Ap(70,10/2)','CPAR','EXDERD','CBA2','FARCHD',
                               'QAR-CIP','MOPNAR','CN2','SD-Alg','MESDIF','NICGAR','SDIGA','NMEEF','SDMap'],
                 'num_conditions_mean': ['SDMap','NMEEF','NICGAR','SDIGA','QAR-CIP','MOPNAR','FARCHD','CBA2','SD-Alg',
                                         'CPAR','Heu.Ops','Ap(70,10/2)','Ap(80,10/4)','EXDERD','Ap(90,10/8)',
                                         'CN2', 'MESDIF'],
                 'num_conditions_std': ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
                                        'NICGAR','MOPNAR','QAR-CIP','CN2','NMEEF', 'MESDIF', 'SDIGA','SDMap','SD-Alg','Heu.Ops','EXDERD'],
                 'coverage_min': ['Ap(70,10/2)','Ap(80,10/4)','Ap(90,10/8)','CBA2','CPAR','FARCHD',
                                  'NICGAR','MOPNAR','QAR-CIP','CN2','NMEEF', 'MESDIF', 'SDIGA','SDMap','SD-Alg','Heu.Ops','EXDERD'],
                 'coverage_mean': ['SDIGA','NMEEF','NICGAR','CN2','SD-Alg','MOPNAR','MESDIF','EXDERD',
                                   'Ap(70,10/2)','QAR-CIP','FARCHD','Heu.Ops','Ap(80,10/4)','Ap(90,10/8)','CPAR','CBA2',
                                     'SDMap'],
                 'global_coverage': ['EXDERD','CN2','Heu.Ops','CPAR','FARCHD','Ap(90,10/8)','Ap(80,10/4)','Ap(70,10/2)','CBA2',
                                     'SDIGA','NICGAR','SD-Alg','MESDIF','MOPNAR','NMEEF','QAR-CIP','SDMap'],
                 'confidence_min': ['Ap(90,10/8)','CBA2','QAR-CIP','NMEEF','Ap(80,10/4)','NICGAR','Ap(70,10/2)','EXDERD','SD-Alg',
                                    'MOPNAR','Heu.Ops','CPAR','CN2','SDIGA','FARCHD','MESDIF','SDMap'],
                 'confidence_mean': ['CBA2','Ap(90,10/8)','QAR-CIP','CPAR','Ap(80,10/4)','EXDERD','NMEEF','NICGAR','Ap(70,10/2)','Heu.Ops','SD-Alg',
                                     'FARCHD','MOPNAR','CN2','SDIGA','MESDIF','SDMap'],
                 'ratio_used_features': ['EXDERD','Heu.Ops','Ap(90,10/8)','CPAR','Ap(80,10/4)','Ap(70,10/2)','FARCHD','CBA2',
                                         'CN2','MESDIF','QAR-CIP','MOPNAR','SD-Alg','NICGAR','SDIGA','NMEEF','SDMap'],
                 'max_feat_freq': ['SDMap','Heu.Ops','CBA2','FARCHD','EXDERD','CPAR','Ap(70,10/2)','Ap(90,10/8)','Ap(80,10/4)',
                                   'NICGAR','SD-Alg','QAR-CIP','MOPNAR','CN2','NMEEF', 'SDIGA', 'MESDIF'],
                 'mean_significance': ['NMEEF','CN2','SD-Alg','NICGAR','QAR-CIP','MOPNAR','EXDERD','Ap(70,10/2)',
                                       'FARCHD','Ap(80,10/4)','Heu.Ops','SDIGA','Ap(90,10/8)','CBA2','MESDIF','CPAR','SDMap'],
                 'mean_wracc': ['NMEEF','CN2','SD-Alg','NICGAR','EXDERD','QAR-CIP','MOPNAR','Ap(70,10/2)','SDIGA','FARCHD','Heu.Ops',
                                'Ap(80,10/4)','Ap(90,10/8)','MESDIF','CBA2','CPAR','SDMap'],
                 'mean_lift': ['Ap(90,10/8)','CBA2','CPAR','Ap(80,10/4)','QAR-CIP','Heu.Ops','EXDERD','NICGAR','MOPNAR',
                               'FARCHD','CN2','Ap(70,10/2)','SD-Alg','NMEEF','SDIGA','MESDIF','SDMap']}

    metrics_v2 = ['num_rules', 'num_conditions_mean',# 'num_conditions_std',
                  # 'coverage_min',
                  'coverage_mean', 'global_coverage', 'confidence_min', 'confidence_mean',
                  'ratio_used_features', 'max_feat_freq', 'mean_significance', 'mean_wracc', 'mean_lift']
    for i in tqdm(metrics_v2):
        if i in logscale_for:
            boxplot(data, i, xlabels, new_order[i],
                    max_or_min='max' if i not in ['num_conditions_mean', 'max_feat_freq'] else 'min', logscale=True,
                    proposal_new_label='EXDERD')
        else:
            boxplot(data, i, xlabels, new_order[i],
                    max_or_min='max' if i not in ['num_conditions_mean', 'max_feat_freq'] else 'min', logscale=False,
                    proposal_new_label='EXDERD')

    metrics_v3 = {#'num_rules':'min',
                  'num_conditions_mean': 'min', 'ratio_used_features': 'max',
                  'max_feat_freq': 'min', 'coverage_mean': 'max',
                  'global_coverage': 'max', 'confidence_mean': 'max', 'confidence_min': 'max',
                  'mean_lift': 'max', 'mean_significance': 'max', 'mean_wracc': 'max'}

    if r_installed:
        compute_ScottKnott_ranks(data, 'coverage_mean', metrics_v3['coverage_mean'])
        ranks = pd.DataFrame(
            {i: compute_ScottKnott_ranks(data, i, metrics_v3[i])[new_order['num_rules']].values[0] for i in metrics_v3.keys()}, index=new_order['num_rules']
        )

        ranks['mean_rank'] = ranks.mean(axis='columns')
        ranks.sort_values(by='mean_rank', ascending=False, inplace=True)
        ranks_style = ranks.style.format(decimal='.', precision=2)
        print('\n*****************************************************')
        print('Non-parametric SK-ESD rankings of the methods over the different metrics')
        print('*****************************************************')
        print(ranks_style.to_latex())


    #NUM RULES per dataset
    dataset_names = pd.DataFrame([list(data[i].keys()) for i in data]).iloc[0,:]
    key = 'num_rules'
    num_rules = pd.DataFrame([
        [np.mean(data[i_method][i][key]) if key in data[i_method][i] and len(data[i_method][i][key]) > 0 else np.nan
         for i in data[i_method]]
        for i_method in data], index=list(data.keys()), columns=dataset_names)
    num_rules.index = [i.replace('ND2', 'EXDERD').replace(
        'QAR_CIP_NSGAII', 'QAR-CIP').replace(
        'SD-Algorithm-SD', 'SD-Alg').replace('SDMap-SD', 'SDMap') for i in num_rules.index]
    num_rules['Avg'] = num_rules.mean(axis=1)
    for_latex = pd.DataFrame(num_rules['Avg'].copy())
    for_latex['Min'] = num_rules.min(axis=1)
    for_latex['Max'] = num_rules.max(axis=1)
    for_latex['Median'] = num_rules.median(axis=1)
    for_latex['25%'] = num_rules.quantile(0.25, axis=1)
    for_latex['75%'] = num_rules.quantile(0.75, axis=1)
    for_latex.fillna(0, inplace=True)
    for_latex = for_latex[['Min', '25%', 'Avg', 'Median', '75%', 'Max']]
    for_latex.sort_values(by='Avg', ascending=False, inplace=True)
    print('\n*****************************************************')
    print('Statistics of the number of rules produced by the different methods')
    print('*****************************************************')
    for_latex = for_latex.style.format(precision=0)
    print(for_latex.to_latex())
