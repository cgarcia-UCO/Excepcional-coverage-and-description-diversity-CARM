import sys
sys.path.insert(1, '.')
from apriori_heuristics.results2latex import read_rules_apriori_from_file
#import 05_results2latex.read_rules_apriori_from_file
sys.path.insert(2,'scripts')
from dummy_link import *
import matplotlib
from tqdm import tqdm
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

def apriori_to_metrics(methods, metrics_v2):
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

    try:
        for i_dataset_Heu_ops, i_dataset_v2, goal_feature, goal_value, whole_dataset, binarize in\
                    zip(
                        ['australian_truefalse_c', 'balance_truefalse_c', 'car_truefalse_c', 'chess_truefalse_c',
                         'german_truefalse_c', 'glass_truefalse_c', 'heart_truefalse_c', 'hepatitis_truefalse_c',
                         'iris_truefalse_c', 'tic-tac-toe_truefalse_c', 'wine_truefalse_c', 'wisconsin_truefalse_c',
                         'ionosphere.arff.csv', 'pima.arff.csv', 'vowel0_forR.arff.csv', 'data.arff.csv'],
                        datasets,
                        ['Class','Balance_scale','Acceptability','Class','Customer','TypeGlass','Class','Class','Class',
                         'Class','Class','Class',
                         'class','class','Class','COMPLICACIONES'],
                        ['True','True','True','True','True','True','True','True','True','True','True','True',
                         'b', 'tested_positive', 'positive', "'Si'"],
                        ['../../datasets/supervised_learning/australian.arff',
                         '../../datasets/supervised_learning/balance.arff',
                         '../../datasets/supervised_learning/car_TrueFalse.arff',
                         '../../datasets/supervised_learning/chess_TrueFalse.arff',
                         '../../datasets/supervised_learning/german_TrueFalse.arff',
                         '../../datasets/supervised_learning/glass_TrueFalse.arff',
                         '../../datasets/supervised_learning/heart_TrueFalse.arff',
                         '../../datasets/supervised_learning/hepatitis_TrueFalse.arff',
                         '../../datasets/supervised_learning/iris_TrueFalse.arff',
                         '../../datasets/supervised_learning/tic-tac-toe_TrueFalse.arff',
                         '../../datasets/supervised_learning/wine_TrueFalse.arff',
                         '../../datasets/supervised_learning/wisconsin_TrueFalse.arff',
                         '../../datasets/imbalanced/ionosphere.arff',
                         '../../datasets/imbalanced/pima.arff',
                         '../../datasets/imbalanced/vowel0.arff',
                         '../../datasets/data.arff'],
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
            # print(i_dataset_CBA_andcia, i_dataset_v2, goal_feature, goal_value, whole_dataset,
            #       rules_file_dataset_MOPNAR_andcia, binarize)
            dataset, _, _, _ = read_arff(whole_dataset, binarize, params)
            print(set(dataset.iloc[:,-1]), dataset.shape, dataset.iloc[:, -1].shape)
            params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:, -1]
            params['POSITIVE_CLASS'] = goal_value
            params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1, inplace=False)

            print_tab_head()

            #### Apriori
            for i_conf in [0.7, 0.8, 0.9]:
                for i_support in [0.1, 0.05, 0.025, 0.0125]:
                    try:
                        rules = read_rules_apriori_from_file('apriori_heuristics/results/rules_apriori_'+
                                                             str(i_conf)+'_'+str(i_support)+'Datasets_'+
                                                             i_dataset_Heu_ops.replace('truefalse_c',
                                                                                       'TrueFalse_forR.arff.csv'),
                                                             i_conf, dataset, goal_value, params)
                        rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                        data = add_metrics(data, 'Apriori_'+str(i_conf)+'_'+str(i_support), i_dataset_v2, rules_metrics)
                        print_metrics('Apriori_'+str(i_conf)+'_'+str(i_support),
                                      data['Apriori_'+str(i_conf)+'_'+str(i_support)][i_dataset_v2])
                    except FileNotFoundError:
                        rules = None
                        print('Apriori & No files apriori '+str(i_conf)+' '+str(i_support)+' '+
                              i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                        # raise Exception('Apriori & No files apriori '+str(i_conf)+' '+str(i_support)+' '+
                        #                 i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))

        with open('metrics_apriori.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return data

    except KeyboardInterrupt:
        print('Process interrupted')
        sys.exit(0)


if __name__ == '__main__':
    metrics_v2 = ['num_rules', 'num_conditions_mean', 'num_conditions_std',
                  'coverage_min', 'coverage_mean', 'global_coverage', 'confidence_min', 'confidence_mean',
                  'ratio_used_features', 'max_feat_freq', 'mean_significance', 'mean_wracc', 'mean_lift']
    methods = ['Apriori_0.7_0.1','Apriori_0.8_0.1','Apriori_0.9_0.1',
               'Apriori_0.7_0.05','Apriori_0.8_0.05','Apriori_0.9_0.05',
               'Apriori_0.7_0.025','Apriori_0.8_0.025','Apriori_0.9_0.025',
               'Apriori_0.7_0.0125','Apriori_0.8_0.0125','Apriori_0.9_0.0125']
    methods_boxplots = ['Ap(70,10/1)','Ap(80,10/1)','Ap(90,10/1)',
                        'Ap(70,10/2)','Ap(80,10/2)','Ap(90,10/2)',
                        'Ap(70,10/4)','Ap(80,10/4)','Ap(90,10/4)',
                        'Ap(70,10/8)','Ap(80,10/8)','Ap(90,10/8)']
    new_order = ['Ap(70,10/1)',
                 'Ap(70,10/2)',
                 'Ap(70,10/4)',
                 'Ap(70,10/8)',
                 'Ap(80,10/1)',
                 'Ap(80,10/2)',
                 'Ap(80,10/4)',
                 'Ap(80,10/8)',
                 'Ap(90,10/1)',
                 'Ap(90,10/2)',
                 'Ap(90,10/4)',
                 'Ap(90,10/8)']
    # data = apriori_to_metrics(methods, metrics_v2)
    data = None
    with open('metrics_apriori.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # print('\n------------------------\nGLOBAL RESULTS')
    # print('                 ',
    #       '{:>11}'.format('#rules'), ' & ',
    #       '{:>12}'.format('conds'), ' & ',
    #       '{:>12}'.format('I.min_cov'), ' & ',
    #       '{:>12}'.format('I.mean_cov'), ' & ',
    #       '{:>12}'.format('G.coverage'), ' & ',
    #       '{:>12}'.format('min_conf'), ' & ',
    #       '{:>12}'.format('mean_conf'), ' & ',
    #       '{:>12}'.format('%used_attrs'), ' & ',
    #       '{:>12}'.format('Attr.dom'), ' & ',
    #       '{:>12}'.format('Significance'), ' & ',
    #       '{:>12}'.format('WRAcc'), ' & ',
    #       '{:>12}'.format('Lift'),
    #       sep='')
    # for i in methods:
    #     print("{:>14}".format(i), '&', end=' ')
    #     print_global_metrics(data[i])

    print('\n-------------------------\nGENERATING PLOTS')
    logscale_for = ['num_rules']
    for i in tqdm(metrics_v2):
        if i in logscale_for:
            boxplot(data, i, methods_boxplots, new_order, logscale=True, suffix_graphfile='_apriori')
        else:
            boxplot(data, i, methods_boxplots, new_order, logscale=False, suffix_graphfile='_apriori')


