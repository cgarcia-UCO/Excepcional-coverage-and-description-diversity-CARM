import os
import sys
sys.path.append(os.getcwd())
#sys.path.insert(1, '.')
from apriori_heuristics.results2latex import read_rules_apriori_from_file
sys.path.append(os.path.join(os.getcwd(), "scripts"))
#sys.path.insert(2,'scripts')
from results2latex import *
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
    confidences = [0.7, 0.8, 0.9]
    support_values = [0.1, 0.05, 0.025, 0.0125]

    # with open(os.path.join('generated_results','results_per_dataset.html'), 'w') as f:
    #     with contextlib.redirect_stdout(f):
    with nostdout():
        with tqdm(total=len(datasets) * len(confidences) * len(support_values)) as pbar:
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
                    dataset = None
                    try:
                        dataset, _, _, _ = read_arff(whole_dataset, binarize, params)
                        print(set(dataset.iloc[:, -1]), dataset.shape, dataset.iloc[:, -1].shape)
                        params['FITNESS_FUNCTION'].training_exp = dataset.iloc[:, -1]
                        params['POSITIVE_CLASS'] = goal_value
                        params['FITNESS_FUNCTION'].training_in = dataset.drop((dataset.columns[-1]), axis=1,
                                                                              inplace=False)
                    except FileNotFoundError:
                        print('Dataset', whole_dataset, 'is not present. Particularly, we had no permission to distribute the colorectal dataset')
                        pbar.update(12)
                        pbar.set_description('')
                        continue
                        # dataset = None


                    print_tab_head()

                    #### Apriori
                    for i_conf in confidences:
                        for i_support in support_values:
                            try:
                                if dataset is None:
                                    raise FileNotFoundError(whole_dataset)

                                pbar.set_description("{:s} - Ap({:.1f},{:.3f}) - {:s}".format(i_dataset_Heu_ops, i_conf, i_support,
                                                                           datetime.datetime.now().strftime("%H:%M:%S")))
                                rules = read_rules_apriori_from_file(os.path.join('apriori_heuristics','results_1','rules_apriori_'+
                                                                    str(i_conf)+'_'+str(i_support)+'Datasets_'+
                                                                    i_dataset_Heu_ops.replace('truefalse_c',
                                                                                            'TrueFalse_forR.arff.csv')),
                                                                    i_conf, dataset, goal_value, params)
                                rules_metrics = compute_rules_metrics(rules, dataset, goal_value)
                                data = add_metrics(data, 'Apriori_'+str(i_conf)+'_'+str(i_support), i_dataset_v2, rules_metrics)
                                print_metrics('Apriori_'+str(i_conf)+'_'+str(i_support),
                                            data['Apriori_'+str(i_conf)+'_'+str(i_support)][i_dataset_v2])
                                del rules_metrics
                                del rules
                            except FileNotFoundError:
                                rules = None
                                print('Apriori & No files apriori '+str(i_conf)+' '+str(i_support)+' '+
                                    i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                                # raise Exception('Apriori & No files apriori '+str(i_conf)+' '+str(i_support)+' '+
                                #                 i_dataset_Heu_ops.replace('truefalse_c','TrueFalse_forR.arff.csv'))
                            pbar.update(1)

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

    data = None

    if path.exists('metrics_apriori_original.pickle'):
        answer = 'p'

        while answer not in ['1', '2', '3']:
            print('\n**********************************************************')
            print('This software should have been provided together with files:')
            print(' - metrics_apriori_original.pickle: File with performance metrics already computed, as used in the submission')
            print(' - metrics_apriori.pickle: File with performance metrics computed from the available data (colorrectal dataset is not available)')
            print('Please, choose one of the following options (1/2/3):')
            print('1: Generate the graphs from metrics_apriori_original.pickle (quickly)')
            print('2: Generate the graphs from metrics_apriori.pickle (quickly)')
            print('3: Recompute the performance metrics into metrics_apriori.pickle from the rules produced by the methods (takes time, 2 hours and a half in my case)')
            print('')
            print('PLEASE, notice that the software always writes to the directory generated_results, regardless of the option chosen.')
            answer = input('Please, choose one of the previous options (1/2/3):\n')
            # answer = '3' # remove

            if answer == '3':
                data = apriori_to_metrics(methods, metrics_v2)
            elif answer == '1':
                print('I am using the previously computed original performance values (metrics_apriori_original.pickle)')
                with open('metrics_apriori_original.pickle', 'rb') as handle:
                    data = pickle.load(handle)
            elif answer == '2':
                print('I am using the previously computed performance values (metrics_apriori.pickle)')
                with open('metrics_apriori.pickle', 'rb') as handle:
                    data = pickle.load(handle)
            else:
                print('Not valid answer: ', answer)
    else:
        data = apriori_to_metrics(methods, metrics_v2)


    print('\n-------------------------\nGENERATING PLOTS')
    logscale_for = ['num_rules']
    metrics_v2 = ['num_rules', 'num_conditions_mean',# 'num_conditions_std',
                  # 'coverage_min',
                  'coverage_mean', 'global_coverage', 'confidence_min', 'confidence_mean',
                  'ratio_used_features', 'max_feat_freq', 'mean_significance', 'mean_wracc', 'mean_lift']
    for i in tqdm(metrics_v2):
        if i in logscale_for:
            boxplot(data, i, methods_boxplots, new_order, logscale=True, suffix_graphfile='_apriori')
        else:
            boxplot(data, i, methods_boxplots, new_order, logscale=False, suffix_graphfile='_apriori')


