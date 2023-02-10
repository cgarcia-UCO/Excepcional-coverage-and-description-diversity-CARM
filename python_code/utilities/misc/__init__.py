import sys
sys.path.insert(1, './python_code')
from algorithm.parameters import params
from utilities.stats.trackers import cache, aux
# noinspection PyUnresolvedReferences
import numpy as np  # for evaluating conditions


def cached_eval(condition, parameters=params):
    """
    This function first test if a condition was previouly evaluated (x['AGE'] > 3) & (x['HAIR_COLOR'] == 'dark'),
    so the cached value is returned.
    Otherwise, it evaluates the condition, caches the result, and returns the result

    :param condition: The condition to be evaluated on a dataset, for instance
     (x['AGE'] > 3) & (x['HAIR_COLOR'] == 'dark')
    :side-effect: the evaluation of the condition is cached
    :return: The evaluation of the condition on the dataset params['FITNESS_FUNCTION'].training_in
    """
    if params['CACHE'] and condition in cache:
        return cache[condition]
    else:
        # noinspection PyUnusedLocal
        x = parameters['FITNESS_FUNCTION'].training_in
        try:
            cache[condition] = eval(condition)
        except TypeError:
            raise
        return cache[condition]


def cached_eval_single_condition(condition, parameters=params):
    """
    This function first test if a condition was previouly evaluated (x['AGE'] > 3),
    so the cached value is returned.
    Otherwise, it evaluates the condition, caches the result, and returns the result

    :param condition: The condition to be evaluated on a dataset, for instance x['AGE'] > 3
    :side-effect: the evaluation of the condition is cached
    :return: The evaluation of the condition on the dataset params['FITNESS_FUNCTION'].training_in
    """
    if params['CACHE'] and condition in aux['cache_single_condition']:
        return aux['cache_single_condition'][condition]
    else:
        # noinspection PyUnusedLocal
        x = parameters['FITNESS_FUNCTION'].training_in
        try:
            aux['cache_single_condition'][condition] = eval(condition)
        except (NameError, SyntaxError):
            print('{',condition,'}\n',x.columns, flush=True)
            raise
        return aux['cache_single_condition'][condition]
