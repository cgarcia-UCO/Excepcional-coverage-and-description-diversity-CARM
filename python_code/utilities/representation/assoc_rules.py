import random
import numpy as np
import re
import sys
sys.path.insert(2, 'python_code')
from algorithm.parameters import params
from utilities.misc import cached_eval, cached_eval_single_condition
from collections import Counter

# noinspection RegExpRedundantEscape
def simplify_not_condition(condition):
    """
    This method receives a condition and, in case it is a not condition,
    transforms it to its positive equivalent.

    @param condition: Just a single condition, for instance (~ x['AGE'] > 3)
    :return: the equivalent positive condition. For instance, x['AGE'] <= 3
    """

    # noinspection RegExpRedundantEscape
    def simplify_not_lessequal_condition(this_condition):
        """
        This method receives a not less-than-or-equal-to condition and returns its
        positive greater than equivalent

        @param this_condition: Just a single not less-than-or-equal-to condition, for instance (~ x['AGE'] <= 3)
        :return: the equivalent positive condition. For instance, x['AGE'] > 3
        """

        feature = re.search("\(\~ \(?\((x\[\'[^\']+\'\]) <=", this_condition).groups()[0]
        value = re.search(" <= (\-?\d+(\.\d+)?)", this_condition).groups()[0]
        return '(' + feature + ' > ' + value + ')'

    # noinspection RegExpRedundantEscape

    def simplify_not_greater_condition(this_condition):
        """
        This method receives a not greater-than condition and returns its
        positive less-than-or-equal-to equivalent

        @param this_condition: Just a single not greater-than condition, for instance (~ x['AGE'] > 3)
        :return: the equivalent positive condition. For instance, x['AGE'] <= 3
        """

        feature = re.search("\(\~ \(?\((x\[\'[^\']+\'\]) > ", this_condition).groups()[0]
        value = re.search(" > (\-?\d+(\.\d+)?)", this_condition).groups()[0]
        return '(' + feature + ' <= ' + value + ')'

    # noinspection RegExpRedundantEscape

    def simplify_not_unequal_condition(this_condition):
        """
        This method receives a not unequal condition and returns its
        positive equal equivalent

        @param this_condition: Just a single not unequal condition, for instance (~ x['EYE_COLOR'] != 'blue')
        :return: the equivalent positive condition. For instance, x['EYE_COLOR'] == 'blue'
        """

        feature = re.search("\(\~ \(?\((x\[\'[^\']+\'\]) != ", this_condition).groups()[0]
        value = re.search(" != (\'[^\']+\')\)", this_condition).groups()[0]
        return '(' + feature + ' == ' + value + ')'

    # noinspection RegExpRedundantEscape

    def simplify_not_equal_condition(this_condition):
        """
        This method receives a not equal condition and returns its
        positive unequal equivalent

        @param this_condition: Just a single not equal condition, for instance (~ x['EYE_COLOR'] == 'blue')
        :return: the equivalent positive condition. For instance, x['EYE_COLOR'] != 'blue'
        """

        feature = re.search("\(\~ \(?\((x\[\'[^\']+\'\]) == ", this_condition).groups()[0]
        value = re.search(" == (\'[^\']+\')\)", this_condition).groups()[0]
        return '(' + feature + ' != ' + value + ')'

    try:
        if re.match("\(\~ \(?\(x\[\'[^\']+\'\] <= ", condition):
            return simplify_not_lessequal_condition(condition)
        elif re.match("\(\~ \(?\(x\[\'[^\']+\'\] > ", condition):
            return simplify_not_greater_condition(condition)
        elif re.match("\(\~ \(?\(x\[\'[^\']+\'\] != ", condition):
            return simplify_not_unequal_condition(condition)
        elif re.match("\(\~ \(?\(x\[\'[^\']+\'\] == ", condition):
            return simplify_not_equal_condition(condition)
    except TypeError:
        print(condition, flush=True)
        raise
    return condition


class Rule:
    """
    This class represents a class association rule, which is a structure
    with an IF part (antecedent) with a set of conditions and a THEN
    part (consequent), with a class label
    """

    @classmethod
    def __get_validators__(cls):
        """
        returns validatos for pydantic, if used
        """
        yield cls.validate

    @classmethod
    def validate(cls, val):
        """
        just check if it is a Rule
        """
        if not isinstance(val, Rule):
            raise TypeError('Rule required')
        else:
            return val

    def __init__(self, parameters = params):
        """
        Initializer. No conditions (so it covers every pattern), and no consequent set.
        """
        self.conditions = []
        self.consequent = None
        self.parameters = parameters
        self.covered_patterns = np.array([True for _ in range(parameters['FITNESS_FUNCTION'].training_in.shape[0])])
        self.used_features = None
        self.confidence = None

    # noinspection RegExpRedundantEscape
    def get_used_features(self):
        """
        This method returns a boolean array with True on the positions corresponding
        to the feature of the datasets that the rule uses in its conditions.

        For instance, if the rule uses feature 'COLOR' and this feature is the fourth
        in the dataset, then, the fourth value of the array (index 3) would be True

        :return: A boolean numpy ndarray with True for the corresponding features of the dataset used in the rule
        """

        # noinspection PyUnresolvedReferences
        if not hasattr(self, "used_features") or self.used_features is None:  # Compute them
            x = self.parameters['FITNESS_FUNCTION'].training_in
            used_features = {re.search("x\[\'([^\']+)\'\]", i).groups()[0]
                                  for i in self.conditions}
            self.used_features = np.array([i in used_features for i in x.columns])

        return self.used_features

    def get_covered_patterns(self):
        """
        This method returns a boolean array with True on the target patterns covered by the rule.
        False is set on every non-target pattern. If not computed before, it is computed and stored

        :return: A boolean array
        """
        if not hasattr(self, "covered_patterns") or self.covered_patterns is None:
            y = self.parameters['FITNESS_FUNCTION'].training_exp
            target_class = self.parameters['POSITIVE_CLASS']
            index_target_patterns = y == target_class
            self.covered_patterns = cached_eval(self.get_antecedent(), self.parameters)
            self.covered_patterns = self.covered_patterns & index_target_patterns

        return self.covered_patterns

    # noinspection PyPep8Naming
    def get_LHS_support(self):
        """
        This method returns the LHS support of the rule, the rate of covered patterns.

        @return: The support of the rule
        """
        x = params['FITNESS_FUNCTION'].training_in
        covered_patterns = cached_eval(self.get_antecedent())
        return sum(covered_patterns) / x.shape[0]

    def get_confidence(self):
        """
        This method returns the confidence of the rule. The rate of covered patterns with target feature
        equal to the prediction of the rule. If not computed before, it is computed and stored

        :return: The confidence of the rule
        """
        if not hasattr(self, "confidence") or self.confidence is None:
            y = self.parameters['FITNESS_FUNCTION'].training_exp
            covered_patterns = cached_eval(self.get_antecedent(), self.parameters)
            true_class = y[covered_patterns]
            predicted_class = self.get_consequent().replace('\'', '')
            hits = true_class == predicted_class

            # if np.sum(hits) == 0:
            #     print(self.get_antecedent())
            #     raise Exception

            num_covered_patterns = np.sum(covered_patterns)
            if num_covered_patterns == 0:
                self.confidence = 0
            else:
                try:
                    self.confidence = np.sum(hits) / num_covered_patterns
                except (ZeroDivisionError, FloatingPointError):
                    self.confidence = 0

        return self.confidence

    def get_lift(self):
        """
        This method returns the lift value of the rule
        @return:
        """

        y = self.parameters['FITNESS_FUNCTION'].training_exp
        confidence = self.get_confidence()
        p_target = np.sum(y == self.get_consequent()) / len(y)

        if p_target <= 0:
            raise Exception

        return confidence / p_target

    def get_significance(self):
        """
        This method returns the significance of the rule, as defined in:
         Kloesgen W (1996) Explora: a multipattern and multistrategy discovery assistant. In: Advances in         Knowledge discovery and data mining. American Association for Artificial Intelligence, pp 249–271
         Franciso Herrera · Cristóbal José Carmona · Pedro González · María José del JesusAn overview on subgroup discovery: foundations and applications. Knowl Inf Syst (2011) 29:495–525 DOI 10.1007/s10115-010-0356-2

        @return:
        """
        y = self.parameters['FITNESS_FUNCTION'].training_exp
        covered_patterns = cached_eval(self.get_antecedent(), self.parameters)
        class_values = set(y)
        result = 0
        p_cond = np.sum(covered_patterns) / len(y)

        for i in class_values:
            covered_of_this_class = np.sum(covered_patterns & (y == i))
            if covered_of_this_class > 0:
                result += covered_of_this_class * np.log(covered_of_this_class / (np.sum(y == i)*p_cond))

        result *= 2

        return result


    def get_WRAcc(self):
        """
        This method returns the WRAcc of the rule, as defined in:
         EXPLORA (Klosgen, ¨ 1996) and MIDOS (Wrobel, 1997)

        @return:
        """
        y = self.parameters['FITNESS_FUNCTION'].training_exp
        covered_patterns = cached_eval(self.get_antecedent(), self.parameters)
        num_covered_patterns = np.sum(covered_patterns)
        p_cond = num_covered_patterns / len(y)
        predicted_class = self.get_consequent().replace('\'', '')
        p_class = np.sum((y == predicted_class) |
                         (y == "'" + predicted_class + "'")) / len(y)

        if num_covered_patterns > 0:
            p_class_cond = np.sum((y[covered_patterns] == predicted_class) |
                                  (y[covered_patterns] == "'" + predicted_class + "'")) /\
                           num_covered_patterns
        else:
            p_class_cond = np.nan


        result = p_cond * (p_class_cond - p_class)

        return result

    # noinspection RegExpRedundantEscape
    def add_condition(self, condition):
        """
        This method adds a condition to the Rule self. It is added at the beginning (for printing purposes).
        It is also simplify in case of having not conditions.

        In addition, it updates the patterns covered and the features used.

        @param condition: A condition to be added into self.
        :return: None
        """
        condition = simplify_not_condition(condition)
        self.get_used_features()  # This call is just to make sure that the property used_features is set
        try:
            self.covered_patterns = self.covered_patterns & cached_eval_single_condition(condition, self.parameters)
        except SyntaxError:
            print('{',condition,'}')
            raise
        self.conditions.insert(0, condition)

        # Update the features used
        x_columns = self.parameters['FITNESS_FUNCTION'].training_in.columns
        feature = re.search("x\[\'([^\']+)\'\]", condition).groups()[0]
        self.used_features[np.where(x_columns == feature)[0][0]] = True

    def set_consequent(self, consequent):
        """
        This method sets the consequent of the Rule
        
        :return: None
        """
        self.consequent = consequent
        try:
            self.consequent = self.consequent.replace("'", "")
        except AttributeError:
            pass
        
        # Set covered patterns and confidence as invalid
        self.covered_patterns = None
        self.get_covered_patterns()
        self.confidence = None

    def get_consequent(self):
        """
        Simple getter
        
        :return: the consequent of the rule
        """
        return self.consequent

    def __str__(self):
        """
        Method to make Rules printable

        :return: a string representation of the rule
        """
        return '&'.join(self.conditions) + ' => ' + self.consequent

    def get_antecedent(self):
        """
        Return a string connecting the conditions of the rule (usually to be evaluated).
        In case it has got no conditions, this returns a code to produce a boolean array
        with True for every pattern in the dataset
        """
        if len(self.conditions) <= 0:
            return '(np.array([True for i in range(x.shape[0])]))'
        else:
            return '&'.join(self.conditions)

    def __eq__(self, other):
        """
        This method can be used to check if two Rules are the same. It checks the conditions, and the consequent

        :return: True/False
        """
        if isinstance(other, Rule):
            if self.consequent != other.consequent:
                return False
            else:
                # I have not considered comparing the lengths because a rule might have the same condition
                # multiple times. I think that simplifying the rule at construction could be interesting
                for i in self.conditions:
                    found = len([j for j in other.conditions if j == i])

                    if found <= 0:
                        return False

                for i in other.conditions:
                    found = len([j for j in self.conditions if j == i])

                    if found <= 0:
                        return False

                return True

        else:
            return NotImplemented

    def dominates(self, another):
        """
        This method implements a simple dominance criterion between two Rules.
        The idea is that self dominates another if:
        - has the same consequent
        - has better or equal confidence
        - covers all the patterns covered by another
        - does not use features out of those in another

        IMPORTANT: Notice that self dominates itself! This is different from the standard dominance criterion in
        multiobjective problems

        @param another: a Rule
        :return: True/False
        """

        # If consequents are different, return False
        if self.get_consequent() != another.get_consequent():
            return False

        # If the confidence of another is higher, return False
        if self.get_confidence() < another.get_confidence():
            return False

        # If another covers some patterns not covered by self, return False
        if sum(another.get_covered_patterns() & (~ (self.get_covered_patterns()))) > 0:
            return False

        # If self uses a feature not in another, return False
        if sum(self.get_used_features() & (~ another.get_used_features())) > 0:
            return False

        return True


class RuleSet:
    """
    This class represents a set of Rules
    """

    def __init__(self, parameters= params):
        """
        Initialiser
        """
        self.rules = []
        self.parameters = parameters
        self.uncovered_target_patterns = (parameters['FITNESS_FUNCTION'].training_exp == parameters['POSITIVE_CLASS'])|\
                                         (parameters['FITNESS_FUNCTION'].training_exp == parameters['POSITIVE_CLASS'].replace("'",''))

    def print_rules_and_metrics(self, file_descriptor):

        for i in self.rules:
            file_descriptor.write(str(i) +
                                  ' LHS Supp:' + str(i.get_LHS_support()) +
                                  ' Conf: ' + str(i.get_confidence()) + '\n')

    def get_support(self, x):
        """
        This method returns the support of the set of rules (just left-hand-side / antecedent)

        @param x: dataset where rules are evaluated on
        :return: A list with the support values of the rules
        """
        values = []

        num_rows = x.shape[0]

        for i in self.rules:
            values.append(sum(cached_eval(i.get_antecedent(),
                                          self.parameters)) / num_rows)

        return values

    def get_confidence(self):
        """
        This function returns the confidence values of a set of rules (just left-hand-side / antecedent)

        :return: A list with the confidence values of the rules
        """
        values = []

        for i in self.rules:
            values.append(i.get_confidence())

        return values

    def get_lengths(self):
        """
        This function returns the number of rules per number of conditions

        :return: A list with the number of conditions as index, and the number of rules
         per number of conditions as value
        """

        num_used_feats = []

        for i in self.rules:
            num_used_feats.append(np.sum(i.get_used_features()))

        return sorted(Counter(num_used_feats).items())


    def get_num_times_used_attributes(self):
        """
        This function returns a vector with the number of times each feature of the dataset is used

        @return: An integer vector
        """

        features = params['FITNESS_FUNCTION'].training_in.columns
        attributes = np.array([0 for _ in features])

        for i in self.rules:
            new_attributes = i.get_used_features()
            attributes = attributes + new_attributes

        return attributes

    def get_used_attributes(self):
        """
        This function returns the used attributes in the set of rules

        @return: A boolean array with True on the used features
        """

        features = self.parameters['FITNESS_FUNCTION'].training_in.columns
        attributes = np.array([False for _ in features])

        for i in self.rules:
            new_attributes = i.get_used_features()
            attributes = attributes | new_attributes

        return attributes

    def get_ratio_used_attributes(self):
        features = self.parameters['FITNESS_FUNCTION'].training_in.columns
        used_features = np.sum(self.get_used_attributes())
        return used_features / len(features)


    def get_lift_values(self):
        values = []
        for i in self.rules:
            values.append(i.get_lift())
        return values


    def get_significances(self):
        significances = []
        for i in self.rules:
            significances.append(i.get_significance())
        return significances


    def get_wracc_values(self):
        wracc_values = []
        for i in self.rules:
            wracc_values.append(i.get_WRAcc())
        return wracc_values


    def get_freq_attributes(self, dataset):
        """
        This function returns a list with the number of rules in rules that use each feature in dataset

        @param dataset: The dataset that informs of the possible features
        @return: A list with the number of rules that use each feature of the dataset
        """
        values = []

        if len(self.rules) > 0:
            for i in dataset.columns:
                # Number of rules with a condition involving the dataset feature i
                frequency = len([i_rules for i_rules in self.rules if
                                 len([j for j in i_rules.conditions if i in j]) > 0])
                values.append(frequency / len(self.rules))

        return np.array(values)

    def filter_duplicates(self):
        """
        This method removes duplicated rules from self

        :return: None
        """

        removing_indexes = set()

        for index_i, i in enumerate(self.rules):
            for index_j, j in enumerate(self.rules[index_i+1:]):
                if j == i:
                    removing_indexes.add(index_i + index_j)

        init_num_rules = len(self.rules)
        self.rules = [i for index, i in enumerate(self.rules) if index not in removing_indexes]
        assert len(self.rules) == init_num_rules - len(removing_indexes)

    def filter_by_confidence(self, threshold):
        """
        This method removes Rules with a confidence below a given threshold

        @param threshold: Minimum confidence for the rules to not be filtered out
        :return: None
        """

        self.rules = [i for i in self.rules if i.get_confidence() >= threshold]

    def filter_by_consequent(self, target):
        """
        This method removes Rules with a consequent different from target

        @param target: the target value interested in
        :return None:
        """
        self.rules = [i for i in self.rules
                      if i.get_consequent() == target or i.get_consequent() == str(target)]

    def add_rule(self, rule):
        """
        This method just adds rules into the set. Though simple, exists to allow method override

        @param rule: the rule to be added
        :return: None
        """
        self.rules.append(rule)

        # Update the patterns not covered by substracting those covered by rule.
        # Notice that not any covered pattern becomes uncovered, because the associated rule can not be
        # dominated
        if rule.get_consequent() == self.parameters['POSITIVE_CLASS'] or \
                rule.get_consequent() == self.parameters['POSITIVE_CLASS'].replace("'",''):
            new_covered = cached_eval(rule.get_antecedent(), self.parameters)
            self.uncovered_target_patterns = self.uncovered_target_patterns & (~ new_covered)

    def _get_consequent(self, tree):
        """
        This method looks for the consequent of a leaf or a node very close to a leaf (for instance, its parent)

        @param tree: a subtree. It should be a leaf or a node leading to just one leaf
        :return: the target value of the leaf
        """
        if len(tree.children) <= 0:
            return tree.root
        elif len(tree.children) == 1:
            return self._get_consequent(tree.children[0])
        else:
            raise Exception('Weird consequent: ' + str(tree))

    def _get_conditions(self, tree):
        """
        This method reads the conditions of a split node. It should not be used on a node which does not lead
        to a splitting condition.

        @param tree: The split node
        :return: a set of conditions on this node
        """
        set_conds = []
        self._r_get_conditions(tree, set_conds)
        return set_conds

    def _r_get_conditions(self, tree, set_conds=None, reading_cond=-1, depth=0):
        """
        This recursive function produces a list with the conditions in a <cond> subtree.
        Basically, it just outputs the leaves in preorder. This was particularly useful in case of
        conditions made of conjuntion of conditions.

        @param tree: The split node
        @param set_conds: Do not use outside recursive calls. It is being grown with conditions from ancestor nodes
        @param reading_cond: Do not use outside recursive calls. It informs the condition that is being constructed
        @param depth: Do not use outside recursive calls. It informs of the level of recursion
        :return: a tuple with the set of conditions and a dummy value (I think it should be -1 in the main call)
        """
        current_cond = ''

        if set_conds is None:
            set_conds = []

        # If the node has children, we have to call recursively the method
        if len(tree.children) > 0:

            # For each child, gets the condition and concatenate them for the following calls
            for index, i in enumerate(tree.children):
                current_output, reading_cond = self._r_get_conditions(i, set_conds, reading_cond, depth + 1)

                if reading_cond >= 0:
                    current_cond += current_output

            # At this point, we have completely read a condition, so it has to be added to set_conds
            if reading_cond > depth:
                set_conds.append(current_cond)
                current_cond = ''
                reading_cond = -1

        else:
            # If the node has not got children, we have finished reading something.
            # We are going to return it, but we check if we are starting to read a new condition
            # reading_cond == -1 means we were not reading any condition.
            # Then, it is set to the current depth in order to close the current codntion when we go back to
            # this depth
            current_cond = tree.root

            if current_cond != ' & ':
                if reading_cond == -1:
                    reading_cond = depth

        return current_cond, reading_cond


