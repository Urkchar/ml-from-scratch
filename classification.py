from pandas import Series, DataFrame
import pandas as pd
from node import Node
from decision import Decision, Condition
# import numpy as np


class DecisionTreeClassifier():
    def __init__(self):
        self.root = None

    def fit(self, x: DataFrame, Y: Series, min_samples: int = 2):
        # self.root = self.create_node(x, Y)
        if len(x) < min_samples:
            return None
        
        df = pd.concat([x, Y], axis=1)
        print(df)
        
        # Determine which category of which feature gives the best split
        best_decision = self.get_best_decision(x, Y)
        # print("Done.")
        # return

        # Select the rows that go to the left...
        # print(best_decision.condition())
        decision_positive = df[best_decision.condition()]
        # print(f"decision_positive: \n{decision_positive}")
        # and to the right
        decision_negative = df[~best_decision.condition()]   # Inverse the boolean series with ~
        # print(f"decision_negative: \n{decision_negative}")

        # print(decision_positive.drop(Y.name, axis=1))
        # Create the left child
        x_positive = decision_positive.drop(Y.name, axis=1)
        Y_positive = decision_positive[Y.name]
        best_decision.left_child = self.get_best_decision(x_positive, Y_positive)

        # Create the right child
        x_negative = decision_negative.drop(Y.name, axis=1)
        Y_negative = decision_negative[Y.name]
        best_decision.right_child = self.get_best_decision(x_negative, Y_negative)

        self.root = best_decision

    def get_best_decision(self, x: DataFrame, Y: Series) -> Decision:
        """Return the best Decision of all features."""
        print(f"len(x): {len(x)}")
        if len(x) < 2:
            return
        candidate_feature_decisions = []
        for column_name, feature in x.items():
            print(f"Creating Decisions for {column_name}...")
            feature_decisions = self.create_decisions(feature, Y)
            print(f"Decisions for {column_name} created.")
            
            # Get the best decision for that feature
            sorted_feature_decisions = sorted(feature_decisions, key=lambda x: x.impurity)
            # print(feature_decisions)
            best_feature_decision = sorted_feature_decisions[0]

            candidate_feature_decisions.append(best_feature_decision)

        sorted_decisions = sorted(candidate_feature_decisions, key=lambda x: x.impurity)
        best_decision = sorted_decisions[0]

        return best_decision

    def create_decisions(self, feature: Series, Y: Series) -> list:
        """Return a list of possible decisions for the feature."""
        decisions = []
        # For categorical columns
        if feature.dtype == "object":
            for category in feature.unique():
                decision = Decision(feature, Y, Condition(feature, category, lambda x, y: x==y))
                decisions.append(decision)
        # For continuous columns
        else:
            feature = feature.sort_values()

            # Get the average value between each numeric value
            averages = feature.rolling(2).mean().dropna()

            # Calculate the Gini Impurity values for each average value
            for average in averages:
                decision = Decision(feature, Y, Condition(feature, average, lambda x, y: x<y))
                decisions.append(decision)
            # raise NotImplementedError
        
        return decisions
    
    def get_best_category(self, feature: Series, Y: Series):
        """Return the Decision that gives the lowest impurity for the feature."""
        decisions = []
        if feature.dtype == "object":
            categories = feature.unique()
            for category in categories:
                decision = Decision(feature, Y, lambda x, y: x == y)
        else:
            feature = feature.sort_values()
            categories = feature.rolling(2).mean().dropna()

    def create_node(self, features: DataFrame, response: Series):
        if len(features) == 0:
            return None
        best_feature_name = self._determine_best_feature(features)
        best_feature = features[best_feature_name]
        features = features.drop(best_feature, axis=1)

        return self.create_node(features, response)

    def _determine_best_feature(self, features):
        """Return which feature out of the list of features predicts the target best."""
        feature_impurity_pairs = []
        for feature in features:
            impurity = self.gini(feature)
            feature_impurity_pairs.append([feature, impurity])

        sorted_pairs = sorted(feature_impurity_pairs, key=lambda x: x[1])
        best_pair = sorted_pairs[0]
        best_feature = best_pair[0]

        return best_feature

    def leaf_gini(self, leaf: list) -> float:
        """Return the Gini Impurity for a leaf of the tree.
        1 - (probability(Class 1) ** 2 - probability(Class 2) ** 2 ... - probability(Class N) ** 2"""
        impurity = 1
        num_samples = sum(leaf)
        num_classes = len(leaf)
        for i in range(num_classes):
            impurity -= (leaf[i] / (num_samples)) ** 2

        return impurity
    
    def gini(self, features: Series, feature_name: str, response: Series):
        """Return the weighted average of Gini Impurities for the Leaves of a candidate feature."""
        feature_series = features[feature_name]
        feature_type = feature_series.dtype
        # print(feature_type)

        if feature_type == "object":
            # print("gini - gini_categorical called.")
            leaves = self.create_categorical_leaves(feature_series, response)
        else:
            leaves = self.create_continuous_leaves(feature_series, feature_name)

        return self.total_gini_impurity(leaves)
    
    def total_gini_impurity(self, leaves: list, features: Series) -> float:
        # print(leaves)
        total = 0
        for leaf in leaves:
            weight = sum(leaf) / len(features)
            total += weight * self.leaf_gini(leaf)

        return total
    
    def create_leaf(self, intersector: Series, response: Series):
        # print(type(intersector))
        classes = response.unique()
        leaf = []

        for cls in classes:
            class_positive = response == cls
            intersection = class_positive & intersector
            if True in intersection.value_counts():
                leaf.append(intersection.value_counts()[True])
            else:
                leaf.append(0)

        return leaf

    def create_categorical_leaves(self, feature: Series, response: Series) -> list:
        """Return a list of leaves for a categorical column."""
        categories = feature.unique()
        # print(f"create_categorical_leaves - categories: {categories}")
        leaves = []
        for category in categories:
            # print(f"category: {category}")
            decision = Decision(feature, category, lambda x, y: x == y)
            leaf = self.create_leaf(decision(), response)
            leaves.append(leaf)

        return leaves

    def create_continuous_leaves(self, feature: Series, feature_name: str, features: Series, response: Series) -> list:
        """Return the weighted average of Gini Impurities for a continuous column."""
        # Recombine feature and Y
        df = pd.concat([feature, response], axis=1)

        # Sort the rows by feature from lowest value to highest value
        df = df.sort_values(by=feature_name)
        # print(df)

        # Calculate the average for all adjacent values
        averages = df[feature_name].rolling(2).mean().dropna()
        # print(averages)

        # Calculate the Gini Impurity values for each average value
        gini_impurities = []
        for average in averages:
            # print(f"average: {average}")
            # Only two possibilites for a threshold
            positive_leaf = self.create_leaf(feature < average)
            negative_leaf = self.create_leaf(feature >= average)

            total_gini_impurity = self.total_gini_impurity([positive_leaf, negative_leaf], features)
            gini_impurities.append(total_gini_impurity)
        # print(gini_impurities)
        zipped = zip(averages, gini_impurities)
        # print(zipped)
        # for average, impurity in zipped:
        #     print(f"{average}: {impurity}")

        # Find the lowest Gini Impurity
        sorted_pairs = sorted(zipped, key=lambda x: x[1])
        # print(sorted_pairs)
        best_pair = sorted_pairs[0]
        best_threshold = best_pair[0]   # (threshold, impurity)

        print(type(feature < best_threshold))
        positive_leaf = self.create_leaf(feature < best_threshold)
        negative_leaf = self.create_leaf(feature >= best_threshold)

        return [positive_leaf, negative_leaf]



            

    # def create_continuous_leaf(self, feature: Series, threshold: float):
    #     less_than_threshold = feature < threshold

