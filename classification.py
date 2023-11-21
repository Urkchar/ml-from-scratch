from pandas import Series, DataFrame
import pandas as pd
# import numpy as np


class DecisionTreeClassifier():
    def __init__(self, x: DataFrame, Y: Series):
        self.x = x
        self.Y = Y
        self.classes = Y.unique()
        self.n_classes = len(self.classes)
        self.n_features = len(x.columns)

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
    
    def gini(self, feature_name: str):
        """Return the weighted average of Gini Impurities for the Leaves of a candidate feature."""
        feature_series = self.x[feature_name]
        feature_type = feature_series.dtype
        # print(feature_type)

        if feature_type == "object":
            # print("gini - gini_categorical called.")
            leaves = self.create_categorical_leaves(feature_series)
        else:
            leaves = self.create_continuous_leaves(feature_series, feature_name)

        return self.total_gini_impurity(leaves)
    
    def total_gini_impurity(self, leaves: list) -> float:
        # print(leaves)
        total = 0
        for leaf in leaves:
            weight = sum(leaf) / len(self.x)
            total += weight * self.leaf_gini(leaf)

        return total
    
    def create_leaf(self, intersector: Series):
        # print(type(intersector))
        leaf = []

        for cls in self.classes:
            class_positive = self.Y == cls
            intersection = class_positive & intersector
            if True in intersection.value_counts():
                leaf.append(intersection.value_counts()[True])
            else:
                leaf.append(0)

        return leaf

    def create_categorical_leaves(self, feature: Series) -> list:
        """Return a list of leaves for a categorical column."""
        categories = feature.unique()
        # print(f"create_categorical_leaves - categories: {categories}")
        leaves = []
        for category in categories:
            # print(f"category: {category}")
            leaf = self.create_leaf(feature == category)
            leaves.append(leaf)

        return leaves

    def create_continuous_leaves(self, feature: Series, feature_name: str) -> list:
        """Return the weighted average of Gini Impurities for a continuous column."""
        # Recombine feature and Y
        df = pd.concat([feature, self.Y], axis=1)

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

            total_gini_impurity = self.total_gini_impurity([positive_leaf, negative_leaf])
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

        positive_leaf = self.create_leaf(feature < best_threshold)
        negative_leaf = self.create_leaf(feature >= best_threshold)

        return [positive_leaf, negative_leaf]



            

    # def create_continuous_leaf(self, feature: Series, threshold: float):
    #     less_than_threshold = feature < threshold

