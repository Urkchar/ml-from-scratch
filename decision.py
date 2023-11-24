from pandas import Series
import pandas as pd


class Condition():
    def __init__(self, left_operand, right_operand, operator):
        # print(right_operand)
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.operator = operator

    def __call__(self):
        return self.operator(self.left_operand, self.right_operand)
    
    # def __str__(self):
    #     # return f"{self.operator}({self.left_operand}, {self.right_operand})"
    #     return str(self.operator(self.left_operand, self.right_operand))


class Decision():
    def __init__(self, feature: Series, Y: Series, condition: Condition):
        self.feature = feature
        self.Y = Y
        self.condition = condition
        self.left_child = None
        self.right_child = None

    @property
    def impurity(self):
        """Return the Gini Impurity of the decision."""
        # print(self.condition)
        # Separate the left and the right leaves
        df = pd.concat([self.feature, self.Y], axis=1)
        left = df[self.condition()]
        # print(f"left: \n{left}")
        right = df[~self.condition()]
        # print(f"right: \n{right}")
        # return
        total_impurity = 0
        sample_count_of_leaves = len(self.feature)
        # print(f"sample_count_of_leaves: {sample_count_of_leaves}")
        for leaf in [left, right]:
            print(f"leaf: \n{leaf}")
            leaf_impurity = self.leaf_impurity(leaf)
            print(f"leaf_impurity: {leaf_impurity}")
            # return
            sample_count_of_leaf = len(leaf)
            weighted_impurity = (sample_count_of_leaf / sample_count_of_leaves) * leaf_impurity
            total_impurity += weighted_impurity
        # Calculate the wegihted impurity for the left leaf
        # classes = left[self.Y.name].unique()
        # print(classes)
        print(f"total_impurity: {total_impurity}")
        return total_impurity
    
    def leaf_impurity(self, leaf):
        """Return the impurity for a single leaf."""
        squared_class_probabilities = []
        classes = leaf[self.Y.name].unique()
        # print(f"classes: {classes}")
        for cls in classes:
            print(f"cls: {cls}")
            # Count the number of positive samples of the class in the leaf
            class_positive = leaf[leaf[self.Y.name] == cls]
            # print(f"class_positive: \n{class_positive}")
            class_positive_count = len(class_positive)
            print(f"class_positive_count: {class_positive_count}")

            # Count the total number of samples in the leaf
            total_samples = len(leaf)
            print(f"total_samples: {total_samples}")

            # Calculate the squared probability of cls
            cls_probability = (class_positive_count / total_samples) ** 2

            squared_class_probabilities.append(cls_probability)

        return 1 - sum(squared_class_probabilities)
