from classification import DecisionTreeClassifier
from decision import Decision, Condition
import pandas as pd


def main():
    df = pd.read_csv("cool_as_ice.csv")
    # print(df)
    # df.info()

    predictors_df = df.drop("loves_cool_as_ice", axis=1)
    response = df["loves_cool_as_ice"]

    algorithm = DecisionTreeClassifier()
    # print(algorithm.gini(predictors_df, "loves_popcorn", response))
    # print(algorithm.gini(predictors_df, "loves_soda", response))
    # print(algorithm.gini("age"))
    # algorithm.gini("age")
    # leaf = (1, 3)
    # print(algorithm.gini(leaf))
    # leaf = (2, 1)
    # print(algorithm.gini(leaf))
    # print(algorithm._determine_best_feature(predictors_df.columns))

    # d = Decision(7, 8, lambda x, y: x == y)
    # print(d())

    # condition = Condition(predictors_df["loves_popcorn"], "yes", lambda x, y: x == y)
    # print(condition())

    algorithm.fit(predictors_df, response)


if __name__ == "__main__":
    main()
