from classification import DecisionTreeClassifier
import pandas as pd


def main():
    df = pd.read_csv("cool_as_ice.csv")
    # print(df)
    # df.info()

    predictors_df = df.drop("loves_cool_as_ice", axis=1)
    response = df["loves_cool_as_ice"]

    algorithm = DecisionTreeClassifier(predictors_df, response)
    # print(algorithm.gini("loves_popcorn"))
    # print(algorithm.gini("loves_soda"))
    # print(algorithm.gini("age"))
    # algorithm.gini("age")
    # leaf = (1, 3)
    # print(algorithm.gini(leaf))
    # leaf = (2, 1)
    # print(algorithm.gini(leaf))
    print(algorithm._determine_best_feature(predictors_df.columns))


if __name__ == "__main__":
    main()
