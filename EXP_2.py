import pandas as pd
import numpy as np

def read_data():
    df = pd.read_csv("CarPrice.csv")  # Corrected file name
    return df

def candidate_elimination(data, target_column):
    attributes = data.iloc[:, :-1].values  # All columns except the last one
    target = data.iloc[:, target_column].values  # Target column (last column - price)

    specific_hypothesis = attributes[0].copy()
    general_hypothesis = [["?" for _ in range(len(specific_hypothesis))] for _ in range(len(specific_hypothesis))]

    for i, instance in enumerate(attributes):
        if target[i] == "yes":  # Assuming 'yes' is for positive examples
            for j in range(len(specific_hypothesis)):
                if instance[j] != specific_hypothesis[j]:
                    specific_hypothesis[j] = "?"
                    general_hypothesis[j][j] = "?"
        else:
            for j in range(len(specific_hypothesis)):
                if instance[j] != specific_hypothesis[j]:
                    general_hypothesis[j][j] = specific_hypothesis[j]
                else:
                    general_hypothesis[j][j] = "?"

    general_hypothesis = [gh for gh in general_hypothesis if gh != ["?" for _ in range(len(specific_hypothesis))]]

    return specific_hypothesis, general_hypothesis

if __name__ == "__main__":
    data = read_data()
    target_column = len(data.columns) - 1  # Assuming price is the target variable
    specific_h, general_h = candidate_elimination(data, target_column)

    print("Final Specific Hypothesis:", specific_h)
    print("Final General Hypothesis:", general_h)
