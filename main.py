import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from id3 import DecisionTree

def create_dataframe(data, target, feature_names) -> pd.DataFrame:
    data = pd.DataFrame(data, columns=feature_names)
    data['target'] = pd.Series(target)
    return data

def test_dataframe(test_df: pd.DataFrame, tree: DecisionTree):
    correct = 0
    for _, row in test_df.iterrows():
        predicted = tree.predict(row)
        if row['target'] in predicted:
            correct += 1
    return correct / len(test_df)


df = create_dataframe(datasets.load_iris()['data'], datasets.load_iris()['target'], datasets.load_iris()['feature_names'])


train, other  = train_test_split(df, train_size=0.7)
validation, test = train_test_split(other, test_size=0.5)

tree = DecisionTree(
    df=train,
    columns_names=datasets.load_iris()['feature_names'],
    max_depth=2,
    number_of_childs=3
)

# print(test_dataframe(validation, tree))

print(tree)

print(tree[0])
