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
        if len(predicted) == 1:
            if row['target'] == predicted[0]:
                correct += 1
        # if row['target'] in predicted:
        #     correct += 1
    return correct / len(test_df)


df = create_dataframe(datasets.load_iris()['data'], datasets.load_iris()['target'], datasets.load_iris()['feature_names'])

train, other  = train_test_split(df, train_size=0.7)
validation, test = train_test_split(other, test_size=0.5)

trees = []

for depth in range(2, 6):
    for childs in range(2, 6):
        tree = DecisionTree(
            df=train,
            columns_names=datasets.load_iris()['feature_names'],
            max_depth=depth,
            number_of_childs=childs,
            min_impurity_split=0.2
        )
        trees.append((tree, depth, childs))

ratings = []

for tree in trees:
    rating = test_dataframe(validation, tree[0])
    ratings.append(rating)

best_index = ratings.index(max(ratings))
best_tree = trees[best_index]

print('Depth\tChilds\tRating')
for tree, rating in zip(trees, ratings):
    print(f'{tree[1]}\t{tree[2]}\t{rating}')

print(f'\nBest hyperparameters:\nmax_depth = {best_tree[1]}\nnumber_of_childs = {best_tree[2]}')

final_rating = test_dataframe(test, best_tree[0])

print(f'\nBest tree rating: {final_rating}')
