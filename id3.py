from numpy import log2
import pandas as pd
from copy import deepcopy


class DecisionTree:
    def __init__(self, df: pd.DataFrame, columns_names: list, max_depth: int, number_of_childs: int) -> None:
        self.df = df
        self.columns_names = columns_names
        self.number_of_childs = number_of_childs
        self.max_depth = max_depth-1
        
        self.best_column_name = get_best_column(self.df, self.columns_names)
        
        self.result = None
        
        if self.best_column_name is None:
            self.result = self.df['target'].unique()
        else:
            if max_depth != 0:
                ranges = self.create_values_ranges()
                self.children = {} # value: tree
                self.create_children(ranges)
            else:
                self.result = self.df['target'].unique()
        
    def __str__(self) -> str:
        if self.result is not None:
            return f'Result: {self.result}'
        else:
            return f'Best column: {self.best_column_name}\nChildren: {sorted(tuple(self.children.keys()))}'
    
    def __getitem__(self, key):
        '''
        Return key'th child tree 
        '''
        if self.max_depth == -1:
            raise ValueError('Leaf node has no children')
        else:
            return self.children[list(self.children.keys())[key]]   
    
    def create_children(self, ranges: list) -> None:
        for range in ranges:
            child_df = self.df[self.df[self.best_column_name].between(range[0], range[1], inclusive='both')].copy(deep=True)
            child_df.drop(self.best_column_name, axis=1, inplace=True)
            child_columns = deepcopy(self.columns_names)
            child_columns.remove(self.best_column_name)
            self.children[range] = DecisionTree(child_df, child_columns, self.max_depth-1)
    
    def create_values_ranges(self) -> list:
        '''
        Returns list of tuples with ranges of values for each child
        '''
        childs_values = tuple(self.df[self.best_column_name].unique())
        min_val = float(min(childs_values))
        max_val = float(max(childs_values))
        
        step = round((max_val - min_val) / self.number_of_childs, 2)
        
        ranges = [(float('-inf'), round(min_val+step, 2))]
        
        for i in range(1, self.number_of_childs-1):
            ranges.append((round(min_val+i*step+0.01, 2), round(min_val+(i+1)*step, 2)))
            
        ranges.append((round(max_val-step+0.01, 2), float('inf')))  
        
        return ranges

    def predict(self, row: pd.Series) -> str:
        '''
        Return predicted value for row
        '''
        if self.result is not None:
            return self.result
        else:
            value = row[self.best_column_name]
            for range in self.children:
                if value >= range[0] and value <= range[1]:
                    return self.children[range].predict(row)
            raise ValueError(f'Value {value} is not in ranges of values of {self.best_column_name}') 


def get_best_column(df: pd.DataFrame, columns: list) -> str:
    best_gain = 0
    best_column = None
    
    for column in columns:
        gain = calc_gain(df, column)
        if gain > best_gain:
            best_gain = gain
            best_column = column
    
    if best_column == None:
        return None
    else:
        return best_column


def calc_entropy(df: pd.DataFrame, column_name: str = None, value: str = None) -> float:
    target_values = df['target'].unique()
    
    if column_name == None:
        rows_count = len(df)
        target_counts = [len(df[df['target'] == target_value]) for target_value in target_values]
    else:
        rows_count = len(df[df[column_name] == value])
        target_counts = [len(df[(df[column_name] == value) & (df['target'] == target_value)]) for target_value in target_values]

    if 0 in target_counts:
        return 0
    
    entropy = 0
    for count in target_counts:
        entropy -= count/rows_count*log2(count/rows_count)
    return entropy


def calc_gain(df: pd.DataFrame, column_name: str) -> float:
    available_values = df[column_name].unique()
    
    entropy_all = calc_entropy(df)
    
    rows_count = len(df)
    
    for value in available_values:
        rows_count_value = len(df[df[column_name] == value])
        entropy_all -= rows_count_value/rows_count*calc_entropy(df, column_name, value)
    
    return entropy_all

'''
def get_split_info(df: pd.DataFrame, column_name: str) -> float:
    available_values = df[column_name].unique()
    df_len = len(df)
    
    split_info = 0
    for value in available_values:
        division = len(df[df[column_name] == value])/df_len
        split_info -= division*log2(division)
    
    return split_info


def get_gain_ration(df: pd.DataFrame, column_name: str) -> float:
    return calc_gain(df, column_name)/get_split_info(df, column_name)
'''

if __name__ == '__main__':

    df = pd.read_csv('simple_data.csv')

    tree = DecisionTree(df, ['outlook', 'temp', 'humidity', 'wind'])

    print(tree)

    print(tree.children['Rain'])
