import pandas as pd
import numpy as np

# task-0
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
subjects = ['Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math']
scores = np.random.randint(50, 101, size=10)

df = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': [''] * 10
})

# task-1
def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(assign_grade)

# task-2
print(df.sort_values(by='Score', ascending=False))

# task-3
print(df.groupby('Subject')['Score'].mean())

# task-4
def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

filtered_df = pandas_filter_pass(df)
print(filtered_df)
