import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('heart.csv')  # Replace with your actual file path

# Display the value counts of the target column
print("Class distribution:")
print(df['target'].value_counts())

# Optional: plot the class distribution
df['target'].value_counts().plot(kind='bar', title='Target Class Distribution', xlabel='Target', ylabel='Count')
plt.show()
