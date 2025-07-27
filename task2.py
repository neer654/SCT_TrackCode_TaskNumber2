import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the datasets
train = pd.read_csv("C:/Users/Neer Tamboli/Desktop/skillcraft/task 2/train.csv")
test = pd.read_csv("C:/Users/Neer Tamboli/Desktop/skillcraft/task 2/test.csv")

# Target column detection
if 'target' in train.columns:
    target_col = 'target'
elif 'LeagueIndex' in train.columns:
    target_col = 'LeagueIndex'
else:
    target_col = None

# 1. Basic Info
print("🚀 Dataset Info:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain info:\n")
print(train.info())

# 2. Missing Values
print("\n🔍 Missing Values:")
missing = train.isnull().sum()
print(missing[missing > 0])

# Visualize missing values
msno.matrix(train)
plt.title("Missing Data Visualization")
plt.show()

# Handle missing values (basic approach — update based on context)
train.fillna(train.median(numeric_only=True), inplace=True)
train.fillna(method='ffill', inplace=True)

# 3. Summary Stats
print("\n📊 Summary Statistics:")
print(train.describe(include='all'))

print("\n🧾 Column Overview:")
print(train.dtypes.value_counts())
print("\nUnique values in categorical columns:")
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"{col}: {train[col].nunique()} unique values")

# 4. Univariate Analysis - Numeric Columns
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
print(f"\n🔢 Numeric columns: {list(numeric_cols)}")

for col in numeric_cols:
    plt.figure()
    sns.histplot(train[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# 5. Univariate Analysis - Categorical Columns (limit high-cardinality)
categorical_cols = train.select_dtypes(include='object').columns
cat_plot_limit = 30  # max unique categories to plot

for col in categorical_cols:
    if train[col].nunique() <= cat_plot_limit:
        plt.figure()
        sns.countplot(data=train, x=col, order=train[col].dropna().value_counts().index)
        plt.title(f'Countplot of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 6. Correlation Heatmap
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# 7. Bivariate Analysis with Target (if detected)
if target_col and target_col in train.columns:
    print(f"\n🎯 Target column: {target_col}")

    for col in numeric_cols:
        if col != target_col:
            plt.figure()
            sns.scatterplot(data=train, x=col, y=target_col)
            plt.title(f'{col} vs {target_col}')
            plt.tight_layout()
            plt.show()

    for col in categorical_cols:
        if train[col].nunique() <= cat_plot_limit:
            plt.figure()
            sns.boxplot(data=train, x=col, y=target_col)
            plt.title(f'{target_col} by {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
else:
    print("\n⚠️ Target column not found — skipping bivariate analysis.")

# 8. Compare Train/Test Columns
print("\n🧪 Train/Test Column Differences:")
print("Columns in train not in test:", set(train.columns) - set(test.columns))
print("Columns in test not in train:", set(test.columns) - set(train.columns))

print("\n✅ EDA Completed.")