import pyreadstat
import os
from constants import TARGET, TOP_FEATURES

xpt_path = "LLCP2015.XPT"  
csv_path = "cleaned_data.csv"

print(f"Loading file: {xpt_path}")
df, meta = pyreadstat.read_xport(xpt_path)
print("File loaded successfully")

print("Unique values in DIABETE3 before cleaning:")
print(df['DIABETE3'].unique())

for col in df.columns:
    if col == 'DIABETE3':
        df[col].fillna(df[col].mode()[0], inplace=True)  
    elif df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("Unique values in DIABETE3 after cleaning:")
print(df['DIABETE3'].unique())

print("Null values before cleaning:")
print(df.isnull().sum())

df = df.loc[:, df.nunique() > 1]
df.drop_duplicates(inplace=True)

df.to_csv(csv_path, index=False)
if os.path.exists(csv_path):
    print(f"Data cleaned and saved as '{csv_path}'")
else:
    print(f"Failed to save '{csv_path}'. Check write permissions.")

keep_cols = TOP_FEATURES + [TARGET]
df = df[keep_cols]
df.to_csv(csv_path, index=False)