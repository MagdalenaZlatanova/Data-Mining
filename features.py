import pandas as pd
import sklearn


path="C:\\Users\\PCI\\Desktop\\mining proekt\\test\\FeatureVectorWithLabel.csv"
pd.set_option('display.max_columns', 200)

df=pd.read_csv(path)
#print(df.head())
print(df.columns)