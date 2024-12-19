# !pip install -q pandasai
import pandas as pd 
import numpy as np 
from pandasai import PandasAI 
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../../.env", override=True)

# Add data to an empty DataFrame
data_dict = { 
	"country": [ 
		"Delhi", 
		"Mumbai", 
		"Kolkata", 
		"Chennai", 
		"Jaipur", 
		"Lucknow", 
		"Pune", 
		"Bengaluru", 
		"Amritsar", 
		"Agra", 
		"Kola", 
	], 
	"annual tax collected": [ 
		19294482072, 
		28916155672, 
		24112550372, 
		34358173362, 
		17454337886, 
		11812051350, 
		16074023894, 
		14909678554, 
		43807565410, 
		146318441864, 
		np.nan, 
	], 
	"happiness_index": [9.94, 7.16, 6.35, 8.07, 6.98, 6.1, 4.23, 8.22, 6.87, 3.36, np.nan], 
} 

df = pd.DataFrame(data_dict) 
print(df.head())

print(df.tail())

# Initialize an instance of pandasai
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY")) 
pandas_ai = PandasAI(llm, conversational=False)

## Prompt 1: Finding index of a value
# finding index of a row using value of a column 
response = pandas_ai(df, "What is the index of Pune?") 
print(response)

# Prompt 2: Using Head() function of DataFrame
response = pandas_ai(df, "Show the first 5 rows of data in tabular form") 
print(response)


# Prompt 3: Using Tail() function of DataFrame
response = pandas_ai(df, "Show the last 5 rows of data in tabular form") 
print(response)

# Prompt 4: Using describe() function of DataFrame
response = pandas_ai(df, "Show the description of data in tabular form") 
print(response)

# Prompt 5: Using the info() function of DataFrame
response = pandas_ai(df, "Show the info of data in tabular form") 
print(response)

# Prompt 6: Using shape attribute of dataframe
response = pandas_ai(df, "What is the shape of data?") 
print(response)

# Prompt 7: Finding any duplicate rows
response = pandas_ai(df, "Are there any duplicate rows?") 
print(response)


# Prompt 8: Finding missing values
response = pandas_ai(df, "Are there any missing values?") 
print(response)


# Prompt 9: Drop rows with missing values
response = pandas_ai(df, "Drop the row with missing values with inplace=True and return True when done else False ") 
print(response)

# Checking if the last has been removed row
print(df.tail())

# Prompt 10: Print all column names
response = pandas_ai(df, "List all the column names") 
print(response)


# Prompt 11: Rename a column
response = pandas_ai(df, "Rename column 'country' as 'Country' keep inplace=True and list all column names") 
print(response)


# Prompt 12: Add a row at the end of the dataframe
response = pandas_ai(df, "Add the list: ['A',None,None] at the end of the dataframe as last row keep inplace=True") 
print(response)

# Prompt 13: Replace the missing values
response = pandas_ai(df, """Fill the NULL values in dataframe with 0 keep inplace=True 
and the print the last row of dataframe""") 
print(response)

# Prompt 14: Calculating mean of a column
response = pandas_ai(df, "What is the mean of annual tax collected") 
print(response)


# Prompt 15: Finding frequency of unique values of a column
response = pandas_ai(df, "What are the value counts for the column 'Country'") 
print(response)


# Prompt 16: Dataframe Slicing
response = pandas_ai(df, "Show first 3 rows of columns 'Country' and 'happiness index'") 
print(response)


# Prompt 17: Using pandas where function
response = pandas_ai(df, "Show the data in the row where 'Country'='Mumbai'") 
print(response)


# Prompt 18: Using pandas where function with a range of values
response = pandas_ai(df, "Show the rows where 'happiness index' is between 3 and 6") 
print(response)

# Prompt 19: Finding 25th percentile of a column of continuous values
response = pandas_ai(df, "What is the 25th percentile value of 'happiness index'") 
print(response)

# Prompt 20: Finding IQR of a column
response = pandas_ai(df, "What is the IQR value of 'happiness index'") 
print(response)


# Prompt 21: Plotting a box plot for a continuous column
response = pandas_ai(df, "Plot a box plot for the column 'happiness index'") 
print(response)


# Prompt 22: Find outliers in a column
response = pandas_ai(df, "Show the data of the outlier value in the columns 'happiness index'") 
print(response)


# Prompt 23: Plot a scatter plot between 2 columns
response = pandas_ai(df, "Plot a scatter plot for the columns'annual tax collected' and 'happiness index'") 
print(response)


# Prompt 24: Describing a column/series
response = pandas_ai(df, "Describe the column 'annual tax collected'") 
print(response)


# Prompt 25: Plot a bar plot between 2 columns
response = pandas_ai(df, "Plot a bar plot for the columns'annual tax collected' and 'Country'") 
print(response)


# Prompt 26: Saving DataFrame as a CSV file and JSON file
# to save the dataframe as a CSV file 
response = pandas_ai(df, "Save the dataframe to 'temp.csv'") 
# to save the dataframe as a JSON file 
response = pandas_ai(df, "Save the dataframe to 'temp.json'")

