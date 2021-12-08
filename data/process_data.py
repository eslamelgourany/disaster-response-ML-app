# Import Libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Inputs:
        messages_filepath: Path for the file messages
        categories_filepath: Path for the file categories
    Returns:
        Merged DataFrame of both csv files.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, on='id')
    
    return df

def clean_data(df):
    """
    Function to clean the Merged DataFrame.
    Input: Messy DataFrame.
    Output: Cleaned DataFrame.
    """
    # Create Catrgories DataFrame which columns of the cateogries split on the char ;
    categories = df['categories'].str.split(pat=';', expand=True)
    first_row = categories.iloc[0]
    
    # Get the names and rename the columns.
    category_colnames = first_row.apply(lambda x: x[0:-2])
    categories = categories.rename(columns=category_colnames)

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop cateogries column.
    df.drop(['categories'], axis=1, inplace=True)
    
    # Concatenate the cateogires withe the original DF.
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates inplace.
    df.drop_duplicates(inplace=True)
    
    # Replace the values which has 2 with 1.
    df['related'].replace(2, 1, inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """
    Function to save the cleaned data.
    Inputs:
        df: DataFrame required to be saved.
        database_filename: Name of the database.
    Returns: None.
    """
    
    
    # Create the DB
    engine = create_engine('sqlite:///' + str(database_filename))
    
    # Create the table
    df.to_sql(database_filename, engine, index=False)


def main():
    """
    Main Function that calls all the above functions.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()