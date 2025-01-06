import sys

import pandas as pd 
from sqlalchemy import create_engine

# File names
join_id = 'id'

# DB name
clean_db = 'cleaned_data.db'

# Model name
model = 'final_model.pkl' 

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge by id
    df = messages.merge(categories, on = join_id)
    return df

def clean_data(df):
    # Columns
    colnames = [col.split('-')[0] for col in df.categories.str.split(";")[0]]
    df[colnames] = df.categories.str.split(";", expand= True).values
    
    for col in colnames:
        df[col] = df[col].str.partition('-')[2].apply(lambda x: int(x))
    
    # Duplicates
    print('MSG. There are {} duplicated records.'.format(df.shape[0]-df.drop_duplicates().shape[0]))
    df = df.drop_duplicates()
    return df

def save_data(df, database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    try:
        df.to_sql('data_msg', engine, index=False)
    except:
        print('MSG. Failed to create table.' )

def main():
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
    