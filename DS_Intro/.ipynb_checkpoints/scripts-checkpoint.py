import numpy as np
import pandas as pd

def split_clean_data(dg, 
                     subset, 
                     cast_as_dt, part=['$','%',','], 
                     tol = 0.80, 
                     split= False, 
                     exc_list = ['host_verifications', 'host_response_time','property_type','cancellation_policy']):
    """
    INPUT: 
    df: DataFrame
    subset: subset of variables considered interesting 
    cast_as_dt: list of date fields
    part: list of problematic particles
    tol: tolerance for null values
    split: boolean to return numerical and categorical variables separately
    exc_list: this is a list of variables that are objects in type, but will not be treated as categorical variables
    
    OUTPUT:
    df_num: cleaned dataframe with numerical variables
    df_cat: cleaned dataframe with categorical variables   
    or
    df: full dataframe with cleaned data
    """
    
    # Droping columns with too many missing values
    df = dg[subset].copy()
    ser_perc = df.describe().transpose()['count']/df.describe().transpose()['count'].max()
    df = df.drop(columns = ser_perc[ser_perc <= tol].index)
    
    # Casting dates
    for col in cast_as_dt:
        try:
            df[col] = pd.DatetimeIndex(df[col])
        except:
            pass
    
    # Variables containing undesirable particles
    list_p_cols = df.select_dtypes(include = ['object']).apply(lambda col: col.astype(str).str.contains(part[0])).columns

    for col in list_p_cols:
        t = df[col].astype(str).str.contains(part[0], regex = False).sum()
        s = df[col].astype(str).str.contains(part[1], regex = False).sum()

        if t!= 0 or s!=0 :
            try:
                df[col] = df[col].astype(str).apply(lambda x: x.replace(part[0],'').replace(part[1],'').replace(part[2],'')).astype(float) 
                print('MSG: The column "{}" was amended'.format(col))
            except:
                pass
    
    # cleaning variables 
    num_vars = df.select_dtypes(include = ['int','float']).columns
    for col in num_vars:
        df[col].fillna(df[col].mean(), inplace = True)
        
    cat_vars = df.select_dtypes(include = ['object']).columns
    for var in cat_vars:
        if var not in exc_list:
            df = pd.concat([df.drop(var, axis= 1), 
                            pd.get_dummies(df[var], 
                                           prefix= var,
                                           prefix_sep = '_',
                                           drop_first = True)],
                           axis = 1)     
    return (df[num_vars], df.drop(columns= num_vars)) if split else df

def query_positive(dg, col_int = 'comments', part = ['great','good','gran','cute', 'spacious','clean','comfortable']):
    """
    INPUT: s
    dg: dataframe with comments
    part: list of strings containing particles to classify news
    col_int: the label of the comment column
    OUTPUT:
    df: queried dataframe. That is, a dataframe that tells how many good reviews a listing has had in a given date<
    """
    df = dg.dropna().copy()
    idx_set = set()
    for word in part:
        dg = df[df[col_int].str.contains(word, regex = False)]
        idx = set(list(dg.index))
        idx_set.update(idx)
    
    # Getting a sentiment score
    df.loc[list(idx_set),'sentiment_new'] = 1
    df['sentiment_new']= df['sentiment_new'].fillna(-1) 
    df = df.drop(columns = [col_int])
    
    idx_cols = ['listing_id','date']    
    dg = df.sort_values(idx_cols, ascending= True).set_index(idx_cols).groupby(level=0).cumsum().reset_index()
    
    return dg