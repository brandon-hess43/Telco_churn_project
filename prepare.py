import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def prep_telco(df):
    '''
    accepts a dataframe, drops all my duplicate columns,
    replaces empty spaces with 0.0
    returns a dataframe
    '''
    
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
     
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    
    return df



def encode_telco(df):
    df = df[['customer_id','tenure','tech_support','monthly_charges','contract_type','churn']] # filter only columns we want
    
    df = df.set_index('customer_id') # set index to customer id
    
    df[['tech_supp_no','no_int_service','tech_supp_yes']] = pd.get_dummies(df.tech_support).astype(int) #encode data
    
    df[['annual_contract','biennial_contract']] = pd.get_dummies(df.contract_type, drop_first=True).astype(int) #encode data
    
    df = df.assign(churned = np.where(df['churn'] == 'Yes', 1, 0)) #encode data
    
    df = df.drop(columns=['tech_support','contract_type','churn','no_int_service']) # drop redundant columns
    
    return df
    


def splitting_data(df, col):
    '''
    send in target df to split into train validate test
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=123,
                     stratify=df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col]
                        
                                     )
    return train, validate, test