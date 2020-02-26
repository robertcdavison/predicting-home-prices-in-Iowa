import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_model(X, y, z, model):
    
    # Split Data into Train and test trial
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
    
    # ridge and lasso models require the data to be scaled first
    ss = StandardScaler()
    X_train_sc = ss.fit_transform(X_train)
    X_test_sc = ss.transform(X_test)
    z_sc = ss.transform(z.drop(columns='id'))
    
    # define two variables for r2 adjusted
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    p_train = X_train.shape[1]
    p_test = X_test.shape[1]
    
    
    if model == 'lr' or model == 'loglr':

        # instantiate and fit a lr model for the train set
        modeler = LinearRegression()
        modeler.fit(X_train, y_train)

        # make predictions
        y_pred_train = modeler.predict(X_train)
        y_pred_test = modeler.predict(X_test)

        # calculate TRAIN metrics
        r2_train = modeler.score(X_train, y_train)
        r2_adj_train = 1 - ((1-r2_train)*(N_train-1))/(N_train-p_train-1)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        if model == 'lr':
            mse_train = mean_squared_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
        elif model == 'loglr':
            mse_train = mean_squared_error(np.exp(y_train), np.exp(y_pred_train))
            rmse_train = np.sqrt(mse_train)

        # calculate TEST metrics
        r2_test = modeler.score(X_test, y_test)
        r2_adj_test = 1 - ((1-r2_test)*(N_test-1))/(N_test-p_test-1)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        if model == 'lr':
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
        elif model == 'loglr':
            mse_test = mean_squared_error(np.exp(y_test), np.exp(y_pred_test))
            rmse_test = np.sqrt(mse_test)        

        # Cross Validation
        cross_list = cross_val_score(modeler, X_train, y_train, cv=5)
        cross_mean = cross_val_score(modeler, X_train, y_train, cv=5).mean()

        # create the lists that will make up the DF
        metric_names  = ['R2', 'R2_adj', 'MAE', 'MSE', 'RMSE']
        train_metrics = [r2_train, r2_adj_train, mae_train, mse_train, rmse_train]
        test_metrics = [r2_test, r2_adj_test, mae_test, mse_test, rmse_test]

        print(f"Cross Val Scores: {cross_list}")
        print(f"  Cross Val Mean: {cross_mean}")

        # put all the values in a Dataframe
        pred_df = pd.DataFrame({'Key Metrics': metric_names,
                                'Train': train_metrics,
                                'Test': test_metrics})

        # this code prevents the df from outputting scientific not
        pred_df['Train'] = pred_df['Train'].apply(lambda x: '%.5f' % x) 
        pred_df['Test'] = pred_df['Test'].apply(lambda x: '%.5f' % x)    

        # now create the kaggle submission df to be output
        # remember z is your kaggle feature df to be tested
        # aka kaggle.loc[:, features]
        # remember that test is the column with corresponding Id's that need to be matched
        # also you have instantiate the Id column
        final_prediction = modeler.predict(z.drop(columns='id'))
        
        # before making predictions, see if you need to exponentiate the answers
        if model == 'lr':
            z['SalePrice'] = final_prediction
            z['Id'] = z['id']
            kaggle_submission = z[['Id', 'SalePrice']]
            return pred_df, y_pred_train, y_train, kaggle_submission
        
        elif model == 'loglr':
            z['SalePrice'] = np.exp(final_prediction)
            z['Id'] = z['id']
            kaggle_submission = z[['Id', 'SalePrice']]
            return pred_df, y_pred_train, y_train, kaggle_submission
    
    
    
    
    
    elif model == 'stats':
        
        # add a constant for the model
        X = sm.add_constant(X)
        
        #instantiate and fit the model
        modeler = sm.OLS(y, X).fit()
        
        # select the significant p values from the stats model output
        summary_df = modeler.summary2().tables[1]
        significant_cols = summary_df.loc[summary_df['P>|t|'] < 0.05, :].index
        significant_col_df = pd.DataFrame(significant_cols, columns=['Features with p < 0.05'])
        
        return modeler.summary(), summary_df, significant_col_df
    
    
    
    
    
    elif model == 'ridge' or model == 'logridge':
        
        # instantiate then fit the ridge model
        modeler = RidgeCV()
        modeler.fit(X_train_sc, y_train)

        # make predictions
        y_pred_train = modeler.predict(X_train_sc)
        y_pred_test = modeler.predict(X_test_sc)

        # calculate TRAIN metrics
        r2_train = modeler.score(X_train_sc, y_train)
        r2_adj_train = 1 - ((1-r2_train)*(N_train-1))/(N_train-p_train-1)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        if model == 'ridge':
            mse_train = mean_squared_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
        elif model == 'logridge':
            mse_train = mean_squared_error(np.exp(y_train), np.exp(y_pred_train))
            rmse_train = np.sqrt(mse_train)

        # calculate TEST metrics
        r2_test = modeler.score(X_test_sc, y_test)
        r2_adj_test = 1 - ((1-r2_test)*(N_test-1))/(N_test-p_test-1)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        if model == 'ridge':
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
        elif model == 'logridge':
            mse_test = mean_squared_error(np.exp(y_test), np.exp(y_pred_test))
            rmse_test = np.sqrt(mse_test)

        # Cross Validation
        cross_list = cross_val_score(modeler, X_train_sc, y_train, cv=5)
        cross_mean = cross_val_score(modeler, X_train_sc, y_train, cv=5).mean()

        # create the lists that will make up the DF
        metric_names  = ['R2', 'R2_adj', 'MAE', 'MSE', 'RMSE']
        train_metrics = [r2_train, r2_adj_train, mae_train, mse_train, rmse_train]
        test_metrics = [r2_test, r2_adj_test, mae_test, mse_test, rmse_test]

        print(f"Cross Val Scores: {cross_list}")
        print(f"  Cross Val Mean: {cross_mean}")


        # put all the values in a Dataframe
        pred_df = pd.DataFrame({'Key Metrics': metric_names,
                                'Train': train_metrics,
                                'Test': test_metrics})

        pred_df['Train'] = pred_df['Train'].apply(lambda x: '%.5f' % x) 
        pred_df['Test'] = pred_df['Test'].apply(lambda x: '%.5f' % x)    

        # add in a dataframe with the features sorted by coefficient value
        key_feature_coefs = pd.DataFrame(zip(list(X_train), modeler.coef_), columns = ['feature', 'coefficient'] ).sort_values(by='coefficient')
        
#         # create the kaggle submission df to be output

        final_prediction = modeler.predict(z_sc)                  
    
    
            # before making predictions, see if you need to exponentiate the answers
        if model == 'ridge':
            z['SalePrice'] = final_prediction
            z['Id'] = z['id']
            kaggle_submission = z[['Id', 'SalePrice']]
            return pred_df, y_pred_train, y_train, kaggle_submission 
        
        elif model == 'logridge':
            z['SalePrice'] = np.exp(final_prediction)
            z['Id'] = z['id']
            kaggle_submission = z[['Id', 'SalePrice']]
            return pred_df, y_pred_train, y_train, kaggle_submission 
        

    
    
    
    
    
    elif model == 'lasso' or model == 'loglasso':
        
        # instantiate then fit the lasso model
        modeler = LassoCV(max_iter = 10_000)
        modeler = modeler.fit(X_train_sc, y_train)

        # make predictions
        y_pred_train = modeler.predict(X_train_sc)
        y_pred_test = modeler.predict(X_test_sc)

        # calculate TRAIN metrics
        r2_train = modeler.score(X_train_sc, y_train)
        r2_adj_train = 1 - ((1-r2_train)*(N_train-1))/(N_train-p_train-1)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        if model == 'lasso':
            mse_train = mean_squared_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mse_train)
        elif model == 'loglasso':
            mse_train = mean_squared_error(np.exp(y_train), np.exp(y_pred_train))
            rmse_train = np.sqrt(mse_train)

        # calculate TEST metrics
        r2_test = modeler.score(X_test_sc, y_test)
        r2_adj_test = 1 - ((1-r2_test)*(N_test-1))/(N_test-p_test-1)
        mae_test = mean_absolute_error(y_test, y_pred_test)     
        if model == 'lasso':
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
        elif model == 'loglasso':
            mse_test = mean_squared_error(np.exp(y_test), np.exp(y_pred_test))
            rmse_test = np.sqrt(mse_test)            
            
        # Cross Validation
        cross_list = cross_val_score(modeler, X_train_sc, y_train, cv=5)
        cross_mean = cross_val_score(modeler, X_train_sc, y_train, cv=5).mean()

        # create the lists that will make up the DF
        metric_names  = ['R2', 'R2_adj', 'MAE', 'MSE', 'RMSE']
        train_metrics = [r2_train, r2_adj_train, mae_train, mse_train, rmse_train]
        test_metrics = [r2_test, r2_adj_test, mae_test, mse_test, rmse_test]

        print(f"Cross Val Scores: {cross_list}")
        print(f"  Cross Val Mean: {cross_mean}")

        # put all the values in a Dataframe
        pred_df = pd.DataFrame({'Key Metrics': metric_names,
                                'Train': train_metrics,
                                'Test': test_metrics})

        pred_df['Train'] = pred_df['Train'].apply(lambda x: '%.5f' % x) 
        pred_df['Test'] = pred_df['Test'].apply(lambda x: '%.5f' % x)    

        # add in a dataframe with the features sorted by coefficient value
        key_feature_coefs = pd.DataFrame(zip(list(X_train), modeler.coef_), columns = ['feature', 'coefficient'] ).sort_values(by='coefficient')
        
        # create the kaggle submission df to be output
        final_prediction = modeler.predict(z_sc)
        z['SalePrice'] = final_prediction
        z['Id'] = z['id']
        kaggle_submission = z[['Id', 'SalePrice']]
        
        return pred_df, y_pred_train, y_train, kaggle_submission, key_feature_coefs
 

    
def submit_a_kaggle(submission, notebook_number):
    submission.to_csv('./submissions/' + str(notebook_number) + '-features-submission.csv', index=False)