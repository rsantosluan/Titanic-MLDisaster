### Imports
import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Machine Learning
from sklearn.linear_model    import RidgeClassifierCV, LogisticRegressionCV, LassoCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_validate
import xgboost as xgb 



class graph():
    
    ###Values ​​in the bar graph
    def plot_values_vbar( ax ):
        for i in ax.patches:
            if i == 0:
                None
            else:
                ax.annotate(
                #Texto a ser plotado
                round(i.get_height()),
                #Posição horizontal
                (i.get_x() + i.get_width() /2, i.get_height() + i.get_height() /80 ),
                ha='center' ,   
                color='black',
                fontsize=15
        );  
        return None


class encoding():    

    ### OneHotEncoding
    def data_hotencoding(df, column, max_cat = np.nan):
        if max_cat != np.nan:
            ohe       = OneHotEncoder( max_categories = max_cat )
        else:
            ohe       = OneHotEncoder()

        g         = ohe.fit_transform( df[column].values.reshape(-1,1) ).toarray()
        df_ohe    = pd.DataFrame( g, columns=[ str( column ) + '_' + str( int( i ) ) for i in range( g.shape[1] ) ] )
        df = pd.concat( [df, df_ohe], axis= 1 )
        df = df.drop(columns=[column])
        print(g)
        return df


    ###-

class modelSelect():
    def scorecv(xtr,ytr,xte,yte):
        #Ramdom Forest
        rfc = RandomForestClassifier()
        rfc.fit( xtr, ytr )
        #Score
        res    = cross_validate( rfc, xte,yte, cv = 5 )
        resrf = pd.DataFrame(
            {'Model Name' : 'Random Forest Classifier',
            'CV Score'    : res['test_score'].mean()},
            index = [0]
        )

        #XGBoost
        xgbc = xgb.XGBClassifier()
        xgbc.fit( xtr, ytr )
        res = cross_validate( xgbc, xte,yte, cv = 5 )
        resxgb= pd.DataFrame(
            {'Model Name' : 'XGBoost Classifier',
            'CV Score'    : res['test_score'].mean()},
            index = [0]
        )

        #Ridge Classifier
        rc    = RidgeClassifierCV()
        rc.fit( xtr, ytr )
        res   = rc.score( xte, yte )
        resrc = pd.DataFrame(
            {'Model Name' : 'Ridge Classifier',
            'CV Score'    : res},
            index = [0]
        )

        #Logistic Regression
        lr    = LogisticRegressionCV()
        lr.fit( xtr, ytr )
        res   = lr.score( xte, yte )
        reslr = pd.DataFrame(
            {'Model Name': 'Logistic Regression',
            'CV Score'   : res},
            index = [0]
        )

        #Lasso
        lss = LassoCV()
        lss.fit( xtr, ytr )
        res = lss.score( xte, yte )
        reslss = pd.DataFrame(
            {'Model Name' : 'Lasso',
            'CV Score'    : res},
            index = [0]
        )
        
        #Result
        result = pd.concat( [resrf,resrc, reslr, reslss, resxgb] )
        result.sort_values( 'CV Score', ascending = False, inplace = True )
        result.reset_index( drop = True, inplace = True )


        return result