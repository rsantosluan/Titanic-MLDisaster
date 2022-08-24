### Imports
from operator import index
import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle as pkl


#Machine Learning
from sklearn.linear_model    import RidgeClassifierCV, LogisticRegressionCV, LassoCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree            import DecisionTreeClassifier
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
    
    def full_pipeline(data_raw):
        ###Prepare Data
        #Sex
        data_raw['Sex']      = data_raw.Sex.apply( lambda x : 0 if x == 'male' else 1 )

        #Age
        data_raw.Age = data_raw.Age.apply( lambda x : 1 if x <= 1 else x  )
        media_m = int( data_raw[data_raw.Sex == 0].Age.mean() )
        media_f = int( data_raw[data_raw.Sex == 1].Age.mean() )
        data_raw['Age'] = data_raw[['Age', 'Sex']].apply(
            lambda x :  media_m if ( np.isnan( x['Age'] ) ) & ( x['Sex'] == 0 ) else 
            media_f if ( np.isnan( x['Age'] ) ) & ( x['Sex'] == 1 ) 
            else x['Age'], axis = 1
        )
        data_raw.Age = data_raw.Age.apply( lambda x : int(x) )
        #Cabin
        data_raw.drop( columns = 'Cabin', inplace = True )
        #Embarked
        data_raw.Embarked.fillna( 'S', inplace = True )
        #Personal Title
        data_raw['P_title'] = data_raw.Name.apply( lambda x : x[x.find( ',' ) + 2 : x.find( '.' )]  )



        ### Encoding
        data_raw = pd.get_dummies( data_raw, columns = ['P_title'], prefix = '_' )

        ### Rescaling
        mms_age    = pkl.load( open( '../models/mms_age.sav', 'rb' ) )
        mms_sibsp  = pkl.load( open( '../models/mms_sibsp.sav', 'rb' ) )
        mms_parch  = pkl.load( open( '../models/mms_parch.sav', 'rb' ) )
        mms_fare   = pkl.load( open( '../models/mms_fare.sav', 'rb' ) )
        mms_pclass = pkl.load( open( '../models/mms_pclass.sav', 'rb' ) )

        df = data_raw[['PassengerId', 'Sex', '__Mr']]        
        df['Age']   = mms_age.transform( data_raw[['Age']].values )
        df['SibSp']   = mms_sibsp.transform( data_raw[['SibSp']].values )
        df['Parch']   = mms_parch.transform( data_raw[['Parch']].values )
        df['Fare']   = mms_fare.transform( data_raw[['Fare']].values )
        df['Pclass']   = mms_pclass.transform( data_raw[['Pclass']].values ) 

        ### Feature Selection
        df = df[['PassengerId','Fare', 'Age', 'Sex', 'Pclass', '__Mr', 'SibSp']]
        
        return df   


    ###-

class modelSelect():
    def scorecv(xtr,ytr,xte,yte):
        
        #XGBoost
        xgbc = xgb.XGBClassifier( random_state = 42 )
        xgbc.fit( xtr, ytr )
        res = cross_validate( xgbc, xte,yte, cv = 5, scoring = 'f1_macro' )
        resxgb= pd.DataFrame(
            {'Model Name' : 'XGBoost Classifier',
            'CV Score F1'    : res['test_score'].mean()},
            index = [0]
        )


        #Ramdom Forest
        rfc = RandomForestClassifier( random_state = 42 )
        rfc.fit( xtr, ytr )
        #Score
        res    = cross_validate( rfc, xte,yte, cv = 5, scoring = 'f1_macro' )
        resrfc = pd.DataFrame(
            {'Model Name' : 'Random Forest Classifier',
            'CV Score F1'    : res['test_score'].mean()},
            index = [0]
        )


        #Decision Tree
        dtc = DecisionTreeClassifier( random_state = 42 )
        dtc.fit( xtr, ytr )
        #Score
        res    = cross_validate( dtc, xte,yte, cv = 5, scoring = 'f1_macro' )
        resdtc = pd.DataFrame(
            {'Model Name': 'Decision Tree',
            'CV Score F1'   : res['test_score'].mean()},
            index = [0]
        )
        
        
        #Ridge Classifier
        rc    = RidgeClassifierCV( scoring = 'f1_macro')
        rc.fit( xtr, ytr )
        res   = rc.score( xte, yte )
        resrc = pd.DataFrame(
            {'Model Name' : 'Ridge Classifier',
            'CV Score F1'    : res},
            index = [0]
        )


        #Logistic Regression
        lr    = LogisticRegressionCV( scoring = 'f1_macro', random_state = 42 )
        lr.fit( xtr, ytr )
        res   = lr.score( xte, yte )
        reslr = pd.DataFrame(
            {'Model Name': 'Logistic Regression',
            'CV Score F1'   : res},
            index = [0]
        )
        

        #Result
        result = pd.concat( [resxgb, resrfc, resdtc, resrc, reslr ] )
        result.sort_values( 'CV Score F1', ascending = False, inplace = True )
        result.reset_index( drop = True, inplace = True )


        return result