### Imports
import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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