class graph(  ):
    
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
    


    ###-