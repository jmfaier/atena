#Bibliotecas para importacao###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
###############################################################################

#Bibliotecas###################################################################
import numpy as np
import re
###############################################################################

#Cria a Big Table##############################################################
def big_table(categorias,clickview,ID,param):
    #Big Table Estruturada e Granular##########################################
    btsg = np.zeros((np.shape(clickview)[0],np.shape(categorias)[0]),dtype="float32")
    k=0
    for i in clickview[:]:
        l = 0
        for j in categorias[1:,1]:
            m = re.compile(j).search(i)
            if m:
                btsg[k,l] = 1
            else: 
                btsg[k,l] = 0
            l = l + 1
        k = k+1
    ###########################################################################
    
    #Big Table Estruturada#####################################################
    u, idx = np.unique(ID, return_inverse = True)
    #Agrupamento
    bts = np.zeros((np.shape(u)[0],np.shape(categorias)[0]), dtype="float32")
    for i in range(0, np.shape(categorias)[0]):
        categorias = btsg[:,i].astype(float)
        bts[:,i] = np.bincount(idx, weights=categorias)
    ###########################################################################
          
    #Parametros################################################################
    alvo_pos = param[0]
    alvo_neg = param[1] 
    explicativas = param[2]
    ###########################################################################
    
    #Alvo e explicatias########################################################
    alvo_pos_idx= np.nonzero(bts[:,alvo_pos]>0)
    bts[alvo_pos_idx,alvo_pos]=1
    bts[:,alvo_neg]=1
    bts[alvo_pos_idx,alvo_neg]=0
    ###########################################################################
    
    #Treino e validacao########################################################
    num_treino = 20000
    bts_treino = bts[0:num_treino-1,:]
    bts_validacao= bts[num_treino:,:]
    ###########################################################################
    
    #Salvamento################################################################
    #np.savetxt("codigo/pupx/dados/bts.csv", bts[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    #np.savetxt("codigo/pupx/dados/bts_treino.csv", bts_treino[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    #np.savetxt("codigo/pupx/dados/bts_validacao.csv", bts_validacao[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    ###########################################################################
    
    #Retorno###################################################################
    return bts, bts_treino, bts_validacao
    ###########################################################################
###############################################################################