#Bibliotecas###################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
###############################################################################

#Funcoes Python################################################################
os.chdir("C:/Users/JF186037/Desktop/JMF/GIT/projects/atena")
from codigo.pp import big_table as bt
###############################################################################

def main():
    #Parametros################################################################
    target = 6
    features = (0,1,2,3,4,5) #Tuple
    neuronios_escondidos = [10,20,10] #lista
    n_classes = 2
    steps = 2000
    model_dir="modelo/atena_model"
    ###########################################################################
    
    #Input#####################################################################
    categorias = np.loadtxt('dados/categorias.csv',dtype=bytes, delimiter=',').astype(str)
    #webdata = np.loadtxt('C:/Users/JF186037/Desktop/JMF/Python/codigo/pupx/dados/webdados_OK.txt',dtype=bytes, delimiter='\t').astype(str)
    webdata = np.loadtxt('dados/webdados.txt',dtype=bytes, delimiter='\t').astype(str)
    clickview = webdata[1:,2]
    wd = webdata[1:,(0,1,3)]
    ###########################################################################
      
    #Big Table#################################################################
    alvo_pos = [6]
    alvo_neg = [7] 
    explicativas = [0,1,2,4,5]
    param = alvo_pos, alvo_neg, explicativas
    ID = wd[:,2]
    bts, bts_treino, bts_validacao = bt.big_table(categorias,clickview,ID,param)
    ###########################################################################
    
    # Especifica que todas as features tem dados de valor real#################
    feature_columns = [tf.feature_column.numeric_column("x", shape=[np.size(features)])]
    ###########################################################################
      
    # DNN de 3 camadas com 10, 20, 10 unidades, respectivamente################
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=neuronios_escondidos,
                                            n_classes=n_classes,
                                            model_dir=model_dir)
    #############################################################################
      
    
    # Inputs de treino###########################################################
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(bts_treino[:,features])},
        y=np.array(bts_treino[:,target],dtype="int"),
        num_epochs=None,
        shuffle=True)
    #############################################################################
    
    # Treina o modelo############################################################
    classifier.train(input_fn=train_input_fn, steps=steps)
    #############################################################################
    
    # Inputs de teste############################################################
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(bts_validacao[:,features])},
        y=np.array(bts_validacao[:,target],dtype="int"),
        num_epochs=1,
        shuffle=False)
    #############################################################################
    
    # Acurácia###################################################################
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    #############################################################################
    
    #############################################################################
    # Classifica as novas amostras###############################################
    #############################################################################
      
    #Novas amostras##############################################################
    new_samples = np.array(
        [[1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0.]], dtype=np.float32)
    #############################################################################
      
    #Input da nova amostra#######################################################
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)
    #############################################################################
      
    #Predicao####################################################################
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    #############################################################################
      
    #Classificacao###############################################################
    predicted_classes = [p["classes"] for p in predictions]
    print(format(predicted_classes))
    #############################################################################
    
    #Armazenamento#############################################################
    #Input
    np.savetxt("codigo/modelo/input/bts.csv", bts[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    np.savetxt("codigo/modelo/input/bts_treino.csv", bts_treino[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    np.savetxt("codigo/modelo/input/bts_validacao.csv", bts_validacao[:,(0,1,2,3,4,5,6)], delimiter=",",  newline='\n',fmt='%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1.1f,%1d')
    #Output
    pc = np.asarray([pc["classes"] for pc in predictions])
    pc = np.reshape(pc,(3,1))
    pc = pc.astype("float")
    np.savetxt("codigo/modelo/output/prediction_classes.csv", pc, delimiter=",", newline='\n',fmt='%1d')
    pr = np.asarray([pr["probabilities"] for pr in predictions])
    pr = np.reshape(pr,(3,2))
    pr = pr.astype("float")
    np.savetxt("codigo/modelo/output/prediction_prob.csv", pc, delimiter=",", newline='\n',fmt='%1.1f')    
    ###########################################################################
# Main#########################################################################
if __name__ == "__main__":
    main()
###############################################################################
    

import os
import tensorflow as tf
import numpy as np

training_dir = 'dados/'
training_filename = 'categorias.csv'

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.str,
        features_dtype=np.str),
        target_column=-1)
