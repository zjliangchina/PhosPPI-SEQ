from keras import models
from tensorflow.keras import layers
from keras import Input
from keras import optimizers, metrics
import numpy as np
import pandas as pd
from keras.layers import Dropout
import os
import tensorflow as tf
from transformers import AutoTokenizer,  TFRobertaModel
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D

    
def build_model_test(dim1,dim2,roberta_layer):
    input_x = Input(shape=(dim1, ), name='motif',dtype='int32') 
    input_x_mask =  Input(shape=(dim1, ), name='motif_mask',dtype='int32') 
    x = roberta_layer(input_ids=input_x,attention_mask=input_x_mask).last_hidden_state
    x = layers.Conv1D(filters=192, kernel_size=6, strides=1, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dropout(0.1)(x) 

    input_y = Input(shape=(dim2, ), name='seq',dtype='int32') 
    input_y_mask =  Input(shape=(dim2, ), name='seq_mask',dtype='int32') 
    y = roberta_layer(input_ids=input_y,attention_mask=input_y_mask).last_hidden_state
    y = layers.Conv1D(filters=192, kernel_size=6, strides=1, padding='same')(y)
    y = layers.ReLU()(y)  
    y = layers.BatchNormalization()(y)  
    y = layers.Dropout(0.1)(y)  

    mha = MultiHeadAttention(num_heads=6, key_dim=32)
    cross_attended_1 = mha(x, y, y) 
    cross_attended_2 = mha(y, x, x) 
    cross_attended_1 = layers.Dropout(0.2)(cross_attended_1)  
    cross_attended_2 = layers.Dropout(0.2)(cross_attended_2)  
    out_1 = layers.Add()([x,cross_attended_1]) 
    out_2 = layers.Add()([y,cross_attended_2]) 
    out_1_norm = layers.LayerNormalization()(out_1)
    out_2_norm = layers.LayerNormalization()(out_2)
    pooling_layer_1 = GlobalAveragePooling1D()(out_1_norm)
    pooling_layer_2 = GlobalAveragePooling1D()(out_2_norm)

    out = layers.Concatenate(axis=-1)([pooling_layer_1, pooling_layer_2]) 
    out = layers.Dense(64,activation='relu')(out)
    out = Dropout(0.3)(out)
    out = layers.Dense(1,activation='sigmoid')(out)

    model = models.Model([input_x,input_x_mask,input_y,input_y_mask], out)
    return model

def train_model(train_motif, train_motif_mask, train_seq, train_seq_mask, train_label, model_name, model_path):
    tf.debugging.set_log_device_placement(True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    dim1 = train_motif.shape[1]
    dim2 = train_seq.shape[1]
    pretrain_model = TFRobertaModel.from_pretrained("../robert_model125", output_attentions=True)
    model = build_model_test(dim1, dim2, pretrain_model)
    model.compile(optimizer=optimizers.RMSprop(lr=6.190058359322792e-05, epsilon=1e-08),
                  loss='binary_crossentropy',
                  metrics=['acc', metrics.AUC(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.Recall(), metrics.Precision()])

    save_file = model_path + model_name + '.h5'
    print(model.summary())

    model.fit([train_motif, train_motif_mask, train_seq, train_seq_mask],
              train_label,
              batch_size=8,
              epochs=5,
              verbose=2)
    model.save(save_file)

def test_model(test_motif,test_motif_mask,test_seq,test_mask, model_name, model_path):
    pred_prob = None
    results_df = pd.DataFrame()  
    model = tf.keras.models.load_model(model_path+ model_name + '.h5',compile=False,custom_objects={'TFRobertaModel':TFRobertaModel})
    pred_prob = model.predict([test_motif,test_motif_mask,test_seq,test_mask])
    results_df[f'pred_prob_model'] = pred_prob.flatten()
    prob = np.array(pred_prob)
    pro_file= model_name + '.npy'
    np.save(pro_file,prob)

def getMatrixLabel(positive_position_file_name):
    all_label = []
    file=pd.read_excel(positive_position_file_name)
    for index,row in file.iterrows():
        if row['PPI_Regulatory']==0:
            all_label.append(0)
        else:
            all_label.append(1)
    targetY =np.array(all_label)

    return targetY

if __name__ == "__main__":
    RES_TYPE = ['Y']

    model_name = "PhosPPI-SEQ_Y"
    model_path = f'./'

    train_file = f'train_data_Y.xlsx'
    test_file = f'test_data_Y.xlsx'

    train_feature = pd.read_excel(train_file, engine='openpyxl')
    test_feature = pd.read_excel(test_file, engine='openpyxl')

    tokenizer = AutoTokenizer.from_pretrained("../token")
    train_motif129 = train_feature['Sequence_193_pad'].tolist()
    test_motif129 = test_feature['Sequence_193_pad'].tolist()
    train_inputs129 = tokenizer(train_motif129, return_tensors="tf", padding=True)
    test_inputs129 = tokenizer(test_motif129, return_tensors="tf", padding=True)

    train_pad_token_index129 = tf.where((train_inputs129.input_ids == tokenizer.pad_token_id))
    test_pad_token_index129 = tf.where((test_inputs129.input_ids == tokenizer.pad_token_id))
    train_mask_index129 = tf.tensor_scatter_nd_update(train_inputs129['attention_mask'], train_pad_token_index129, tf.zeros((train_pad_token_index129.shape[0]), dtype=tf.int32))
    test_mask_index129 = tf.tensor_scatter_nd_update(test_inputs129['attention_mask'], test_pad_token_index129, tf.zeros((test_pad_token_index129.shape[0]), dtype=tf.int32))
    train_inputs129['attention_mask'] = train_mask_index129
    test_inputs129['attention_mask'] = test_mask_index129

    train_seq = train_feature['Sequence_pad'].tolist()
    test_seq = test_feature['Sequence_pad'].tolist()
    train_inputs = tokenizer(train_seq, return_tensors="tf", padding=True)
    test_inputs = tokenizer(test_seq, return_tensors="tf", padding=True)

    train_pad_token_index = tf.where((train_inputs.input_ids == tokenizer.pad_token_id))
    test_pad_token_index = tf.where((test_inputs.input_ids == tokenizer.pad_token_id))
    train_mask_index = tf.tensor_scatter_nd_update(train_inputs['attention_mask'], train_pad_token_index, tf.zeros((train_pad_token_index.shape[0]), dtype=tf.int32))
    test_mask_index = tf.tensor_scatter_nd_update(test_inputs['attention_mask'], test_pad_token_index, tf.zeros((test_pad_token_index.shape[0]), dtype=tf.int32))
    train_inputs['attention_mask'] = train_mask_index
    test_inputs['attention_mask'] = test_mask_index

    train_label = getMatrixLabel(train_file)

    # 训练模型
    train_model(train_inputs129['input_ids'], train_inputs129['attention_mask'], train_inputs['input_ids'], train_inputs['attention_mask'],train_label, model_name, model_path)

    # 测试模型
    test_model(test_inputs129['input_ids'], test_inputs129['attention_mask'], test_inputs['input_ids'], test_inputs['attention_mask'], model_name, model_path)
