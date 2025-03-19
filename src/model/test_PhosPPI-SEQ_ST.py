import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer,  TFRobertaModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

def test_model(test_motif,test_motif_mask,test_seq,test_mask, model_name, model_path):
    pred_prob = None

    results_df = pd.DataFrame()  
    model = tf.keras.models.load_model(model_path+'PhosPPI-SEQ_ST.h5',compile=False,custom_objects={'TFRobertaModel':TFRobertaModel})
    pred_prob = model.predict([test_motif,test_motif_mask,test_seq,test_mask])
    results_df[f'pred_prob_model'] = pred_prob.flatten()
    prob = np.array(pred_prob)
    pro_file=model_name + '.npy'
    np.save(pro_file,prob)

if __name__ == "__main__":
    RES_TYPE = ['ST']

    model_name = "PhosPPI-SEQ_ST"
    model_path = f'./'

    test_file = f'test_data_ST.xlsx'
    test_feature = pd.read_excel(test_file, engine='openpyxl')

    tokenizer = AutoTokenizer.from_pretrained("../token")
    test_motif129 = test_feature['Sequence_193_pad'].tolist()
    test_inputs129 = tokenizer(test_motif129, return_tensors="tf", padding=True)

    test_pad_token_index129 = tf.where((test_inputs129.input_ids == tokenizer.pad_token_id))
    test_mask_index129 = tf.tensor_scatter_nd_update(test_inputs129['attention_mask'], test_pad_token_index129, tf.zeros((test_pad_token_index129.shape[0]), dtype=tf.int32))
    test_inputs129['attention_mask'] = test_mask_index129

    test_seq = test_feature['Sequence_pad'].tolist()
    test_inputs = tokenizer(test_seq, return_tensors="tf", padding=True)

    test_pad_token_index = tf.where((test_inputs.input_ids == tokenizer.pad_token_id))
    test_mask_index = tf.tensor_scatter_nd_update(test_inputs['attention_mask'], test_pad_token_index, tf.zeros((test_pad_token_index.shape[0]), dtype=tf.int32))
    test_inputs['attention_mask'] = test_mask_index

    # 测试模型
    test_model(test_inputs129['input_ids'], test_inputs129['attention_mask'], test_inputs['input_ids'], test_inputs['attention_mask'], model_name, model_path)
