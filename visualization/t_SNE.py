import sklearn.metrics as skm
import numpy as np
import torch.nn as nn
import torch
import pickle
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
sys.path.append('/home/liyang/github repos/CS598_PROJECT-20240428/CS598_PROJECT')

global_params = {
    'batch_size': 256,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir': './CS598_PROJECT/modeloutput/', # output folder
    'best_name': 'NextXVisit',  # output model name
    'model_name': 'MLM_MODEL', # model name
    'max_len_seq': 64,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit':2
}


NextXVisit_model_file = os.path.join(global_params['output_dir'],global_params['best_name'])
MLM_model_file = os.path.join(global_params['output_dir'],global_params['model_name'])

def load_model(path):
    # load pretrained model and update weights
    model_dict = torch.load(path)
    # print("pretrained_dict")
    # for k, v in pretrained_dict.items():
    #     print(k, v.shape)
    # print("model_dict")
    # for k, v in model_dict.items():
    #     print(k, v.shape)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in model_dict.items()}
    # 2. overwrite entries in the existing state dict
    return pretrained_dict

def get_category(code):
    try:
        code = int(code)
        if 1 <= code <= 139:
            return 'Infectious and Parasitic Diseases'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        elif 240 <= code <= 279:
            return 'Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders'
        elif 280 <= code <= 289:
            return 'Diseases of the Blood and Blood-forming Organs'
        elif 290 <= code <= 319:
            return 'Mental, Behavioral and Neurodevelopmental Disorders'
        elif 320 <= code <= 389:
            return 'Diseases of the Nervous System and Sense Organs'
        elif 390 <= code <= 459:
            return 'Diseases of the Circulatory System'
        elif 460 <= code <= 519:
            return 'Diseases of the Respiratory System'
        elif 520 <= code <= 579:
            return 'Diseases of the Digestive System'
        elif 580 <= code <= 629:
            return 'Diseases of the Genitourinary System'
        elif 630 <= code <= 679:
            return 'Complications of Pregnancy, Childbirth, and the Puerperium'
        elif 680 <= code <= 709:
            return 'Diseases of the Skin and Subcutaneous Tissue'
        elif 710 <= code <= 739:
            return 'Diseases of the Musculoskeletal System and Connective Tissue'
        elif 740 <= code <= 759:
            return 'Congenital Anomalies'
        elif 760 <= code <= 779:
            return 'Certain Conditions Originating in the Perinatal Period'
        elif 780 <= code <= 799:
            return 'Symptoms, Signs, and Ill-defined Conditions'
        elif 800 <= code <= 999:
            return 'Injury and Poisoning'
        else:
            return 'Other'
    except ValueError:
        if code.startswith('E'):
            return 'Supplementary Classification of External Causes of Injury and Poisoning'
        elif code.startswith('V'):
            return 'Supplementary Classification of Factors Influencing Health Status and Contact with Health Services'
        else:
            return 'Other'
        
with open('./CS598_PROJECT/output/token2idx.pkl', 'rb') as f:
    numpy_array = pickle.load(f)
idx2token = numpy_array['idx2token']


nextxvisit = load_model(NextXVisit_model_file)
embedding_data = nextxvisit['bert.embeddings.word_embeddings.weight'].detach().cpu().numpy()

model = load_model(MLM_model_file)
tmp = model['bert.embeddings.word_embeddings.weight'].detach().cpu().numpy()


similarities = cosine_similarity(tmp, embedding_data)
most_similar_indices = np.argmax(similarities, axis=1)
most_similar_codes = [idx2token[idx] for idx in most_similar_indices]



diseases_labels = [get_category(code) for code in most_similar_codes]
categories = list(set(diseases_labels))


# Step 2: t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(embedding_data)

# Step 3: plt
plt.figure(figsize=(20, 8))
for category in categories:
    indices = [i for i, label in enumerate(diseases_labels) if label == category]
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=category)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE Visualization of Disease Embeddings")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
