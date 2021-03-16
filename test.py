import numpy as np
import os, re, random
import pickle


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)



label_path = '/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_label.pkl'
# load label
with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f)

label = np.array(label)

print(label.min(), label.max())

classes = [0,1,2,3,4,5,6,7,8,9,10,11]

annotations = []
for l in label:
    labels = [1*(classes==l)]
    annotations.append(labels)

annotations = np.array(annotations[-3])
print(annotations)