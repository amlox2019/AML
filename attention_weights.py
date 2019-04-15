import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = '26'
file_name = model_name+"_attention_weights"
file_name = 'RNNAw30one_batch_attention_weights'

example_id = 90

def cut_eos(sent):
    cut_id = len(sent)
    for i, word in enumerate(sent):
        #print(word)
        if word == '<pad>':
            cut_id = i
            break
    return sent[:i]

with open(file_name, "rb") as file:
    data = pickle.load(file)
    print(data['weights'][example_id].size())
    print(len(data['weights']), len(data['source']), len(data['translation']))
    #print(data['source'], data['translation'])
    weights = data['weights'][example_id].t()
    source = cut_eos(data['source'][example_id].split(' '))
    hypothesis = data['translation'][example_id].split(' ')

print(weights.size())


fig, ax = plt.subplots()
im = ax.imshow(weights.cpu().numpy()[:len(source),:len(hypothesis)], cmap='gray', interpolation='nearest')

ax.set_xticks(np.arange(len(hypothesis)))
ax.set_yticks(np.arange(len(source)))

ax.set_xticklabels(hypothesis)
ax.set_yticklabels(source)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax.set_title('Attention')
fig.tight_layout()
plt.savefig('attention'+file_name)
plt.show()
