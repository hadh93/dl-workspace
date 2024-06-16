#%%
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
broken_image = torch.FloatTensor(pickle.load(open('./broken_image_t.p', 'rb'), encoding='latin1'))

plt.imshow(broken_image.view(100, 100))
plt.show()
