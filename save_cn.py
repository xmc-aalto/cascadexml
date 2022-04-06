import torch
import sys
import numpy as np
model = torch.load(f'{sys.argv[1]}/model_best_test.pth')
model = model['state_dict']
Cn = model['Cn.2.weight']
Cn_bias = model['Cn_bias.2.weight']

np.save(f'Dismec_init/{sys.argv[2]}_weights', Cn.cpu().numpy())
np.save(f'Dismec_init/{sys.argv[2]}_biases', Cn_bias.cpu().numpy())
