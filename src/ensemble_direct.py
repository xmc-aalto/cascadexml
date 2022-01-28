import tqdm
import torch
import numpy as np
from dataset import createDataCSV

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, required=False, default='')
parser.add_argument('--model2', type=str, required=False, default='')
parser.add_argument('--model3', type=str, required=False, default='')

parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')

args = parser.parse_args()

def predict(pred, tr, res):
    match = (pred[..., None] == tr).any(-1).double()
    res += torch.cumsum(match, dim=0)
    return res

def psp(pred, tr, num, den, inv_prop):
    # for pred, tr in zip(preds, y_true):
    match = (pred[..., None] == tr).any(-1).double()
    match[match > 0] = inv_prop[pred[match > 0]]
    num += torch.cumsum(match, dim=0)

    inv_prop_sample = torch.sort(inv_prop[tr], descending=True)[0]

    match = torch.zeros(5)
    match_size = min(tr.shape[0], 5)
    match[:match_size] = inv_prop_sample[:match_size]
    den += torch.cumsum(match, dim=0)
    return num, den

if __name__ == '__main__':
    using_group = args.dataset in ['wiki500k', 'amazon670k', 'AT670', 'WSAT', 'WT500']
    model_labels, model_scores = [], []

    models = [args.model1, args.model2, args.model3]
    models = [i for i in models if i != '']
    for model in models:
        print(f'loading {model}')
        model_scores.append(np.load(f'./results/{model}-scores.npy', allow_pickle=True))
        if using_group:
            model_labels.append(np.load(f'./results/{model}-labels.npy', allow_pickle=True))
    
    df, label_map = createDataCSV(args.dataset)
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    df = df[df.dataType == 'test']
    # results = {k:[0, 0, 0] for k in models + ['all']}
    res = torch.zeros(5)
    num, den = torch.zeros(5), torch.zeros(5)
    # inv_prop = torch.tensor(np.load('data/WikiSeeAlsoTitles-350K/inv_prop.npy'))
    inv_prop = torch.tensor(np.load('data/WikiTitles-500K/inv_prop.npy'))

    bar = tqdm.tqdm(total=len(df))

    inv_lab_map = {v:k for k,v in label_map.items()}
    for i, true_labels in enumerate(df.label.values):
        # true_labels = set([label_map[i] for i in true_labels.split()])
        true_labels = set([int(i) for i in true_labels.split()])

        bar.update()

        # import pdb; pdb.set_trace()
        preds = [int(inv_lab_map[p]) for p in model_labels[0][i][:5]]
        num, den = psp(torch.tensor(preds), torch.tensor(list(true_labels)), num, den, inv_prop)
        res = predict(torch.tensor(preds), torch.tensor(list(true_labels)), res)


        # if using_group:
        #     pred_labels = {}
        #     for j in range(len(models)):
        #         # results[models[j]][0] += len(set([model_labels[j][i][0]]) & true_labels)
        #         # results[models[j]][1] += len(set(model_labels[j][i][:3]) & true_labels)
        #         # results[models[j]][2] += len(set(model_labels[j][i][:5]) & true_labels)
                
        #         preds = [inv_lab_map[p] for p in models[j][i][:5]]
        #         import pdb; pdb.set_trace()
        #         num, den = psp(torch.tensor(preds), torch.tensor(list(true_labels)), num, den, inv_prop)
        #         # for l, s in sorted(list(zip(model_labels[j][i], model_scores[j][i])), key=lambda x: x[1], reverse=True):
        #         #for l, s in zip(model_labels[j][i], model_scores[j][i]):
        #     #         if l in pred_labels:
        #     #             pred_labels[l] += s
        #     #         else:
        #     #             pred_labels[l] = s
        #     # pred_labels = [k for k, v in sorted(pred_labels.items(), key=lambda item: item[1], reverse=True)]
            
        #     # results['all'][0] += len(set([pred_labels[0]]) & true_labels)
        #     # results['all'][1] += len(set(pred_labels[:3]) & true_labels)
        #     # results['all'][2] += len(set(pred_labels[:5]) & true_labels)
        # else:
        #     index = i
        #     logits = [torch.sigmoid(torch.tensor(model_scores[i][index])) for i in range(len(models))]
        #     logits.append(sum(logits))
        #     logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        #     for i, logit in enumerate(logits):
        #         name = models[i] if i != len(models) else 'all'
        #         results[name][0] += len(set([logit[0]]) & true_labels)
        #         results[name][1] += len(set(logit[:3]) & true_labels)
        #         results[name][2] += len(set(logit[:5]) & true_labels)

    total = len(df)

    psp = num*100/den
    prec = res*100/(i * np.arange(1, 6))
    print(f'p@1: {prec[0]:.2f}, p@3: {prec[2]:.2f}, p@5: {prec[4]:.2f}, psp@1: {psp[0]:.2f}, psp@3: {psp[2]:.2f}, psp@5: {psp[4]:.2f}')
    
    # for k in results:
    #     p1 = results[k][0] / total
    #     p3 = results[k][1] / total / 3
    #     p5 = results[k][2] / total / 5
    #     print(f'{k}: p1:{p1} p3:{p3} p5:{p5}')
