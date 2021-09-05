from collections import defaultdict

def load_pred(filename):
    pred_labels = []
    f = open(filename, 'r')
    line = f.readline()
    while True:
        try:
            line = f.readline()
            temp = line.strip().split(',')
            pred_labels.append(temp[1])
        except:
            break
    f.close()
    return pred_labels

K = 5
pred_all = []
for i in range(1,K+1):
    pred_all.append(load_pred('pred_' + str(i)))

K = 10
for i in range(1,K+1):
    pred_all.append(load_pred('../DIN-0725/pred_' + str(i)))

out = open('pred_0731_2.csv', 'w')
out.write('id,category\n')
for j in range(len(pred_all[0])):
    candidate_dict = defaultdict(int)
    for i in range(K):
        candidate_dict[pred_all[i][j]] += 1
    label_sorted = sorted(candidate_dict.items(), key = lambda x:x[1], reverse = True)
    label = label_sorted[0][0]
    out.write('%d,%s\n' %(j+1, label))
out.close()
