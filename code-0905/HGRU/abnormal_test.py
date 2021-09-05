def get_pred_label(filename):
    labels = []
    f = open(filename, 'r')
    f.readline()
    while True:
        try:
            line = f.readline()
            temp = line.strip().split(',')
            # user_id user_click_history user_protrait exposed_items labels time        
            labels.append(list(map(int, temp[1].split(' '))))
        except:
            break
    return labels

labels = get_pred_label('pred_on_test_0704')

sum_dict = {}
for d in labels:
    a, b, c = sum(d[:3]), sum(d[3:6]), sum(d[6:])
    try:
        sum_dict[(a,b,c)] += 1
    except:
        sum_dict[(a,b,c)] = 1

cnt = 0
for key, value in sum_dict.items():
    if key[0] < 3 and (key[1]+key[2]) > 0:
        cnt += value
    if key[0] == 3 and key[1] < 3 and key[2] > 0:
        cnt += value
print(cnt)