import numpy as np

predict = np.load('total_predict.npy')
updown_data = np.load('../result_zero_sigma.npy')

def f(x): return '/'.join([i[1:] if i[0]
                               == '0' else i for i in x.split('/')])
updown_dict = {f(k): int(v) for k, v in updown_data}

# confusion_matrix = np.zeros(2,2)
tp,tn,fp,fn = 0,0,0,0
garbage = 0
for i in predict:
    predict_value = int(i[1])
    ground_true = updown_dict[i[0]]
    if(predict_value==0):
        garbage+=1
    elif(ground_true==1):
        if(predict_value==1):
            tp+=1
        elif(predict_value==-1):
            tn+=1
    else:
        if(predict_value==1):
            fp+=1
        elif(predict_value==-1):
            fn+=1
print('出手率',1-garbage/len(predict))
print('Accuracy : ',(tp+fn)/(tp+tn+fp+fn))

            
