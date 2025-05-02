from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  roc_auc_score, \
precision_recall_fscore_support, matthews_corrcoef

def class_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    metrics = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1": f1,
               "MCC": mcc}
    return metrics

def calculate_metrics_NN(data):
    data = data[data["good"].isin([0,1])]
    pred = np.array([np.array(d_["logits"]) for d_ in data.iloc()])
    pred = np.argmax(pred,axis=1)
    c = class_metrics(data["POS_class"], pred)
    print('NN', c)
    if 'POS_class_pred_template_levin' in data.columns:
        cl = class_metrics(data["POS_class"], data['POS_class_pred_template_levin'])
        print('LEV', cl)
    return c

def calculate_metrics_NN_pae(data_nn, num_classes=12):
    n = 0
    data_nn = data_nn[data_nn["good"].isin([0,1])]
    pos_adj = []
    names_adj = []
    for d_ in data_nn.iloc():
        class_pred = np.argmax(data_nn[data_nn["Name"]==d_["Name"]].iloc()[0]["logits"])
        pae_nn = float(d_["PAE_NN"])
        if pae_nn < 13.5:#10.6:
            n = n + 1
            names_adj.append(d_["Name"])
            class_pred = d_["POS_class_pred_template_levin"]
                
        pos_adj.append(class_pred)
    
    data_nn["POS_pred_adj"] = pos_adj
    c = class_metrics(data_nn["POS_class"], data_nn["POS_pred_adj"])
    print("adjusted metrics", c)
    data_x = data_nn[data_nn["good"]==1]
    c = class_metrics(data_x["POS_class"], data_x["POS_pred_adj"])
    print("adjusted metrics for PDB subset", c)
    return names_adj
