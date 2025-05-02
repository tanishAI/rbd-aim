import os
import json
import subprocess
import sys
import pickle
import random

import numpy as np
import pandas as pd

from abnumber import Chain
from pathlib import Path
from tqdm import tqdm

import transformers
from transformers import AdamW

import torch
from torch.utils.data import DataLoader,  Dataset

from epitope_model import AntiBERTyFAB_CDR_Pair, AntiBERTyFAB_CLS_Pair, EpitopesPairDatasetCDR


def similarity(stest,strain, CDR_only = True):#True):
    l_ids_cdr = list(range(24,35))+list(range(50,57))+list(range(89,98))
    h_ids_cdr = list(range(31,36))+list(range(50,66))+list(range(95,103))

    alignment = stest.align(strain)

    n = 0
    nt = 0
    for pos, (aa, bb) in alignment:
        i = str(pos)[1:]
        if not i[-1].isdigit():
            i = i[:-1]
        i = int(i)
        if CDR_only and str(pos)[0]=="L" and i not in l_ids_cdr:
            continue
        if CDR_only and str(pos)[0]=="H" and i not in h_ids_cdr:
            continue
        if aa == bb:
            n += 1
        nt += 1

    return n/nt

def compare_seqs_anarci(dataset, target_seq_l, target_seq_h):
    if isinstance(target_seq_l, str):
        l = Chain(target_seq_l.replace("-",""), scheme="chothia")
        h = Chain(target_seq_h.replace("-",""), scheme="chothia")
    else:
        l = target_seq_l
        h = target_seq_h
            
    dataset_ = dataset[dataset["good"].isin([0,1])]
    scores = []
    poss = []
    seqs = []
    hits_ = []

    for j,d_train in enumerate(dataset_.iloc()):
        strain = d_train["light_anarci_abnumber"]
        s1 = similarity(l, strain)
        strain = d_train["heavy_anarci_abnumber"]
        s2 = similarity(h, strain)
        scores.append(0.5*(s1+s2))
        hits_.append(d_train)
        poss.append(d_train["POS_class"])

    hits = {i:None for i in range(12)}
    vals = {i:0 for i in range(12)}
    for i,[s,p] in enumerate(zip(scores,poss)):
        if vals[p]<s:
            vals[p] = s
            hits[p] = hits_[i]

    return [vals[ii] for ii in range(12)], [hits[ii] for ii in range(12)]



 def train_and_evaluate_model(model,
                             train_dataloader,
                             test_dataloader, 
                             config):
    device = config["device"]
    model.to(device)
    num_epochs = config["epochs"]
    model.train()
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(num_epochs):
        losses = []
        i = 0
        for batch in tqdm(train_dataloader, desc='Epoch ' + str(epoch+1), position=0, leave=True):
            batch = {k: v.to(device) for k, v in batch.items()}  # Send input data to the device (GPU or CPU)
            batch.pop("antibody_id")
            #batch.pop("cdr_ids_light")
            #batch.pop("cdr_ids_heavy")
            outputs = model(**batch)
            losses.append(outputs["loss"].item())
            i+=1
            val = np.average(losses[:-25])
            if i%100==0:
                print(val)

            loss = outputs["loss"]
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=0.1
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #break

    # Evaluate the model on a separate validation dataset
    model.eval()  # Set the model in evaluation mode
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}

    with torch.no_grad():
        # Validation loop
        i = 0
        for batch in test_dataloader:
            i+=1
            batch = {k: v.to(device) for k, v in batch.items()}  # Send input data to the device (GPU or CPU)
            labels = batch.pop("labels").detach().cpu().numpy()
            batch["labels"] = None
            fab_ids = batch.pop("antibody_id").detach().cpu().numpy()
            outputs = model(**batch)
            pred = outputs["logits"].detach().cpu().numpy()
            test_predictions["predictions"].append(pred)
            test_predictions["labels"].append(labels)
            test_predictions["antibody_ids"].append(fab_ids)
            test_predictions["embeddings"].append(outputs["embeddings"].detach().cpu().numpy())

    return test_predictions


def run_experiment(dataset,
                   config,
                   device = "cuda:0",
                   model_type = "CDR",
                   cv_id = 0,
                   ensemble_id = 0,
                   full_model=True,
                   rewrite=False,
                   experiment_name = "AntiBERTyFAB_Pair",
                  out_dir='./weights',
                  num_classes=12):
    model_name = f'{out_dir}/weights/NN_cv_{cv_id}_{model_type}_ens_{ensemble_id}.pth'

    if not rewrite and os.path.exists(model_name):
        return

    cv_predictions = None
    dataset["our_test"] = [True if d else False for d in dataset["our_test"].iloc()]
    pred_per_cv = {}

    model = None
    #for cv_id in range(10):
    if True:
        q1 = [True if d else False for d in dataset[f"cross_val_test_{cv_id}"].iloc()]
        q2 = [not q for q in q1]

        train_df = dataset[q2]
        test_df  = dataset[q1]

        if full_model:
            train_df = dataset[~dataset["our_test"]]

        ds_train = EpitopesPairDatasetCDR(train_df, config["num_classes"])
        ds_test = EpitopesPairDatasetCDR(test_df, config["num_classes"])

        train_dataloader = DataLoader(ds_train,
                                      batch_size=1,
                                      shuffle=True)

        test_dataloader = DataLoader(ds_test,
                                      batch_size=1,
                                      shuffle=False)

        if model_type == "CLS":
            model = AntiBERTyFAB_CLS_Pair(config)
        else:
            model = AntiBERTyFAB_CDR_Pair(config)

        pred = train_and_evaluate_model(model,
                                        train_dataloader,
                                        test_dataloader, 
                                        config)

        pred_per_cv[cv_id] = pred
        if cv_predictions is None:
            cv_predictions = pred
        else:
            for k,v in pred.items():
                cv_predictions[k].extend(v)

    cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
    cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
    dataset[experiment_name+"_predictions"] = None
    dataset[experiment_name+"_predictions"] = dataset.apply(lambda row: cv[row['index']]
                                                 if row['index'] in cv else row[experiment_name+"_predictions"],
                                                            axis=1)
    for cv_id in pred_per_cv:
        cv_predictions = pred_per_cv[cv_id]
        cv_predictions = {k:np.concatenate(v,axis=0) for k,v in cv_predictions.items()}
        cv = {i:v for i,v in zip(cv_predictions["antibody_ids"], cv_predictions["predictions"])}
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = None
        dataset[experiment_name+f"_predictions_cv_{cv_id}"] = dataset.apply(lambda row: cv[row['index']]
                                                     if row['index'] in cv else row[experiment_name+"_predictions"],
                                                                axis=1)


    model.eval()
    model.to("cpu")

    model_name = f'{out_dir}/weights/NN_cv_{cv_id}_{model_type}_ens_{ensemble_id}.pth'

    torch.save(model.state_dict(), model_name)
    return dataset

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


def evaluate_model(model, test_data_loader, config):
    model.eval()
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}
    with torch.no_grad():
        i = 0
        for batch in test_data_loader:
            i+=1
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            labels = batch.pop("labels").detach().cpu().numpy()
            batch["labels"] = None
            fab_ids = batch.pop("antibody_id").detach().cpu().numpy()
            outputs = model(**batch)
            pred = outputs["logits"].detach().cpu().numpy()
            test_predictions["predictions"].append(pred)
            test_predictions["labels"].append(labels)
            test_predictions["antibody_ids"].append(fab_ids)
            test_predictions["embeddings"].append(outputs["embeddings"].detach().cpu().numpy())

    return test_predictions

def get_closest_hits_by_levinstein(data,
                                   dataset):
    fabs = []
    scores=  []

    template_id = []
    template_class = []
    other_hits = []


    template_id_NN = []
    template_class_NN = []

    for uid,d_ in enumerate(data.iloc()):
            
        ps, hits = compare_seqs_anarci(dataset=dataset[dataset["Name"]!=d_["Name"]],
                                       target_seq_h=d_["heavy_anarci_abnumber"],
                                       target_seq_l=d_["light_anarci_abnumber"])        
        ps = np.average([ps],axis=0).flatten()
        id_close = list(reversed(np.argsort(ps)))[0]
        score = ps[id_close]
        d_close = hits[id_close]
        # print(score,id_close,ps)        

        
        if "logits" in data:
            d_close_NN = hits[np.argmax(d_["logits"])]
            template_id_NN.append((d_close_NN["heavy_anarci"],
                            d_close_NN["light_anarci"],
                            d_close_NN["pdb_ids"]))
      
        template_id.append((d_close["heavy_anarci"],
                            d_close["light_anarci"],
                            d_close["pdb_ids"]))
        template_class.append(d_close["POS_class"])

    data["template_pred_levin"] = template_id
    data["POS_class_pred_template_levin"] = template_class

    if len(template_id_NN)!=0:
        data["template_pred_levin_NN"] = template_id_NN
        
    
    d_test = data[data["good"].isin([1])]
    
    print(list(d_test["POS_class"]))
    print(list(d_test["POS_class_pred_template_levin"]))
    print(class_metrics(d_test["POS_class"],
                        d_test["POS_class_pred_template_levin"]))

    return data
        
def get_closest_hit(with_classifier = False, data_with_hits = None, data_with_emb = None, data_pred_file=None, out_dir='./'):   
    if data_with_hits:
        data = pickle.load(open(data_with_hits,'rb'))
        data_pred = pickle.load(open(data_pred_file,'rb'))
        data_pred["embeddings"] = data["embeddings"]
    else:
        data =  pickle.load(open(data_with_emb,'rb'))
        data_pred = pickle.load(open(data_pred_file,'rb'))
        data_pred["embeddings"] = data["embeddings"]
        data["logits"] = data_pred["logits"]
        data = get_closest_hits_by_levinstein(data, data[data["good"]==1])
        pickle.dump(data,
                    open(f"{out_dir}/predictions/dataset_production_all_w_hits.pkl",'wb'))

    data_ref = data[data["good"]==1]
    data_pred["POS_class_pred_template_levin"] = data["POS_class_pred_template_levin"]

    n = 0

    template_class = []
    template_id = []
    embeddings_all = np.array([np.array(dd["embeddings"]) for dd in data_ref.iloc()])[:,0,:]
    
    for i,d in enumerate(data_pred.iloc()):
        seq_test = (d["heavy_anarci"],
                    d["light_anarci"])

        if with_classifier:
            data_ref_ = data_ref[data_ref["POS_class"]==np.argmax(d["logits"])]
            embeddings = np.array([np.array(dd["embeddings"]) for dd in data_ref_.iloc()])[:,0,:]
        else:
            data_ref_ = data_ref
            embeddings = embeddings_all
            
        class_test = d["POS_class"]
        
        e_cur = data_pred.iloc()[i]["embeddings"]
        e_close = np.linalg.norm(embeddings-e_cur,axis=1)
        id_close = np.argsort(e_close)[1]
        
        d_close = data_ref_.iloc()[id_close]
        
        seq_pred  = (d_close["heavy_anarci"],
                     d_close["light_anarci"])
        class_pred = d_close["POS_class"]
        
        template_id.append((d_close["heavy_anarci"],
                            d_close["light_anarci"],
                            d_close["pdb_ids"]))
        
        template_class.append(d_close["POS_class"])
        
    d_test = data_pred[data_pred["good"].isin([1])]
    
    print(class_metrics(d_test["POS_class"],
                        d_test["POS_class_pred_template_levin"]))

        
    if with_classifier:
        data_pred["template_pred"] = template_id
        data_pred["POS_class_pred_template"] = template_class
        d_test = data_pred[data_pred["good"].isin([1])]        
        print(class_metrics(d_test["POS_class"],
                            d_test["POS_class_pred_template"]))

    if not with_classifier:
        data_pred["template_pred_no_NN"] = template_id
        data_pred["POS_class_pred_template_no_NN"] = template_class
        d_test = data_pred[data_pred["good"].isin([1])]
        print(class_metrics(d_test["POS_class"],
                            d_test["POS_class_pred_template_no_NN"]))

        
    return data_pred


def calculate_embeddings(dataset, config, num_classes=12):
    #dataset = dataset[~dataset["our_test"]]

    ds = EpitopesPairDatasetCDR(dataset, config["num_classes"])

    dataloader = DataLoader(ds,
                            batch_size=1,
                            shuffle=False)
    model = AntiBERTyFAB_CDR_Pair(config)
    device = config["device"]
    model.to(device)
    model.eval()
    
    test_predictions = {"predictions":[],
                        "labels":[],
                        "antibody_ids":[],
                        "embeddings":[]}
    with torch.no_grad():
        i = 0
        for batch in dataloader:
            i+=1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").detach().cpu().numpy()
            batch["labels"] = None
            fab_ids = batch.pop("antibody_id").detach().cpu().numpy()
            outputs = model(**batch)
            pred = outputs["logits"].detach().cpu().numpy()
            test_predictions["predictions"].append(pred)
            test_predictions["labels"].append(labels)
            test_predictions["antibody_ids"].append(fab_ids)
            test_predictions["embeddings"].append(outputs["embeddings"].detach().cpu().numpy())
    dataset["embeddings"] = test_predictions["embeddings"]
    return dataset

def test_classifier_cv(dataset, config, cv_id = 0, models = ["CLS","CDR"], w_dir = './'):
    q1 = [True if d else False for d in dataset[f"cross_val_test_{cv_id}"].iloc()]
    q2 = [not q for q in q1]
    test_df  = dataset[q1]
    ds_test = EpitopesPairDatasetCDR(test_df, config["num_classes"])
    test_dataloader = DataLoader(ds_test,
                                 batch_size=1,
                                 shuffle=False)
    predictions = []

    for model_type in models:
        for i in range(1):
            weights_path = f"{w_dir}/weights/NN_cv_{cv_id}_{model_type}_ens_{i}.pth"
            if model_type=="CLS":
                model = AntiBERTyFAB_CLS_Pair(config)
            if model_type=="CDR":
                model = AntiBERTyFAB_CDR_Pair(config)
                
            state_dict = torch.load(weights_path)
            
            model.load_state_dict(state_dict)
            model.eval()
            model.to(config["device"])

            pred = evaluate_model(model,
                                  test_dataloader,
                                  config)
            logits = pred["predictions"]
            predictions.append(logits)

    predictions = np.average(predictions,axis=0)[:,0,:]
    for uid,i in enumerate(test_df.index):
        dataset.at[i, 'logits'] = predictions[uid]
    return dataset


def test_classifier_all_cv_models(dataset, config, m_dir='./'):
    dataset["logits"] = None
    for cv_id in range(10):
        test_classifier_cv(dataset, cv_id, models=config['model_type'], w_dir=m_dir)
    Path(f"{m_dir}/predictions/").mkdir(exist_ok=True)
    pickle.dump(dataset, open(f"{m_dir}/predictions/dataset_with_predictions_{config['model_type']}.pkl",'wb'))



