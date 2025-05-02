import os
import numpy as np

import transformers
import antiberty
from antiberty import AntiBERTy

import torch
from torch import nn
from torch.utils.data import Dataset

project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')
CHECKPOINT_PATH = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')

class EpitopesPairDatasetCDR(Dataset):
    """
    Dataset of labeled Fab pairs
    and corresponding epitope binding class
    """
    def __init__(self, df, n_epitopes):
        """
        :param df: input PandasDataframe
        :param n_epitopes: number of epitope classes
        """
        self.df    = df
        self.n_epitopes = n_epitopes

        VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')

        self.tokenizer = transformers.BertTokenizer(vocab_file=VOCAB_FILE,
                                                    do_lower_case=False)

        self.indices = self.df["index"].to_list()

        self.cdr_ids_l = self.df["light_ids_cdr"].to_list()
        self.cdr_ids_h = self.df["heavy_ids_cdr"].to_list()

        for i in range(len(self.cdr_ids_l)):
            self.cdr_ids_l[i] = np.array(self.cdr_ids_l[i])
            self.cdr_ids_l[i]-=1

        for i in range(len(self.cdr_ids_h)):
            self.cdr_ids_h[i] = np.array(self.cdr_ids_h[i])
            self.cdr_ids_h[i]-=1

        self.sequences_l = self.df["light_anarci"].to_list()
        self.sequences_h = self.df["heavy_anarci"].to_list()

        self.class_labels = np.eye(self.n_epitopes)[self.df['POS_class'].to_numpy()]
        self.max_length = 150

    def __len__(self):
        return len(self.sequences_l)
    
    def __getitem__(self, idx):
        sequence_l = " ".join(list(self.sequences_l[idx]))
        sequence_h = " ".join(list(self.sequences_h[idx]))

        cdr_ids_l = self.cdr_ids_l[idx]
        cdr_ids_h = self.cdr_ids_h[idx]

        class_label = self.class_labels[idx]
        
        encoding_l = self.tokenizer.encode_plus(
            sequence_l,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        encoding_h = self.tokenizer.encode_plus(
            sequence_h,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )        
        # Return the tokenized sequence and corresponding class label
        return {
            'antibody_id': self.indices[idx], 
            'input_ids_light': encoding_l['input_ids'].flatten(),
            'attention_mask_light': encoding_l['attention_mask'].flatten(),
            'input_ids_heavy': encoding_h['input_ids'].flatten(),
            'attention_mask_heavy': encoding_h['attention_mask'].flatten(),
            'cdr_ids_light': torch.tensor(cdr_ids_l, dtype=torch.long),
            'cdr_ids_heavy': torch.tensor(cdr_ids_h, dtype=torch.long),
            'labels': torch.tensor(class_label, dtype=torch.long)
        }

class AntiBERTyFAB_CLS_Pair(nn.Module):
    """
    Model predicting the epitope class based on concatenated CLS embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        self.model_l = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.model_h = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.epitopes = nn.Linear(self.model_l.config.hidden_size*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                input_ids_light, 
                attention_mask_light, 
                cdr_ids_light,
                input_ids_heavy, 
                attention_mask_heavy, 
                cdr_ids_heavy,
                labels=None,
                return_embeddings = False):
        
        outputs_l = self.model_l.bert(
                input_ids=input_ids_light,
                attention_mask=attention_mask_light,
                output_hidden_states=True,
                output_attentions=True)

        outputs_h = self.model_h.bert(
                input_ids=input_ids_heavy,
                attention_mask=attention_mask_heavy,
                output_hidden_states=True,
                output_attentions=True)

        _,sequence_output_l = outputs_l[:2]
        _,sequence_output_h = outputs_h[:2]

        embeddings = torch.concat([sequence_output_l, sequence_output_h], axis=-1)
        pred = self.epitopes(embeddings)
        if labels is not None:
            loss = self.loss(pred.view(-1), labels.view(-1).float())
        else:
            loss = None

        return {"logits":pred,
                "loss": loss, 
                "labels":labels,
                "embeddings":embeddings}

class AntiBERTyFAB_CDR_Pair(nn.Module):
    """
    Model predicting the epitope class based on concatenated average CDS embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        self.model_l = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.model_h = AntiBERTy.from_pretrained(CHECKPOINT_PATH)
        self.epitopes = nn.Linear(self.model_l.config.hidden_size*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, 
                input_ids_light, 
                attention_mask_light, 
                cdr_ids_light,
                input_ids_heavy, 
                attention_mask_heavy, 
                cdr_ids_heavy,
                labels=None,
                return_embeddings = False):

        outputs_l = self.model_l.bert(
                input_ids=input_ids_light,
                attention_mask=attention_mask_light,
                output_hidden_states=True,
                output_attentions=True)

        outputs_h = self.model_h.bert(
                input_ids=input_ids_heavy,
                attention_mask=attention_mask_heavy,
                output_hidden_states=True,
                output_attentions=True)

        cdr_l = torch.index_select(outputs_l[0], dim=1, index=cdr_ids_light[0])
        cdr_h = torch.index_select(outputs_h[0], dim=1, index=cdr_ids_heavy[0])

        cdr_l = torch.mean(cdr_l, dim=1)
        cdr_h = torch.mean(cdr_h, dim=1)

        embeddings = torch.concat([cdr_l, cdr_h],axis=-1)
        pred = self.epitopes(embeddings)        

        if labels is not None:
            loss = self.loss(pred.view(-1), labels.view(-1).float())
        else:
            loss = None

        return {"logits":pred, 
                "loss": loss, 
                "labels":labels,
                "embeddings":embeddings}



class ESMEpitopesPairDatasetCDR(Dataset):
    def __init__(self, df, n_epitopes=None):

        _, self.tokenizer = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = self.tokenizer.get_batch_converter()
        
        self.df = df
        self.n_epitopes = n_epitopes

        self.indices = self.df["index"].to_list()

        self.cdr_ids_l = [np.array(ids) - 1 for ids in self.df["light_ids_cdr"]]
        self.cdr_ids_h = [np.array(ids) - 1 for ids in self.df["heavy_ids_cdr"]]

        self.sequences_l = self.df["light_anarci"].to_list()
        self.sequences_h = self.df["heavy_anarci"].to_list()

        if self.n_epitopes:
            self.class_labels = np.eye(self.n_epitopes)[self.df['POS_class'].to_numpy()]
        else:
            self.class_labels = self.df['KC_50'].to_numpy()

    def __len__(self):
        return len(self.sequences_l)

    def __getitem__(self, idx):
        sequence_l = self.sequences_l[idx]
        sequence_h = self.sequences_h[idx]

        # ESM expects a string with a prefix
        # sequence_l = f">light\n{sequence_l}"
        # sequence_h = f">heavy\n{sequence_h}"

        _, _, tokens_l = self.esm_batch_converter([('' , sequence_l)])
        _, _, tokens_h = self.esm_batch_converter([('' , sequence_h)])
        # tokens_l = self.tokenizer(sequence_l, return_tensors="pt")
        # tokens_h = self.tokenizer(sequence_h, return_tensors="pt")

        return {
            'antibody_id': self.indices[idx],
            'input_light': tokens_l[0],
            'input_heavy': tokens_h[0],
            'cdr_ids_light': torch.tensor(self.cdr_ids_l[idx], dtype=torch.long),
            'cdr_ids_heavy': torch.tensor(self.cdr_ids_h[idx], dtype=torch.long),
            'labels': torch.tensor(self.class_labels[idx], dtype=torch.long)
        }


class ESM_CDR_Pair(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        # self.model_l, _ = esm.pretrained.esm2_t33_650M_UR50D()
        # self.model_h, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.epitopes = nn.Linear(1280*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                input_light,
                input_heavy,
                cdr_ids_light,
                cdr_ids_heavy,
                labels=None,
                return_embeddings=False):

        outputs_l = self.esm.forward(input_light, repr_layers=[33])
        outputs_h = self.esm.forward(input_heavy, repr_layers=[33])

        rep_l = outputs_l["representations"][33].squeeze(0)
        rep_h = outputs_h["representations"][33].squeeze(0)

        cdr_l = torch.index_select(rep_l, dim=0, index=cdr_ids_light[0])
        cdr_h = torch.index_select(rep_h, dim=0, index=cdr_ids_heavy[0])

        cdr_l = torch.mean(cdr_l, dim=0)
        cdr_h = torch.mean(cdr_h, dim=0)

        embedding = torch.cat([cdr_l, cdr_h], dim=-1).unsqueeze(0)

        pred = self.epitopes(embedding).squeeze(1)

        loss = self.loss(pred.view(-1), labels.view(-1).float()) if labels is not None else None

        return {
            "logits": pred,
            "loss": loss,
            "labels": labels,
            "embeddings": embedding 
        }

class ESM_CLS_Pair(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.num_epitopes = config["num_classes"]
        self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.epitopes = nn.Linear(1280*2, self.num_epitopes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                input_light,
                input_heavy,
                cdr_ids_light,
                cdr_ids_heavy,
                labels=None):

        outputs_l = self.esm.forward(input_light, repr_layers=[33])
        outputs_h = self.esm.forward(input_heavy, repr_layers=[33])

        rep_l = outputs_l["representations"][33][:,0,:]
        rep_h = outputs_h["representations"][33][:,0,:]

        embedding = torch.cat([rep_l, rep_h], dim=-1)

        pred = self.epitopes(embedding).squeeze(1)

        loss = self.loss(pred.view(-1), labels.view(-1).float()) if labels is not None else None

        return {
            "logits": pred,
            "loss": loss,
            "labels": labels,
            "embeddings": embedding
        }