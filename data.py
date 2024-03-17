import re
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, DatasetDict

import pandas as pd


class CDialogueDataset:
    
    @staticmethod
    def load_dataset(tokenizer, seed=42, max_length=512):
        csv_path = 'CDialog Dataset.csv'
        df = pd.read_csv(csv_path)

        eos_token = tokenizer.eos_token
        eos_token_id = tokenizer.eos_token_id

        dialogues = defaultdict(list)
        conv_ids = df['conv_id'].unique()
        for conv_id in conv_ids:
            valid = True
            sub_df = df[df['conv_id'] == conv_id]

            utterances = []
            symptoms = []
            for i, row in sub_df.iterrows():
                # 처음 시작은 Patient, 한번씩 turn을 주고 받음
                valid = valid & (row['speaker'] == ('Patient' if i % 2 == 0 else 'Doctor'))
                
                utterance = row['utterance']
                speaker = row['speaker']
                symptom = row['symptom']
                symptom = str(symptom)
                
                sentence = '{}: {}'.format(speaker.strip(), utterance.strip())
                utterances.append(sentence)
                symptoms.append(symptom)

            if valid:
                dialogues['utterances'].append(utterances)
                dialogues['symptoms'].append(symptoms)
        dataset = Dataset.from_dict(dialogues)

        def process_doc(doc):
            text = ''.join([ ' {}{}'.format(utterance, eos_token) for utterance in doc['utterances'] ])
            text = eos_token + text  # NOTE: 처음 붙이냐에 따라 다름
            input_ids = tokenizer(text, max_length=max_length, truncation=True).input_ids
            labels = input_ids.copy()
            
            start = 0
            for i in range(labels.count(eos_token_id)):
                sep_idx = labels.index(eos_token_id, start) + 1

                if i % 2 == 0:  # 의사 발화는 label 제외
                    labels[start:sep_idx] = [-100] * (sep_idx - start)
                start = sep_idx
            
            return {'text': text,
                    'input_ids': input_ids,
                    'labels': labels}

        dataset = dataset.map(process_doc)
        dataset.set_format('torch', columns=['input_ids', 'labels'])

        # split dataset
        train_validtest = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
        train_ds = train_validtest['train']
        valid_test = train_validtest['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        valid_ds = valid_test['train']
        test_ds = valid_test['test']

        dataset = DatasetDict({
            'train': train_ds,
            'validation': valid_ds,
            'test': test_ds
        })

        return dataset
    
    @staticmethod
    def get_collate_fn(pad_token_id):
    
        def collate_fn(batch):
            input_ids = [ example['input_ids'] for example in batch ]
            labels = [ example['labels'] for example in batch ]
            
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            attention_map = torch.where(input_ids != pad_token_id, 1, 0)
            
            return {'input_ids': input_ids,
                    'attention_mask': attention_map,
                    'labels': labels}
        
        return collate_fn