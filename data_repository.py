import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings
import json
import os
import ast

warnings.filterwarnings("ignore")

class DataRepository:
    def __init__(self, data_path: str):
        self.records = []
        self.simple_ds = data_path
        self.ds_with_embeddings = data_path.replace('.jsonl', '_with_embeddings.jsonl')

    def load_data(self):
        print(f"Loading {self.simple_ds}...")
        if os.path.exists(self.ds_with_embeddings):
            df = pd.read_json(self.ds_with_embeddings, lines=True)
            self._parse_and_store(df)
        else:
            if not os.path.exists(self.simple_ds):
                print(f"File {self.simple_ds} missing.")
                return
                
            df = pd.read_json(self.simple_ds, lines=True)
            self._parse_and_store(df)
            self._add_embeddings_and_save()
            
    def _parse_and_store(self, df):
        def parse_stringified_json(val):
            if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return val
            return val
            
        if 'address' in df.columns:
            df['address'] = df['address'].apply(parse_stringified_json)
        if 'primary_naics' in df.columns:
            df['primary_naics'] = df['primary_naics'].apply(parse_stringified_json)
            
        self.records = df.to_dict(orient='records')

    def _add_embeddings_and_save(self):
        print("Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for i, entry in enumerate(self.records):
            desc = entry.get('description', '') or ''
            
            offerings_list = entry.get('core_offerings', [])
            offerings = " ".join(offerings_list)

            
            if offerings.strip():
                entry['core_offerings_embedding'] = model.encode(offerings).tolist()

            if desc.strip():
                entry['description_embedding'] = model.encode(desc).tolist()

        os.makedirs(os.path.dirname(self.ds_with_embeddings), exist_ok=True)
        
        with open(self.ds_with_embeddings, 'w', encoding='utf-8') as f:
            for entry in self.records:
                f.write(json.dumps(entry) + '\n')
                
        print("Done! Embeddings generated and saved successfully.")
