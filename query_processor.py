import json
import numpy as np
import warnings
from thefuzz import fuzz
from sentence_transformers import SentenceTransformer
from google import genai
import json

from query_parser import PostProcessingResponse

warnings.filterwarnings("ignore")


class QueryProcessor:
    def __init__(self):
        pass

    def get_embedding(self, text: str) -> list[float]:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding failed: {e}")
            return []

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        a, b = np.array(vec1), np.array(vec2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def query_filter_companies(self, query_filters_json = {}, records = [], original_query: str = "", test_filter: bool = False):
        try:
            if test_filter:
                with open("data/filters_example.json", "r") as f:
                    filters = json.load(f)
            else:
                filters = json.loads(query_filters_json)
                with open("data/filters_example.json", "w") as f:
                    json.dump(filters, f)

        except Exception as e:
            print(f"JSON error: {e}")
            return []
            
        matches = []

        scoring = filters.get("scoring", {})
        keywords = scoring.get("keywords", []) or []
        core_offerings = scoring.get("core_offerings", []) or []

        search_query = original_query + " " + " ".join(keywords) + " " + " ".join(core_offerings)
        query_embedding = self.get_embedding(search_query)
                
        mandatory_filters = {k: v for k, v in filters.get("mandatory", {}).items() if v is not None and v != []}
        scoring_filters   = {k: v for k, v in scoring.items() if v is not None and v != []}

        count_embeded_match = 0
        for entry in records:

            passed = True
            entry_desc_embedding = entry.pop("description_embedding", None)
            entry_offerings_embedding = entry.pop("core_offerings_embedding", None)

            for key, value in mandatory_filters.items():
                if isinstance(value, list):
                    results = [self.find_match(key, item, entry) for item in value]
                    if not any(r is True for r in results) and any(r is False for r in results):
                        passed = False
                        break
                else:
                    result = self.find_match(key, value, entry)
                    if result is False:
                        passed = False
                        break

            if not passed:
                continue

            print('passed ', entry.get('operational_name'))
            score: float = 0.0
            for key, value in scoring_filters.items():
                if isinstance(value, list):
                    for item in value:
                        if self.find_match(key, item, entry) is True:
                            score += 1
                else:
                    if self.find_match(key, value, entry) is True:
                        score += 1

            if entry_offerings_embedding:
                off_sim = self.cosine_similarity(query_embedding, entry_offerings_embedding)
                if off_sim > 0.2:
                    score += (off_sim * 1.5)
            
            if entry_desc_embedding:
                desc_sim = self.cosine_similarity(query_embedding, entry_desc_embedding)
                if desc_sim > 0.2:
                    score += desc_sim
            
            if score > 1:
                matched_entry = entry.copy()
                matched_entry['match_score'] = round(score, 3)
                matched_entry.pop('description_embedding', None)
                matches.append(matched_entry)

        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches


    def _scalar_match(self, val, search_value, operator):            
        if search_value is None or search_value == "": 
            return False
            
        if operator == ">=" and isinstance(val, (int, float)) and isinstance(search_value, (int, float)):
            return val >= search_value
        elif operator == "<=" and isinstance(val, (int, float)) and isinstance(search_value, (int, float)):
            return val <= search_value
        elif operator == "==":
            if isinstance(val, str) and isinstance(search_value, str):
                if search_value.lower() in val.lower():
                    return True
                return fuzz.token_set_ratio(search_value.lower(), val.lower()) >= 85
            return val == search_value
        return False

    def find_match(self, search_key, search_value, entry):
        base_key = search_key
        operator = "=="
        if search_key.startswith("min_"):
            base_key = search_key[4:]
            operator = ">="
        elif search_key.startswith("max_"):
            base_key = search_key[4:]
            operator = "<="

        found_key = False
        
        if isinstance(entry, dict):
            if base_key in entry:
                found_key = True
                val = entry[base_key]
                if isinstance(val, list):
                    for item in val:
                        if self._scalar_match(item, search_value, operator):
                            return True
                else:
                    if self._scalar_match(val, search_value, operator):
                        return True
            for k, v in entry.items():
                res = self.find_match(search_key, search_value, v)
                if res is True:
                    return True
                if res is False:
                    found_key = True
                    
        elif isinstance(entry, list):
            for item in entry:
                res = self.find_match(search_key, search_value, item)
                if res is True:
                    return True
                if res is False:
                    found_key = True
                    
        return False if found_key else None


class LLMReranker:
    def __init__(self, client: genai.Client):
        self.client = client

    def post_proccessing(self, query: str, matches: list[dict], limit: int = 20):
                
        if not matches:
            return []

        top_matches = matches[:limit]

        prompt = f"""
        You are a data analyst. You have a list of companies that match a query. 
        Your job is to rank them based on their relevance to the query.
        You will receive a list of companies and a query.
        Please reorder based on relevance to the query.
        Response format: {PostProcessingResponse.model_json_schema()}
        Companies: {json.dumps(top_matches, indent=2)}
        Query: {query}
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[PostProcessingResponse],
                },
            )

            ranked_results = json.loads(response.text)
            
            final_ordered_matches = []
            for ranked_item in ranked_results:
                for match in top_matches:
                    match_id = match.get("operational_name")
                    if match_id == ranked_item.get("id"):
                        match['llm_reasoning'] = ranked_item.get("reasoning", "")
                        final_ordered_matches.append(match)
                        break
                        
            return final_ordered_matches
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return top_matches
