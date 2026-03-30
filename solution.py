import os
import json
from google import genai

from data_repository import DataRepository
from query_parser import QueryParser
from query_processor import QueryProcessor, LLMReranker


class CompanySearchEngine:
    def __init__(self, data_path: str):
        self.check_api_key()
        self.client = genai.Client()

        self.repo = DataRepository(data_path)
        self.repo.load_data()
        self.records = self.repo.records

        self.parser = QueryParser(client=self.client)
        self.retriever = QueryProcessors()
        self.reranker = LLMReranker(client=self.client)

    def check_api_key(self):
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("API Key not found")

    def process_query(self, query):
        return self.parser.process_query(query, self.records[0])

    def query_filter_companies(self, query_filters_json, original_query: str = "", test_filter: bool = False):
        return self.retriever.query_filter_companies(query_filters_json, self.records, original_query, test_filter)

    def post_proccessing(self, query: str, matches: list[dict]):
        return self.reranker.post_proccessing(query, matches)

def get_results_for_query(query: str):
    
    engine = CompanySearchEngine("data/companies.jsonl")
    
    print(f"\nSearching for: {query}")
    
    json_filters = engine.process_query(query)
    print(f"Filters:\n{json_filters}")
    
    matches = engine.query_filter_companies(json_filters, query, test_filter=False)
    print(f"\nFound {len(matches)} matches.")

    post_processed_matches = engine.post_proccessing(query, matches)
    
    
    for i, match in enumerate(matches[:10]):
        print(f"#{i+1} [Score: {match['match_score']}] - {match.get('operational_name')}")
    print("\n" + "-"*30 + "\nRefined Results\n" + "-"*30)
    for i, match in enumerate(post_processed_matches[:10]):
        print(f"#{i+1} - {match.get('operational_name')}")
        print(f"Why: {match.get('llm_reasoning', 'Not generated')}\n")

    return post_processed_matches


if __name__ == "__main__":
    queries = []

    with open("data/queries.txt", "r") as f:
        for line in f:
            queries.append(line.strip())
    for query in queries:
        result = get_results_for_query(query)

        with open("data/results.jsonl", "a") as f:
            json.dump(result, f, indent=4)
            f.write("\n")
    

    
        