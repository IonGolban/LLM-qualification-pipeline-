# High-Performance Hybrid Company Search Engine

This project implements company search engine. It matches user queries against a dataset of companies by combining filtering, semantic embeddings, and LLM-based reranking.

## Setup

```bash
export GEMINI_API_KEY="your_api_key"
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## System Architecture

The pipeline consists of four main stages:
0. **Data Loading**: The dataset is loaded from a JSONL file and preprocessed to extract embeddings for semantic search.
1. **Query Parsing**: The user's query is processed by an LLM (Gemini) to extract strict constraints (mandatory filters like country codes or revenue) and semantic preferences (soft features like keywords and core offerings).
2. **Hybrid Retrieval**: A customized retrieval mechanism applies the extracted strict rules via hard filtering, calculates the cosine similarity of the query against pre-computed sentence embeddings, and scores the records.
3. **LLM Refinement**: The top matched results are reranked using an LLM to confirm relevance.

### Architecture Diagram

![System Architecture Diagram](./data/query-processor-diagram.svg)

## Description of each stage

### 0. Data Loading
In this stage the dataset is loaded from a JSONL, the embeddings are generated for the core_offerings and description fields. To make the processing fater the dataset is preloaded and embeddings are generated. In case of database update in real scenarios we can update separately for new added or updated raws. This is done to make the processing faster and more efficient. 

### 1. Query Parsing
There is schema defined for the query parsing, which is used to extract the filters from the user query. The schema is defined in the `query_parser.py` file. It consist from mandatory filters for hard filtering of result, for example if User asks for a companies that have more than 1000 employees, the LLM will extract the employee count and the operator, and so on for other filters. Scoring filters are used for company classification and ranking and there we have direct comparasion with the company data (using fuzzy matching for strings and exact matching for numbers and codes).

The user query is parsed by an LLM (Gemini) to extract strict constraints (mandatory filters like country codes) and semantic preferences (soft features like keywords and core offerings). The prompt is designed to extract the filters in a JSON format and the restriction are written in order to get the most accurate results, for example if the user asks for "companies in Germany" the LLM will extract the country code for Germany and the country name, also for continent it will extract all the country codes for the countries in that continent. 

**Details:**
- I extractd also some important keywords and their sinonyms from the query to make the search more accurate for embedding search in description and core_offerings. I made this decision in order to compare with embeddings fields and to increase the accuracy of the search, the importance of certain words will be higher.
- Also I observed some anomalies from LLM and infinite keywords associations, for example "AI" it was associating with "Artificial Intelligence" and "AI based solutions" and "AI powered solutions" and so on, which is not what we want. So I added a restriction to the prompt to limit the number of keywords and to ensure that the keywords are relevant to the query. 

### 2. Query Processing
In this stage:
1. Filter by mandatory filters (country codes, revenue, employee count, etc.)
2. Apply scoring filters for non embedding fields (like business_model, target_markets...). I made this decision to make comparasion between strings directly because there are standard values for these fields and it is better to compare them directly than using embedding and this is also more efficient. For example if we have a company with business model "SaaS" and user asks for "SaaS" companies, the score will rank higher and prioritze this company over others with significant score to ensure that over companies with not relevant business models are ranked lower.
3. Apply embedding similarity scoring for description and core_offerings. IMPORTANT DECISION: I deciede to compare directly the embedded concatenation of (input query + keywords (extracted with LLM) + core_offerings) with embedded company description and after that compare with embedded company core_offerings. THis is used to increase the accuracy of the search and to ensure that the most relevant results are ranked higher, because some important keywords might be present in the description and in the same time in the core_offerings, so we will have a higher score for these companies. This was deduced after some tries and comparasion of results.
4. After all the scoring, the results are ranked based on the match score and the sorted results are returned.

**Details:**
- I tried to make the functions as abstract as possible, for example the find_match function and _scalar_match function are used to find the match between the search key and the search value in the entry. In this way we can parse multile type of filters in case of change of schema or database.
- First priority is given to the mandatory filters (not passed if is not matched), then to the scoring filters (1 point for each matched filter) and then to the embedding similarity scoring (1.5x for core_offerings and 1x for description, because of relevance).

`Formula: QueryFilterScore = (MandatoryFilters * infinite) + sum[MatchedScoringFilters(1 point)] + (DescriptionEmbeddingScore * 1) + (CoreOfferingsEmbeddingScore * 1.5)`

### 3. LLM Refinement
Top results are reranked using an LLM to confirm relevance. This is used to prevent some miss-rankings caused by the query processing stage. For example if a company has a very high embedding similarity score but does not match the query filters, the LLM will rerank it lower.

## Approach and Q/A

**What components does it include? How they interact?**
- Find response in the diagram and description of each stage (0 -> 1 -> 2 -> 3).
- Other: Local Embedding Generator (SentenceTransformer all-MiniLM-L6-v2), Local Fuzzy Matching Engine (thefuzz), Local Vector Similarity Engine (numpy), LLM Parser (Gemini 2.5 Flash), LLM Reranker (Gemini 2.5 Flash)

**Design Logic:**
This design was chosen to balance determinism and semantic understanding. LLMs are not efficient or reliable for massive linear filtering, so by chaining structured extraction -> hybrid filters -> embedding similarity -> LLM reranking, the engine filters out obvious noise cheaply and applies expensive reasoning only to the best candidates.

## Tradeoffs

I optimized the engine in order to make it more efficient and accurate. THe speed is also decent, mostly it dependes on LLM response, but for the rest of the operations it is very fast. For example I precomputed the embeddings for the dataset and stored them in memory, so the embedding similarity search is very fast. In case of massive dataset I would use a vector database. The Accuracy is also decent, but it can be improved by using a more powerful LLM (During testing I encountered some issues with LLM responses, for example it was returning empty arrays for some filters, but I managed to fix it by adjusting the prompt), also sometimes the LLM just return a big amount of noisy data.

I sacrifiend some speed over accuracy by sending the top results to rerank with LLM. This is not the most efficient way to do it, but it is the most accurate way and the results are better.

## Error Analysis and Failure Modes

**Where the system struggles:**
The engine struggles with highly nuanced queries, for example a query that combines multiple concepts with "but not" or "except" clauses. It also strugles with such queries "Companies that provide AI solutions for logistic companies", it can return companies that are not related to Software industry.
- To solve these issues I added a reranking stage with LLM, which is used to rerank the top results and to remove the irrelevant ones.

**Confident but Incorrect Results:**
- **Parsing:** The LLM parser might incorrectly convert a soft preference into a mandatory JSON filter, causing the engine to return 0 matches. This is rare but it can happen. Also the LLM can return a big amount of noisy data that is not relevant to the query.
- **Database:** Some companies can contain in their description or core_offerings the keywords extracted from the query, but they are not really relevant to the query.

## Scaling

**If the system needed to handle 100,000 companies per query instead of 500:**
- A vector database to store the embeddings and to perform the similarity search. Also I can use a more efficient way to filter the mandatory and scoring filters.
- A more powerful LLM can be used to parse the query and to extract the filters.
- Using SQL/NOSQL database with proper indexing can be done on the mandatory and scoring filters to make the filtering more efficient.

**What I've done:**
- Abstracted some functions to be able to use it with different filters, but still it is about abstraction of the filtering logic, not really about scaling. 
- Precomputed the embeddings for the dataset and stored them in memory, so the embedding similarity search is faster.

**What to monitor in production:**
Zero match companies, LLM latency and error rates, token usage and cost.
