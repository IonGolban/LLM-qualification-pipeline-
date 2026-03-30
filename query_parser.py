import os
import json
from pydantic import BaseModel, Field
from google import genai


class MandatoryFilters(BaseModel):
    is_public: bool | None = Field(default=None)

    country_code: str | list[str] | None = Field(default=None)
    country_name: str | list[str] | None = Field(default=None)
    county: str | list[str] | None = Field(default=None)
    min_latitude: float | None = Field(default=None)
    max_latitude: float | None = Field(default=None)
    min_longitude: float | None = Field(default=None)
    max_longitude: float | None = Field(default=None)
    postcode: str | list[str] | None = Field(default=None)
    region_code: str | list[str] | None = Field(default=None)
    region_name: str | list[str] | None = Field(default=None)
    town: str | list[str] | None = Field(default=None)

    min_employee_count: float | None = Field(default=None)
    max_employee_count: float | None = Field(default=None)
    min_revenue: float | None = Field(default=None)
    max_revenue: float | None = Field(default=None)
    min_year_founded: float | None = Field(default=None)
    max_year_founded: float | None = Field(default=None)


class ScoringFilters(BaseModel):
    website: str | None = Field(default=None)
    operational_name: str | None = Field(default=None)
    house_number: str | None = Field(default=None)
    raw_text: str | None = Field(default=None)
    street: str | None = Field(default=None)
    suburb: str | None = Field(default=None)
    code: str | None = Field(default=None)
    label: str | None = Field(default=None)
    min_share: float | None = Field(default=None)
    max_share: float | None = Field(default=None)
    description: str | None = Field(default=None)
    business_model: list[str] | None = Field(default=None)
    target_markets: list[str] | None = Field(default=None)
    core_offerings: list[str] | None = Field(default=None)
    keywords: list[str] | None = Field(default=None)


class QueryFilters(BaseModel):
    mandatory: MandatoryFilters = Field(default_factory=MandatoryFilters)
    scoring: ScoringFilters = Field(default_factory=ScoringFilters)


class PostProcessingResponse(BaseModel):
    id: str
    reasoning: str

class QueryParser:
    def __init__(self, client: genai.Client):
        self.client = client

    def process_query(self, query: str, example_row: dict):
        row_to_send = example_row.copy()
        row_to_send.pop("description_embedding")
        row_to_send.pop("core_offerings_embedding")

        prompt = f"""
        You are an expert search query parser. Your goal is to translate a user's natural language search into a strictly formatted, concise JSON object to query a company database.

        CRITICAL DIRECTIVE: Maximize signal, minimize noise. Output concise arrays.

        OUTPUT STRUCTURE:
        - "mandatory": Strict filters (country_code, min_employee_count, min_revenue, is_public, etc.).
        - "scoring": Semantic preferences (business_model, target_markets, core_offerings, keywords, code, etc.).

        EXTRACTION RULES:
        1. GEOGRAPHY: Resolve regions/continents to an array of 2-letter ISO country codes in `mandatory.country_code`. Never output a continent name as is (iterate over countries in that continent).
        2. NAICS: Infer the most likely 6-digit NAICS code for the main industry and put them in `scoring.code`, the label in `scoring.label`.
        3. CONCEPT ROUTING:
           - Who they sell to -> `scoring.target_markets`
           - What they sell -> `scoring.core_offerings`
           - How they operate -> `scoring.business_model` (Expand B2B -> Business-to-Business, SaaS -> Software as a Service, D2C -> Direct-to-Consumer, etc.) Do not put there if there is ambiguity.
        4. SYNONYMS (STRICT LIMIT): If the industry is broad, generate EXACTLY 3 to 7 high-level industry synonyms or action verbs in `scoring.keywords`.
           - DO NOT generate job titles, personnel roles, or internal company positions if it is not explicitly stated in the query.
           - DO NOT repeat the base keyword in endless variations.

        Example of row in database: {example_row}
        User Query: {query}
        """
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': QueryFilters,
            },
        )
        return response.text
