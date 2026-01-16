

from keras.models import load_model
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.agents import LlmAgent
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import pandas as pd
import joblib
import asyncio
import os 
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("Błąd: Nie znaleziono klucza API w pliku .env")
else: 
    print("Klucz API załadowany pomyślnie")


os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'False'

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Model paths (adjust as needed)
model_path = '/content/model.keras'
model = load_model(model_path)
scaler_x = joblib.load('scaler_x.gz')
scaler_y = joblib.load('scaler_y.gz')
df_og = pd.read_csv('/content/dataset.csv')
df_og.drop(columns=['avg_salary'], inplace=True)

# Standard lists
standard_job_titles = [
    "Data Scientist", "Data Engineer", "Data Analyst", "Other",
    "Clinical & Safety", "Pharma & Engineering", "Analytics Management & BI",
    "Machine Learning Engineer", "Research", "Medical", "Data Architecture",
    "Advanced Tech"
]

sector_list = [
    'Aerospace & Defense', 'Health Care', 'Business Services',
    'Oil, Gas, Energy & Utilities', 'Real Estate', 'Finance',
    'Information Technology', 'Retail', 'Biotech & Pharmaceuticals',
    'Media', 'Insurance', 'Transportation & Logistics',
    'Telecommunications', 'Manufacturing', 'Mining & Metals',
    'Government', 'Education', 'Agriculture & Forestry',
    'Travel & Tourism', 'Non-Profit',
    'Arts, Entertainment & Recreation',
    'Construction, Repair & Maintenance', 'Accounting & Legal',
    'Consumer Services'
]

# Schemas
class JobDetailsSchema(BaseModel):
    job_state: str = Field(description="Two-letter US state abbreviation (e.g., NY, CA) or 'Unknown'.")
    job_title: str = Field(description="The normalized job title (e.g., Data Scientist, Backend Engineer).")
    Senior_value: int = Field(description="1 if the job is for senior, else 0.")
    company_name: str = Field(description="The name of the company if mentioned, else 'Unknown'.")
    Size: str = Field(description="Company size/number of employees if mentioned, else 'Unknown'.")
    Sector: str = Field(description="Industry sector (e.g., Finance, Health, Tech) if mentioned, else 'Unknown'.")
    Competitors: int = Field(description="1 if any competitors exist, else 'Unknown'.")
    job_description: int = Field(description="Length of the job description.")
    python_yn: int = Field(description="1 if Python is required/nice-to-have, else 0.")
    spark: int = Field(description="1 if Spark/PySpark is mentioned, else 0.")
    aws: int = Field(description="1 if AWS/Cloud is mentioned, else 0.")
    excel: int = Field(description="1 if Excel is mentioned, else 0.")

class CompanyInfoSchema(BaseModel):
    Size: str = Field(description="Company size/number of employees return numerical value, else 'Unknown'.")
    Sector: str = Field(description="Industry sector, else 'Unknown'.")
    Competitors: int = Field(description="1 if any competitors exist, else '0'.")

# === MODIFIED: Result schema that includes sources ===
class JobAnalysisResult(BaseModel):
    """Complete job analysis result with sources"""
    job_details: dict = Field(description="Extracted and enriched job details")
    sources: List[str] = Field(default_factory=list, description="URLs used to enrich the data")
    predicted_yearly_salary: float = Field(description="Predicted salary in thousands")

async def call_agent(query: str, runner, user_id, session_id):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.content:
            for i, part in enumerate(event.content.parts):
                if part.text:
                    print(f"Part {i}: {part.text}")
                if part.function_call:
                    print(f"Part {i} Function call: {part.function_call.name}({part.function_call.args})")
        if event.is_final_response():
            found_response = False
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_response_text = part.text
                        found_response = True
                        break
            else:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific error message'}"
                found_response = True
            if found_response:
                break
    return final_response_text

def extract_job_details(job_description: str) -> dict:
    """Extract job details from description"""
    prompt = f"""
    Analyze the following job description text and extract the details exactly according to the schema.
    For missing information, strictly use 'Unknown'. Make sure the state is mentioned, don't return any state if you are no certain.
    Try to extract the company name from the beginning of the job description if possible.

    When extracting the 'job_title', please normalize it to one of the following standard titles if a clear match exists:
    {', '.join(standard_job_titles)}. If a perfect match isn't found, return "Other".
    When extracting the 'sector', please normalize it to one of the following standard sectors if a clear match exists:
    {', '.join(sector_list)}. It is crucial to choose one of the standard sectors. If a perfect match isn't found, return "Unknown".

    JOB DESCRIPTION:
    {job_description}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': JobDetailsSchema,
        },
    )

    return response.parsed.model_dump()

def get_company_info(company_name: str, missing_info_keys: list[str]) -> dict:
    """
    Searches for company info and returns data WITH sources.
    Returns dict with company details + 'sources' key containing list of URLs.
    """
    missing_info_str = ", ".join(missing_info_keys)
    print(f"Searching for info on: {company_name} for missing fields: {missing_info_keys}...")

    search_prompt = f"""
    Search specifically for the following details about the company '{company_name}': {missing_info_str}.
    Provide the answer in a detailed text format. If you find numbers (competitors, employees), include them clearly.
    If you need to find size of the company return the number of employees it must be numerical value.
    """

    found_text = ""
    top_sources = []

    try:
        raw_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=search_prompt,
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
                response_modalities=["TEXT"]
            )
        )

        found_text = raw_response.text
        candidate = raw_response.candidates[0]
        meta = candidate.grounding_metadata

        if meta and meta.grounding_chunks and meta.grounding_supports:
            chunk_map = {
                i: chunk.web.uri
                for i, chunk in enumerate(meta.grounding_chunks)
                if chunk.web and chunk.web.uri
            }

            used_links = []
            for support in meta.grounding_supports:
                for idx in support.grounding_chunk_indices:
                    if idx in chunk_map:
                        used_links.append(chunk_map[idx])

            top_sources = list(dict.fromkeys(used_links))
            print(f"   [Grounding] Found {len(top_sources)} relevant sources used in text.")

    except Exception as e:
        print(f"Search step failed: {e}")
        return {key: "Unknown" for key in missing_info_keys} | {'sources': []}

    parsing_prompt = f"""
    Based ONLY on the text below, extract the company details. For 'Size' you must return numerical value.
    IMPORTANT for 'Sector': You MUST match the sector exactly to one of the strings in this list:
    {sector_list}
    Do not invent new categories. If the exact phrase isn't found, pick the conceptually closest one from the list.

    SOURCE TEXT:
    {found_text}

    Extract fields: {missing_info_str}.
    """

    try:
        json_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=parsing_prompt,
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=CompanyInfoSchema,
            )
        )

        company_details = json_response.parsed.model_dump()
        company_details['sources'] = top_sources

        print(f"--- RETRIEVED: {company_details} ---")
        return company_details

    except Exception as e:
        print(f"Parsing step failed: {e}")
        return {key: "Unknown" for key in missing_info_keys} | {'sources': []}

def get_size_range(num_employees):
    """Maps numeric value to employee count label"""
    num_employees = str(num_employees).replace(',', '.')
    try:
        num_employees = int(float(num_employees))
    except (ValueError, TypeError):
        return 'Unknown'
    
    if num_employees <= 0:
        return 'Unknown'
    elif num_employees <= 50:
        return '1 to 50 employees'
    elif num_employees <= 200:
        return '51 to 200 employees'
    elif num_employees <= 500:
        return '201 to 500 employees'
    elif num_employees <= 1000:
        return '501 to 1000 employees'
    elif num_employees <= 5000:
        return '1001 to 5000 employees'
    elif num_employees <= 10000:
        return '5001 to 10000 employees'
    else:
        return '10000+ employees'

def create_df_from_data(data, df_og):
    """Create DataFrame from job details"""
    try:
        df = pd.DataFrame([data])
        df.drop(columns=['company_name'], inplace=True, errors='ignore')
        df.rename(columns={
            'job_description': 'Job Description',
            'job_title': ' Job Title'
        }, inplace=True)

        df_dummies = pd.get_dummies(df, dtype=int)
        df_dummies = df_dummies.reindex(columns=df_og.columns, fill_value=0)
        
        return df_dummies
    except Exception as e:
        print(f"Error in create_df_from_data: {e}")
        return None

def calculate_salary_prediction(job_details):
    """Calculate salary prediction"""
    try:
        df_dummies = create_df_from_data(job_details, df_og)
        if df_dummies is None:
            return "Error in data processing"

        scaled = scaler_x.transform(df_dummies)
        filter_cols = df_dummies.columns.isin(['python_yn', 'spark', 'aws', 'excel'])
        scaled_1 = scaled[:, ~filter_cols]
        scaled_2 = scaled[:, filter_cols]

        prediction_scaled = model.predict([scaled_1, scaled_2], verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        return round(float(prediction[0][0]), 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "N/A"

# === MODIFIED: Now returns structured result with sources ===
def process_job_offer(job_description_text: str) -> dict:
    """
    Main orchestration function that returns job details AND sources.
    Returns a dictionary with 'job_details', 'sources', and 'predicted_yearly_salary'.
    """
    print("--- 1. Analyzing job description ---")
    job_details = extract_job_details(job_description_text)
    
    # Track all sources from external searches
    all_sources = []
    
    company_name = job_details.get('company_name')

    if not company_name or company_name == 'Unknown':
        print("No company name found. Skipping search.")
    else:
        enrichable_fields = ['Size', 'Sector', 'Competitors']
        missing_keys = [
            key for key in enrichable_fields
            if job_details.get(key) == 'Unknown'
        ]

        if missing_keys:
            print(f"--- 2. Detected missing fields: {missing_keys}. Searching for {company_name}... ---")
            external_info = get_company_info(company_name, missing_keys)
            
            # === MODIFIED: Extract and store sources ===
            if 'sources' in external_info:
                all_sources.extend(external_info['sources'])
                del external_info['sources']  # Remove from dict before merging
            
            for key, value in external_info.items():
                if value != 'Unknown' and key in job_details:
                    print(f"   -> Updating {key}: {value}")
                    job_details[key] = value
                else:
                    print(f"   -> Could not find {key}")
        else:
            print("Complete company data in job description.")

    # Format data
    job_details['job_description'] = len(job_description_text)
    job_details['Size'] = get_size_range(job_details['Size'])
    job_details['Sector'] = job_details['Sector'].title()
    job_details['job_state'] = " " + job_details['job_state']

    print("--- 3. Calculating salary prediction ---")
    salary = calculate_salary_prediction(job_details)
    
    # === MODIFIED: Return structured result with sources ===
    result = {
        'job_details': job_details,
        'sources': all_sources,
        'predicted_yearly_salary': salary
    }
    
    print(f"   -> Predicted salary: {salary}k $")
    print(f"   -> Sources used: {len(all_sources)}")
    
    return result

# === MODIFIED: Updated agent instructions to handle sources ===
job_offer_agent = LlmAgent(
    model="gemini-2.5-pro",
    name="job_offer_agent",
    description="Specialist agent that processes job descriptions and predicts salary with source tracking.",
    tools=[process_job_offer],
    instruction="""
    You are an expert HR Analyst Agent with source tracking capabilities.

    YOUR GOAL:
    To provide a comprehensive summary of a job offer, including company details, estimated salary, AND the sources used.

    PROTOCOL:
    1. When you receive a job description text, call the tool `process_job_offer`.
    2. The tool will return a structured JSON object containing:
       - 'job_details': extracted job information
       - 'sources': list of URLs used to enrich the data
       - 'predicted_yearly_salary': estimated salary

    3. Present this data to the user in a professional format:
       - **Start with the Estimated Salary**: Highlight the 'predicted_yearly_salary' clearly at the top (add currency symbol $ if missing).
       - Use Markdown bullet points for job details.
       - Group information logically (Job Details, Company Profile, Tech Stack).
       - **IMPORTANT**: If sources are available, include a "Sources" section at the end with clickable links.
       - If a field is 'Unknown', output "Not specified".
       
    4. Example format for sources section:
       
       **Sources Used:**
       - [Source 1](url1)
       - [Source 2](url2)
    """
)

coordinator_agent = LlmAgent(
    name="coordinator_agent",
    model="gemini-2.5-flash",
    description="Main interface agent.",
    sub_agents=[job_offer_agent],
    instruction="""
    You are the Coordinator.

    1. If the user sends a job description, route it to `job_offer_agent`.
    2. If the user says "Hello", greet them and ask for a job description to analyze.
    3. Ensure that any sources provided by sub-agents are passed through to the final response.
    """
)

# Session setup
async def setup_session():
    session_service = InMemorySessionService()
    APP_NAME = "salary_prediction_app"
    USER_ID = "user_1"
    SESSION_ID = "session_1"

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    print("Session created")

    runner = Runner(
        agent=coordinator_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    print("Runner created")
    
    return runner, USER_ID, SESSION_ID

# Test prompt
async def test_agent():
    runner, USER_ID, SESSION_ID = await setup_session()
    
    new_user_prompt = """
    Google Company is
    Hiring: Senior Data Scientist in New York, NY.
    We are a large corporation.
    Our sector is Information Technology
    Requirements: Python, AWS, and strong Excel skills.
    Nice to have: experience with Spark.
    """
    
    final_response = await call_agent(new_user_prompt, runner, USER_ID, SESSION_ID)
    print("\n=== FINAL RESPONSE ===")
    print(final_response)

# Run test
# asyncio.run(test_agent())