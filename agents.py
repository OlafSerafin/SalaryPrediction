#%% md
# #Load env
#%%
import os 
from dotenv import load_dotenv
#%%
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print("Błąd: Nie znaleziono klucza API w pliku .env")
else: 
    print("Klucz API załadowany pomyślnie")


os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'False'

#%% md
# #Loading the model
# 
#%%
from keras.models import load_model

#%%
model_path='model.keras'
model=load_model(model_path)
#%% md
# #New Version
#%% md
# ##Session
# 
#%%
from google import genai
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import os



async def call_agent(query:str, runner, user_id,session_id):
  content = types.Content(role='user',parts=[types.Part(text=query)])
  async for event in runner.run_async(user_id=user_id,session_id=session_id, new_message=content):
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
            final_response_text=part.text
            found_response=True
            break
      else:
        final_response_text=f"Agent escalated: {event.error_message or 'No specific error message'}"
        found_response=True
      if found_response:
        break
  return final_response_text
#%% md
# ##Data Extracion
#%%
import os
from google import genai
from pydantic import BaseModel, Field
from typing import Optional

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

standard_job_titles = [
    "Data Scientist",
    "Data Engineer",
    "Data Analyst",
    "Other",
    "Clinical & Safety",
    "Pharma & Engineering",
    "Analytics Management & BI",
    "Machine Learning Engineer",
    "Research",
    "Medical",
    "Data Architecture",
    "Advanced Tech "]

sector_list=['Aerospace & Defense', 'Health Care', 'Business Services',
       'Oil, Gas, Energy & Utilities', 'Real Estate', 'Finance',
       'Information Technology', 'Retail', 'Biotech & Pharmaceuticals',
       'Media', 'Insurance', 'Transportation & Logistics',
       'Telecommunications', 'Manufacturing', 'Mining & Metals',
       'Government', 'Education', 'Agriculture & Forestry',
       'Travel & Tourism', 'Non-Profit',
       'Arts, Entertainment & Recreation',
       'Construction, Repair & Maintenance', 'Accounting & Legal',
       'Consumer Services']
# 1. Definicja Struktury Danych (Schema)
class JobDetailsSchema(BaseModel):

    job_state: str = Field(description="Two-letter US state abbreviation (e.g., NY, CA) or 'Unknown'.")

    job_title: str = Field(description="The normalized job title (e.g., Data Scientist, Backend Engineer).")

    Senior_value: int = Field(description="1 if the job is for senior, else 0.")

    company_name: str = Field(description="The name of the company if mentioned, else 'Unknown'.")

    Size: str = Field(description="Company size/number of employees if mentioned, else 'Unknown'.")

    Sector: str = Field(description="Industry sector (e.g., Finance, Health, Tech) if mentioned, else 'Unknown'.")

    Competitors: int = Field(description="1 if any competitors exist, else 'Unknown'.")

    job_description: int = Field(description="Length of the job description.")

    # Umiejętności (0 lub 1)

    python_yn: int = Field(description="1 if Python is required/nice-to-have, else 0.")

    spark: int = Field(description="1 if Spark/PySpark is mentioned, else 0.")

    aws: int = Field(description="1 if AWS/Cloud is mentioned, else 0.")

    excel: int = Field(description="1 if Excel is mentioned, else 0.")

# 2. Ekstrakcja
def extract_job_details(job_description: str) -> dict:
    """
    Uses AI to analyze a job description text and extract structured data
    (skills, location, company info) for salary prediction.
    """
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
        config=
         {
            'response_mime_type': 'application/json',
            'response_schema': JobDetailsSchema,
        },
    )

    # Zwracamy słownik (dict)
    return response.parsed.model_dump()
#%% md
# ##Functions to find missing info
#%%
import os
from google import genai
from google.genai.types import Tool, GenerateContentConfig
from pydantic import BaseModel, Field
from typing import Optional

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Define a schema for the company information to be retrieved by search
class CompanyInfoSchema(BaseModel):
    Size: str = Field(description="Company size/number of employees return numerical value, else 'Unknown'.")
    Sector: str = Field(description="Industry sector, else 'Unknown'.")
    Competitors: int = Field(description="1 if any competitors exist, else '0'.")


#%%
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

def get_company_info(company_name: str, missing_info_keys: list[str]) -> dict:


    """
    Robust version: Searches first, then formats to JSON to avoid 'Unknown' loops.
    """
    missing_info_str = ", ".join(missing_info_keys)
    print(f"Searching for info on: {company_name} for missing fields: {missing_info_keys}...")

    # KROK 1: Wyszukiwanie (tryb tekstowy - pozwala modelowi "pomyśleć" i użyć źródeł)
    # Nie wymuszamy tu JSONa, dajemy modelowi swobodę skorzystania z Google.
    search_prompt = f"""
    Search specifically for the following details about the company '{company_name}': {missing_info_str}.
    Provide the answer in a detailed text format. If you find numbers (competitors, employees), include them clearly.
    If you need to find size of the company return the number of employees it must be numerical value.
    """

    sector_list=['Aerospace & Defense', 'Health Care', 'Business Services',
       'Oil, Gas, Energy & Utilities', 'Real Estate', 'Finance',
       'Information Technology', 'Retail', 'Biotech & Pharmaceuticals',
       'Media', 'Insurance', 'Transportation & Logistics',
       'Telecommunications', 'Manufacturing', 'Mining & Metals',
       'Government', 'Education', 'Agriculture & Forestry',
       'Travel & Tourism', 'Non-Profit',
       'Arts, Entertainment & Recreation',
       'Construction, Repair & Maintenance', 'Accounting & Legal',
       'Consumer Services']

    try:
        # Zapytanie z włączonym Google Search
        raw_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=search_prompt,
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
                response_modalities=["TEXT"]
            )
        )

        found_text = raw_response.text
        # Opcjonalnie: Wyświetl źródła dla debugowania
        if raw_response.candidates[0].grounding_metadata and raw_response.candidates[0].grounding_metadata.search_entry_point:
             print(f"   [Grounding] Sources found.")

    except Exception as e:
        print(f"Search step failed: {e}")
        return {key: "Unknown" for key in missing_info_keys}

    # KROK 2: Ekstrakcja do JSON (czysty model bez narzędzi)
    # Teraz bierzemy "brudny" tekst z wynikami i prosimy model o wpisanie go w schemat.
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
        print(f"--- RETRIEVED: {company_details} ---")
        return company_details

    except Exception as e:
        print(f"Parsing step failed: {e}")
        return {key: "Unknown" for key in missing_info_keys}
#%%
def get_size_range(num_employees):
    """
    Maps a numeric value to a specific employee count label
    based on predefined thresholds.
    """
    # Handle invalid data (e.g., None, strings, or values <= 0)
    num_employees=num_employees.replace(',','.')
    print(num_employees)
    try:
        num_employees = int(num_employees)
    except (ValueError, TypeError):
        return 'Unknown'
    if num_employees <= 0:
        return 'Unknown'

    # Check thresholds in ascending order
    if num_employees <= 50:
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
#%%
import pandas as pd

def calculate_salary_prediction(job_details):
    """
    Funkcja pomocnicza wykonująca predykcję na podstawie słownika job_details.
    Korzysta z globalnych obiektów: model, scaler_x, scaler_y, df_og.
    """
    try:
        # Tworzenie DataFrame z jednego wiersza
        df_dummies = create_df_from_data(job_details, df_og)

        if df_dummies is None:
            return "Error in data processing"

        scaled = scaler_x.transform(df_dummies)

        # Zmieniono: Dodano 'excel' do listy filter_cols, aby pasowała do oczekiwań modelu
        filter_cols = df_dummies.columns.isin(['python_yn', 'spark', 'aws', 'excel'])

        scaled_1 = scaled[:, ~filter_cols] # Reszta cech
        scaled_2 = scaled[:, filter_cols]  # Cechy binarne/tech

        # Predykcja
        prediction_scaled = model.predict([scaled_1, scaled_2], verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)

        return round(float(prediction[0][0]), 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return "N/A"
#%%
def process_job_offer(job_description_text: str):
    """
    Główna funkcja orkiestrująca proces:
    Ekstrakcja -> Weryfikacja braków -> Wyszukiwanie -> Scalanie -> Predykcja
    """

    # 1. KROK PIERWSZY: Ekstrakcja danych z ogłoszenia
    print("--- 1. Analiza ogłoszenia ---")
    job_details = extract_job_details(job_description_text)

    # Sprawdzamy, czy mamy nazwę firmy
    company_name = job_details.get('company_name')

    # 2. KROK DRUGI i TRZECI: Uzupełnianie danych z Google (jeśli są braki i jest nazwa firmy)
    if not company_name or company_name == 'Unknown':
        print("Nie znaleziono nazwy firmy. Pomijam wyszukiwanie.")
    else:
        enrichable_fields = ['Size', 'sector', 'Competitors']
        missing_keys = [
            key for key in enrichable_fields
            if job_details.get(key) == 'Unknown'
        ]

        if missing_keys:
            print(f"--- 2. Wykryto braki: {missing_keys}. Wyszukuję dane dla {company_name}... ---")
            external_info = get_company_info(company_name, missing_keys)

            # 4. KROK CZWARTY: Scalanie danych
            for key, value in external_info.items():
                if value != 'Unknown' and key in job_details:
                    print(f"   -> Aktualizacja {key}: {value}")
                    job_details[key] = value
                else:
                    print(f"   -> Nie udało się znaleźć {key}")
        else:
            print("Komplet danych o firmie w ogłoszeniu.")

    # Formatowanie danych (wymagane przed wrzuceniem do modelu)
    job_details['job_description'] = len(job_description_text)
    # Konwersja Size na przedział (np. '1 to 50 employees') - ważne dla create_df_from_data
    job_details['Size'] = get_size_range(job_details['Size'])
    job_details['Sector'] = job_details['Sector'].title()
    job_details['job_state'] = " " + job_details['job_state']

    # --- NOWY KROK: Predykcja pensji ---
    print("--- 3. Obliczanie predykcji pensji (AI Model) ---")
    salary = calculate_salary_prediction(job_details)
    job_details['predicted_yearly_salary'] = salary
    print(f"   -> Przewidziana pensja: {salary}k $")

    return job_details
#%% md
# ## Creating new DataFrame
#%%
import pandas as pd
#%%
def create_df_from_data(data,df_og):
  try:
    df=pd.DataFrame([data])

    df.drop(columns=['company_name'], inplace=True)
    df.rename(columns={'job_description': 'Job Description',
                       'job_title': 'Job Title'}
              , inplace=True)
    df_dummies=pd.get_dummies(df).replace({True: 1, False: 0})

    print(df_dummies.info()) # Opcjonalnie

        # --- NOWA CZĘŚĆ KODU: Wyświetlanie nadmiarowych kolumn ---
        # Obliczamy różnicę zbiorów: Kolumny w nowym - Kolumny w starym
    extra_cols = set(df_dummies.columns) - set(df_og.columns)

    if extra_cols:
      print(f"\n⚠️ UWAGA: Znaleziono {len(extra_cols)} nowych kolumn, których nie ma w oryginalnym modelu:")
      print(extra_cols)
    else: print("\nBrak nowych, nieznanych kolumn.")
        # ---------------------------------------------------------

    missing_cols = set(df_og.columns) - set(df_dummies.columns)
    for c in missing_cols:
      df_dummies[c] = 0
    df_dummies= df_dummies[df_og.columns]

    return df_dummies

  except Exception as e:
    print(e)
    return None

#%%
import pandas as pd

def create_df_from_data(data, df_og):
  try:
    df = pd.DataFrame([data])

    # Używamy errors='ignore', żeby nie wywaliło błędu, jak kolumny nie będzie
    df.drop(columns=['company_name'], inplace=True, errors='ignore')

    # Rename columns to match df_og's structure
    df.rename(columns={'job_description': 'Job Description',
                       'job_title': ' Job Title'},
              inplace=True)

    df_dummies = pd.get_dummies(df,dtype=int)

    # Dopasowuje kolumny do wzorca (df_og), braki wypełnia zerami (fill_value=0).
    df_dummies = df_dummies.reindex(columns=df_og.columns, fill_value=0)
    # --------------

    return df_dummies

  except Exception as e:
    print(f"Błąd w create_df_from_data: {e}")
    return None
#%% md
# ##Loading og df
# 
#%%
def load_og_df(path): #use this once at the start of the program
  df_og=pd.read_csv(path)
  df_og.drop(columns=['avg_salary'], inplace=True)
  return df_og
#%%
df_og=load_og_df('dataset.csv')
#%% md
# ###Loading scalers
#%%
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler_x = joblib.load('scaler_x.gz')
scaler_y = joblib.load('scaler_y.gz')

#%% md
# ##Agentic System
#%%
from google.adk.agents import LlmAgent


job_offer_agent = LlmAgent(
    model="gemini-2.5-pro",
    name="job_offer_agent",
    description="Specialist agent that processes job descriptions and predicts salary.", # Zmiana opisu

    tools=[process_job_offer],

    instruction="""
    You are an expert HR Analyst Agent.

    YOUR GOAL:
    To provide a comprehensive summary of a job offer, including company details AND estimated salary.

    PROTOCOL:
    1. When you receive a job description text, call the tool `process_job_offer`.
    2. The tool will return a structured JSON object containing extracted data AND a 'predicted_yearly_salary' field.

    3. Present this data to the user in a professional format:
       - **Start with the Estimated Salary**: Highlight the 'predicted_yearly_salary' clearly at the top (add currency symbol $ if missing).
       - Use Markdown bullet points for the rest.
       - Group information logically (Job Details, Company Profile, Tech Stack).
       - If a field is 'Unknown', output "Not specified".
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
    """
)
#%%
import asyncio


# Funkcja, która przygotuje runnera
async def setup_system():
    session_service = InMemorySessionService()
    APP_NAME = "salary_predicition_app"
    USER_ID = "user_1"
    SESSION_ID = "session_1"

    # Tutaj używamy await legalnie, bo jesteśmy w async def
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(
        agent=coordinator_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    return runner, USER_ID, SESSION_ID

