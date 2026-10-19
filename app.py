import os
import io
import re
import json
import zipfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from langfuse import Langfuse

load_dotenv()

st.set_page_config(page_title="Synthetic Data & Talk to Data", layout="wide")

# --- 1. Configuration & Connections ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gd-gcp-gridu-genai")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MODEL_ID = "gemini-2.5-flash"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "mysecretpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "genai_db")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_resource
def get_db_engine():
    return create_engine(DATABASE_URL)

@st.cache_resource
def get_genai_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def get_langfuse_client():
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if pub_key and sec_key:
        return Langfuse(public_key=pub_key, secret_key=sec_key, host=host)
    return None

client = get_genai_client()
engine = get_db_engine()
langfuse = get_langfuse_client()

# --- 2. Session State Initialization ---
if "generated_tables" not in st.session_state:
    st.session_state.generated_tables = {}
if "table_order" not in st.session_state:
    st.session_state.table_order = []
if "ddl_schema" not in st.session_state:
    st.session_state.ddl_schema = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- 3. Helper Functions ---
def call_gemini_json(prompt: str, temperature: float = 0.2, trace_name: str = "gemini_call") -> dict:
    """Calls Gemini with JSON response mime type and logs to Langfuse (compatible with v2 & v3+)."""
    if langfuse and hasattr(langfuse, "start_as_current_observation"):
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=trace_name,
            model=MODEL_ID,
            input=prompt,
            model_parameters={"temperature": temperature}
        ) as generation:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json"
                )
            )
            generation.update(output=response.text)
        langfuse.flush()
    elif langfuse and hasattr(langfuse, "trace"):
        trace = langfuse.trace(name=trace_name)
        generation = trace.generation(
            name=trace_name,
            model=MODEL_ID,
            input=prompt,
            model_parameters={"temperature": temperature}
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )
        generation.end(output=response.text)
        langfuse.flush()
    else:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
        )

    return json.loads(response.text)

def sanitize_ddl_for_postgres(ddl_sql: str) -> str:
    """Converts common MySQL/SQLServer DDL syntax (AUTO_INCREMENT, inline ENUM, etc.) into valid PostgreSQL DDL."""
    # 1. Replace Unicode em-dash/en-dash comments (— or –) with standard SQL comments (--)
    ddl = re.sub(r'^[\s]*[—–]+', '--', ddl_sql, flags=re.MULTILINE)
    # 2. Remove MySQL backticks
    ddl = ddl.replace('`', '')
    # 3. Convert INT PRIMARY KEY AUTO_INCREMENT or INT AUTO_INCREMENT PRIMARY KEY -> SERIAL PRIMARY KEY
    ddl = re.sub(
        r'\b(INT|INTEGER|BIGINT|SMALLINT)\s+(?:UNSIGNED\s+)?PRIMARY\s+KEY\s+AUTO_INCREMENT\b',
        'SERIAL PRIMARY KEY',
        ddl,
        flags=re.IGNORECASE
    )
    ddl = re.sub(
        r'\b(INT|INTEGER|BIGINT|SMALLINT)\s+(?:UNSIGNED\s+)?AUTO_INCREMENT\s+PRIMARY\s+KEY\b',
        'SERIAL PRIMARY KEY',
        ddl,
        flags=re.IGNORECASE
    )
    ddl = re.sub(r'\bAUTO_INCREMENT\b', '', ddl, flags=re.IGNORECASE)
    # 4. Convert inline MySQL ENUM('a', 'b', ...) or SET('a', 'b', ...) -> VARCHAR(100)
    ddl = re.sub(r'\b(ENUM|SET)\s*\([^)]*\)', 'VARCHAR(100)', ddl, flags=re.IGNORECASE)
    # 5. Convert MySQL DATETIME -> TIMESTAMP, TINYINT(1) -> SMALLINT, DOUBLE -> DOUBLE PRECISION
    ddl = re.sub(r'\bDATETIME\b', 'TIMESTAMP', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\bTINYINT(?:\s*\(\d+\))?\b', 'SMALLINT', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\b(MEDIUMINT|INT)\s*\(\d+\)', 'INT', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\bBIGINT\s*\(\d+\)', 'BIGINT', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\b(TINYTEXT|MEDIUMTEXT|LONGTEXT)\b', 'TEXT', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\bDOUBLE(?!\s+PRECISION)\b', 'DOUBLE PRECISION', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\bUNSIGNED\b', '', ddl, flags=re.IGNORECASE)
    # 6. Strip MySQL ON UPDATE CURRENT_TIMESTAMP and ENGINE=... suffix
    ddl = re.sub(r'\bON\s+UPDATE\s+CURRENT_TIMESTAMP\b', '', ddl, flags=re.IGNORECASE)
    ddl = re.sub(r'\)\s*ENGINE\s*=\s*\w+[^;]*', ')', ddl, flags=re.IGNORECASE)
    return ddl

def save_to_postgres(ddl_sql: str, table_order: list, tables_dict: dict):
    """Executes sanitized DDL and inserts generated DataFrames into PostgreSQL in FK-safe order."""
    pg_ddl = sanitize_ddl_for_postgres(ddl_sql)
    with engine.begin() as conn:
        # Cleanly reset public schema so any previous or conflicting tables are removed
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))

        # Execute the sanitized DDL statements
        for statement in pg_ddl.split(";"):
            lines = [line for line in statement.splitlines() if not line.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if stmt:
                conn.execute(text(stmt))

        # Fetch actual table names created in PostgreSQL (lowercase unless quoted)
        existing_tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )).fetchall()
        pg_table_map = {r[0].lower(): r[0] for r in existing_tables}

        # Insert records in parent -> child order
        for tbl in table_order:
            # Match table name case-insensitively in tables_dict
            matching_key = next((k for k in tables_dict.keys() if k.lower() == tbl.lower()), tbl)
            df = tables_dict.get(matching_key)
            if df is not None and not df.empty:
                actual_tbl_name = pg_table_map.get(tbl.lower(), tbl.lower())
                df_to_insert = df.copy()
                df_to_insert.columns = [c.lower() for c in df_to_insert.columns]
                df_to_insert.to_sql(actual_tbl_name, con=conn, if_exists="append", index=False)

def create_zip_archive(tables_dict: dict) -> bytes:
    """Packages all generated DataFrames into a downloadable ZIP archive of CSVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for table_name, df in tables_dict.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{table_name}.csv", csv_bytes)
    buf.seek(0)
    return buf.getvalue()

# --- 4. Sidebar Navigation ---
st.sidebar.title("Navigation")
active_tab = st.sidebar.radio(
    "Go to:",
    ["Data Generation", "Talk to your data"]
)

st.sidebar.divider()
st.sidebar.subheader("System Status")
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.sidebar.success("PostgreSQL: Connected")
except Exception as e:
    st.sidebar.error(f"PostgreSQL Offline: {e}")

if langfuse:
    st.sidebar.success("Langfuse: Active")
else:
    st.sidebar.info("Langfuse: Keys not set (Running without tracing)")

# ==========================================
# TAB 1: DATA GENERATION
# ==========================================
if active_tab == "Data Generation":
    st.title("Synthetic Data Generation")
    st.markdown("Upload a DDL schema (up to 5–7 tables), provide instructions, and generate relationally valid data.")

    # 1. Upload DDL File
    uploaded_file = st.file_uploader(
        "Upload DDL Schema (.sql, .txt, .ddl)",
        type=["sql", "txt", "ddl"]
    )
    if uploaded_file is not None:
        st.session_state.ddl_schema = uploaded_file.read().decode("utf-8")

    with st.expander("View / Edit DDL Schema", expanded=bool(st.session_state.ddl_schema)):
        st.session_state.ddl_schema = st.text_area(
            "SQL DDL Statements:",
            value=st.session_state.ddl_schema,
            height=200
        )

    # 2. User Prompt & Generation Parameters
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        user_instructions = st.text_area(
            "Data Generation Instructions (Prompt):",
            value="Generate realistic data. Ensure all Primary Keys are unique and all Foreign Keys strictly reference valid Primary Keys in parent tables. Format dates as YYYY-MM-DD.",
            height=100
        )
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.4, step=0.1)
    with col3:
        rows_per_table = st.number_input("Rows per table", min_value=1, max_value=100, value=10)

    # 3. Generate Button
    if st.button("Generate", type="primary"):
        if not st.session_state.ddl_schema.strip():
            st.error("Please upload or paste a DDL schema first.")
        else:
            with st.spinner("Generating relational data via Gemini 2.5 Flash & storing in PostgreSQL..."):
                generation_prompt = f"""
You are a database expert. Generate synthetic, realistic, and relationally valid data for the following SQL DDL schema.

DDL SCHEMA:
{st.session_state.ddl_schema}

USER INSTRUCTIONS:
{user_instructions}
Generate approximately {rows_per_table} rows per table.

CRITICAL RULES:
1. Respect all data types, NOT NULL constraints, UNIQUE constraints, DATE/TIME formats (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS), Primary Keys, and Foreign Keys.
2. Return a JSON object with EXACTLY two top-level keys:
   - "table_order": A list of exact table names in valid parent-to-child insertion order (so foreign keys never fail).
   - "tables": An object mapping each exact table name from "table_order" to an array of row objects (column_name: value).
"""
                try:
                    result = call_gemini_json(
                        prompt=generation_prompt,
                        temperature=temperature,
                        trace_name="generate_initial_dataset"
                    )
                    tables_data = result.get("tables", {})
                    table_order = result.get("table_order", list(tables_data.keys()))

                    st.session_state.table_order = table_order
                    st.session_state.generated_tables = {
                        tbl: pd.DataFrame(rows) for tbl, rows in tables_data.items()
                    }

                    # Save to PostgreSQL
                    save_to_postgres(
                        st.session_state.ddl_schema,
                        st.session_state.table_order,
                        st.session_state.generated_tables
                    )
                    st.success("Data generated and saved to PostgreSQL successfully!")
                except Exception as e:
                    st.error(f"Error during generation or DB insertion: {e}")

    # 4. Preview, Modify per Table, and Download CSV / ZIP
    if st.session_state.generated_tables:
        st.divider()
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.subheader("Generated Tables Preview & Feedback Modification")
        with header_col2:
            zip_bytes = create_zip_archive(st.session_state.generated_tables)
            st.download_button(
                label="Download All Tables (ZIP)",
                data=zip_bytes,
                file_name="generated_data.zip",
                mime="application/zip",
                use_container_width=True
            )

        table_Tabs = st.tabs(st.session_state.table_order)
        for idx, tbl_name in enumerate(st.session_state.table_order):
            with table_Tabs[idx]:
                df = st.session_state.generated_tables[tbl_name]
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {tbl_name}.csv",
                    data=csv_data,
                    file_name=f"{tbl_name}.csv",
                    mime="text/csv",
                    key=f"dl_{tbl_name}"
                )

                st.markdown(f"#### Modify `{tbl_name}` Data")
                mod_col1, mod_col2 = st.columns([4, 1])
                with mod_col1:
                    feedback_prompt = st.text_input(
                        f"Enter instructions to modify `{tbl_name}` (e.g., 'Change all state values to CA', 'Add 3 more rows'):",
                        key=f"Prompt_{tbl_name}"
                    )
                with mod_col2:
                    st.write("")
                    st.write("")
                    submit_mod = st.button("Submit", key=f"btn_{tbl_name}", use_container_width=True)

                if submit_mod and feedback_prompt.strip():
                    with st.spinner(f"Updating `{tbl_name}` and syncing with PostgreSQL..."):
                        current_state_json = {
                            k: v.to_dict(orient="records")
                            for k, v in st.session_state.generated_tables.items()
                        }
                        mod_prompt = f"""
You are modifying synthetic database data based on user feedback.

DDL SCHEMA:
{st.session_state.ddl_schema}

CURRENT DATABASE DATA (JSON):
{json.dumps(current_state_json)}

TARGET TABLE TO MODIFY: "{tbl_name}"
USER FEEDBACK FOR "{tbl_name}":
{feedback_prompt}

CRITICAL RULES:
1. Apply the requested modification to "{tbl_name}".
2. Ensure Primary Key uniqueness and Foreign Key integrity across all tables are strictly maintained.
3. Return a JSON object with a single key "rows" containing the updated array of row objects for "{tbl_name}".
"""
                        try:
                            mod_result = call_gemini_json(
                                prompt=mod_prompt,
                                temperature=temperature,
                                trace_name=f"modify_table_{tbl_name}"
                            )
                            updated_rows = mod_result.get("rows", [])
                            st.session_state.generated_tables[tbl_name] = pd.DataFrame(updated_rows)

                            save_to_postgres(
                                st.session_state.ddl_schema,
                                st.session_state.table_order,
                                st.session_state.generated_tables
                            )
                            st.success(f"Table `{tbl_name}` updated and saved to PostgreSQL!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Modification failed: {e}")

# ==========================================
# TAB 2: TALK TO YOUR DATA
# ==========================================
elif active_tab == "Talk to your data":
    st.title("Talk to Your Data")
    st.markdown("Ask questions in natural language. Gemini converts your question into PostgreSQL queries and summarizes the results.")

    try:
        with engine.connect() as conn:
            tables_res = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)).fetchall()
            db_tables = [r[0] for r in tables_res]
    except Exception as e:
        db_tables = []
        st.error(f"Cannot connect to PostgreSQL: {e}")

    if not db_tables:
        st.warning("No tables found in PostgreSQL yet. Please generate data in the 'Data Generation' tab first.")
    else:
        with st.expander("Available Tables in PostgreSQL", expanded=False):
            for t in db_tables:
                df_db = pd.read_sql(f'SELECT * FROM "{t}" LIMIT 5', engine)
                st.write(f"**`{t}`** (Sample rows)")
                st.dataframe(df_db, use_container_width=True)

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sql" in msg:
                    st.code(msg["sql"], language="sql")
                if "df" in msg and msg["df"] is not None:
                    st.dataframe(msg["df"], use_container_width=True)

        user_question = st.chat_input("Ask a question about your data (e.g., 'How many companies are in each industry?')")
        if user_question:
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Generating SQL and querying PostgreSQL..."):
                    with engine.connect() as conn:
                        cols_info = conn.execute(text("""
                            SELECT table_name, column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_schema = 'public'
                            ORDER BY table_name, ordinal_position
                        """)).fetchall()
                    schema_summary = "\n".join([f"{r[0]}.{r[1]} ({r[2]})" for r in cols_info])

                    sql_prompt = f"""
You are a PostgreSQL expert. Convert the user's question into a valid, read-only PostgreSQL SELECT query.

DATABASE SCHEMA:
{schema_summary}

DDL CONTEXT:
{st.session_state.ddl_schema}

USER QUESTION:
{user_question}

Return a JSON object with two keys:
- "sql_query": The exact PostgreSQL SELECT query (no markdown formatting).
- "explanation": A brief explanation of how the query answers the question.
"""
                    try:
                        sql_res = call_gemini_json(
                            prompt=sql_prompt,
                            temperature=0.0,
                            trace_name="talk_to_data_text2sql"
                        )
                        sql_query = sql_res.get("sql_query", "")
                        explanation = sql_res.get("explanation", "")

                        st.markdown(explanation)
                        st.code(sql_query, language="sql")

                        result_df = pd.read_sql(sql_query, engine)
                        st.dataframe(result_df, use_container_width=True)

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": explanation,
                            "sql": sql_query,
                            "df": result_df
                        })
                    except Exception as e:
                        err_msg = f"Failed to execute query: {e}"
                        st.error(err_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": err_msg
                        })
