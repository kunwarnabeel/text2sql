import os
import re
import time
from typing import Optional

import pandas as pd
import streamlit as st

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables should be set manually
    pass

# === Vanna imports (LLM + Vector Store combinations) ===
from vanna.qdrant.qdrant_vector import Qdrant_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat
from openai import OpenAI

try:
    # Ollama is optional. If not installed, we hide the option in the UI.
    from vanna.ollama.ollama import Ollama
    OLLAMA_AVAILABLE = True
except Exception:
    Ollama = None  # type: ignore
    OLLAMA_AVAILABLE = False

# --------------
# App constants
# --------------
APP_TITLE = "Text‑to‑SQL Chatbot"
CHROMA_DIR = os.environ.get("VANNA_CHROMA_DIR", ".vanna_chroma")  # persistent local vector store

# --------------
# OpenRouter Chat Class
# --------------

class OpenRouter_Chat(OpenAI_Chat):
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Extract OpenRouter specific config
        api_key = config.get("api_key")
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        model = config.get("model", "openai/gpt-4o-mini")
        
        # Initialize parent class
        super().__init__(config=config)
        
        # Override the client with OpenRouter configuration
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

# --------------
# Helpers
# --------------

def make_vanna(provider: str, model_name: str, db_identifier: str, openai_api_key: Optional[str] = None, ollama_base_url: Optional[str] = None, openrouter_api_key: Optional[str] = None):
    """Create a Vanna instance using Qdrant cloud vector store.

    provider: 'openai', 'openrouter', or 'ollama'
    model_name: e.g. 'gpt-4o-mini' (OpenAI), 'openai/gpt-4o-mini' (OpenRouter) or 'llama3' (Ollama)
    db_identifier: unique identifier for the database
    """
    qdrant_cfg = {"url": ":memory:", "collection_name": f"vanna_{db_identifier}"}

    if provider == "openai":
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key")

        class MyVanna(OpenAI_Chat, Qdrant_VectorStore):
            def __init__(self, config=None):
                OpenAI_Chat.__init__(self, config=config)
                Qdrant_VectorStore.__init__(self, config=config)

        return MyVanna(config={"api_key": openai_api_key, "model": model_name, **qdrant_cfg})

    elif provider == "openrouter":
        if not openrouter_api_key:
            raise ValueError("Missing OpenRouter API key")

        class MyVanna(OpenRouter_Chat, Qdrant_VectorStore):
            def __init__(self, config=None):
                OpenRouter_Chat.__init__(self, config=config)
                Qdrant_VectorStore.__init__(self, config=config)

        return MyVanna(config={
            "api_key": openrouter_api_key, 
            "model": model_name, 
            "base_url": "https://openrouter.ai/api/v1",
            **qdrant_cfg
        })

    elif provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install `vanna[ollama]` and ensure Ollama is running.")

        class MyVanna(Ollama, Qdrant_VectorStore):  # type: ignore
            def __init__(self, config=None):
                Ollama.__init__(self, config=config)  # type: ignore
                Qdrant_VectorStore.__init__(self, config=config)

        cfg = {"model": model_name, **qdrant_cfg}
        if ollama_base_url:
            cfg["base_url"] = ollama_base_url
        return MyVanna(config=cfg)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def connect_database(vn, db_kind: str, **kwargs):
    """Connect Vanna to a database using built-in helpers.

    Supported db_kind: 'postgres', 'mysql', 'sqlite', 'mssql', 'snowflake', 'duckdb'
    """
    if db_kind == "postgres":
        vn.connect_to_postgres(
            host=kwargs.get("host"),
            dbname=kwargs.get("dbname"),
            user=kwargs.get("user"),
            password=kwargs.get("password"),
            port=int(kwargs.get("port") or 5432),
        )
    elif db_kind == "mysql":
        vn.connect_to_mysql(
            host=kwargs.get("host"),
            dbname=kwargs.get("dbname"),
            user=kwargs.get("user"),
            password=kwargs.get("password"),
            port=int(kwargs.get("port") or 3306),
        )
    elif db_kind == "sqlite":
        vn.connect_to_sqlite(kwargs.get("url"))  # file path or URL (e.g., Chinook sample)
    elif db_kind == "mssql":
        # expects a full ODBC connection string
        vn.connect_to_mssql(kwargs.get("odbc_conn_str"))
    elif db_kind == "snowflake":
        vn.connect_to_snowflake(
            account=kwargs.get("account"),
            username=kwargs.get("username"),
            password=kwargs.get("password"),
            database=kwargs.get("database"),
            schema=kwargs.get("schema") or None,
            role=kwargs.get("role") or None,
        )
    elif db_kind == "duckdb":
        vn.connect_to_duckdb(kwargs.get("url", ":memory:"))
    else:
        raise ValueError(f"Unsupported db_kind: {db_kind}")


def simple_sql_guard(sql: str) -> bool:
    """Basic protection: block DDL/DML and other potentially dangerous commands.
    You can replace this with vn.is_sql_valid(sql=...) if you have a stricter policy.
    """
    forbidden = [r"\\bINSERT\\b", r"\\bUPDATE\\b", r"\\bDELETE\\b", r"\\bDROP\\b", r"\\bALTER\\b", r"\\bTRUNCATE\\b", r"\\bCREATE\\b"]
    return not any(re.search(p, sql, re.IGNORECASE) for p in forbidden)


# --------------
# Streamlit UI
# --------------

st.set_page_config(page_title=APP_TITLE, layout="wide")


st.title(APP_TITLE)
st.caption("Local RAG (ChromaDB) + your choice of LLM. Generate SQL, run it, and chart the results.")

# Get settings from environment variables
provider = os.environ.get("LLM_PROVIDER", "openai")

if provider == "openai":
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    openrouter_api_key = None
    openrouter_model = None
    ollama_model = None
    ollama_base_url = None
elif provider == "openrouter":
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    openai_api_key = None
    openai_model = None
    ollama_model = None
    ollama_base_url = None
elif provider == "ollama":
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "")
    openai_api_key = None
    openai_model = None
    openrouter_api_key = None
    openrouter_model = None
else:
    raise ValueError(f"Unsupported LLM provider: {provider}. Set LLM_PROVIDER to 'openai', 'openrouter', or 'ollama'")

# Get database settings from environment variables
db_kind = os.environ.get("DB_TYPE", "postgres")
safe_charts = os.environ.get("SAFE_CHARTS", "true").lower() == "true"

# Database connection parameters from environment
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "port": int(os.environ.get("DB_PORT", "5432" if db_kind == "postgres" else "3306")),
    "url": os.environ.get("DB_URL"),
    "odbc_conn_str": os.environ.get("DB_ODBC_CONN_STR"),
    "account": os.environ.get("DB_ACCOUNT"),
    "username": os.environ.get("DB_USERNAME"),
    "database": os.environ.get("DB_DATABASE"),
    "schema": os.environ.get("DB_SCHEMA"),
    "role": os.environ.get("DB_ROLE")
}

# Sidebar info
with st.sidebar:
    st.header("Configuration (from environment)")
    st.write(f"**LLM Provider:** {provider}")
    if provider == "openai":
        st.write(f"**Model:** {openai_model}")
        st.write(f"**API Key:** {'✓ Set' if openai_api_key else '✗ Missing'}")
    elif provider == "openrouter":
        st.write(f"**Model:** {openrouter_model}")
        st.write(f"**API Key:** {'✓ Set' if openrouter_api_key else '✗ Missing'}")
    elif provider == "ollama":
        st.write(f"**Model:** {ollama_model}")
        st.write(f"**Base URL:** {ollama_base_url or 'default'}")
    
    st.write(f"**Database:** {db_kind}")
    st.write(f"**Vector store:** Qdrant (in-memory)")

# Create Vanna instance (memoized)
@st.cache_resource(show_spinner=False)
def _get_vn(provider, openai_model, openai_api_key, openrouter_model, openrouter_api_key, ollama_model, ollama_base_url, db_identifier):
    if provider == "openai":
        return make_vanna("openai", openai_model, db_identifier, openai_api_key=openai_api_key)
    elif provider == "openrouter":
        return make_vanna("openrouter", openrouter_model, db_identifier, openrouter_api_key=openrouter_api_key)
    else:
        return make_vanna("ollama", ollama_model, db_identifier, ollama_base_url=ollama_base_url)

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {question, sql, df_head}

if "trained" not in st.session_state:
    st.session_state.trained = False

# Prepare connection arguments based on database type
conn_args = {}

if db_kind in ("postgres", "mysql"):
    conn_args = {
        "host": db_config["host"],
        "dbname": db_config["dbname"],
        "user": db_config["user"],
        "password": db_config["password"],
        "port": db_config["port"]
    }
elif db_kind == "sqlite":
    conn_args["url"] = db_config["url"] or "Chinook.sqlite"
elif db_kind == "mssql":
    conn_args["odbc_conn_str"] = db_config["odbc_conn_str"]
elif db_kind == "snowflake":
    conn_args = {
        "account": db_config["account"],
        "username": db_config["username"],
        "password": db_config["password"],
        "database": db_config["database"],
        "schema": db_config["schema"],
        "role": db_config["role"]
    }
elif db_kind == "duckdb":
    conn_args["url"] = db_config["url"] or ":memory:"

st.subheader("Database Connection")
if not st.session_state.get("db_connected", False):
    st.info(f"Connecting to {db_kind} database...")

# Auto-connect to database on startup
if "vn" not in st.session_state:
    try:
        # Create database identifier from connection info
        if db_kind in ("postgres", "mysql"):
            db_identifier = f"{db_kind}_{conn_args.get('host', 'localhost')}_{conn_args.get('dbname', 'default')}"
        elif db_kind == "sqlite":
            url = conn_args.get('url', 'default').replace('/', '_').replace('\\', '_')
            db_identifier = f"sqlite_{url}"
        elif db_kind == "snowflake":
            db_identifier = f"snowflake_{conn_args.get('account', 'default')}_{conn_args.get('database', 'default')}"
        else:
            db_identifier = f"{db_kind}_default"
        
        # Clean identifier (remove special chars)
        db_identifier = re.sub(r'[^a-zA-Z0-9_]', '_', db_identifier.lower())
        
        vn = _get_vn(provider, openai_model, openai_api_key, openrouter_model, openrouter_api_key, ollama_model, ollama_base_url, db_identifier)
        connect_database(vn, db_kind, **conn_args)
        st.session_state.vn = vn
        st.session_state.db_connected = True
        st.session_state.current_db_id = db_identifier
        st.success(f"Connected to {db_kind} database! Using vector collection: vanna_{db_identifier}")
    except Exception as e:
        st.session_state.db_connected = False
        st.session_state.vn = None
        st.error(f"Failed to connect to {db_kind} database: {e}")
        st.info("Please check your environment variables and database configuration.")

else:
    if st.session_state.get("db_connected", False):
        st.success(f"Connected to {db_kind} database!")
    else:
        st.error("Database connection failed. Please check your configuration.")

st.divider()
st.subheader("Training")

colA, colB = st.columns([1, 2])
with colA:
    bootstrap = st.button("Scan INFORMATION_SCHEMA & Train")
with colB:
    st.caption("This will read INFORMATION_SCHEMA.COLUMNS, create a training plan, and ingest descriptions into the vector store.")

if bootstrap:
    if not st.session_state.db_connected or st.session_state.vn is None:
        st.warning("Connect to your database first.")
    else:
        vn = st.session_state.vn
        try:
            with st.spinner("Reading information schema..."):
                df_info = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
            with st.spinner("Creating training plan..."):
                plan = vn.get_training_plan_generic(df_info)
            with st.spinner("Training (embedding metadata)..."):
                vn.train(plan=plan)
            st.session_state.trained = True
            st.success("Training complete!")
        except Exception as e:
            st.error(f"Training failed: {e}")

with st.expander("Add custom training (DDL / question-SQL / docs)"):
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        ddl_text = st.text_area("DDL", placeholder="CREATE TABLE ...")
        if st.button("Train DDL"):
            if st.session_state.vn is None:
                st.warning("Connect first.")
            else:
                try:
                    st.session_state.vn.train(ddl=ddl_text)
                    st.success("Trained on DDL.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with tcol2:
        q = st.text_area("Question (for Q&A training)")
        sql_example = st.text_area("SQL (matching the question)")
        if st.button("Train Q&A"):
            if st.session_state.vn is None:
                st.warning("Connect first.")
            else:
                try:
                    st.session_state.vn.train(question=q, sql=sql_example)
                    st.success("Trained on question/SQL pair.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    docs_text = st.text_area("Documentation (plain text)")
    if st.button("Train docs"):
        if st.session_state.vn is None:
            st.warning("Connect first.")
        else:
            try:
                st.session_state.vn.train(documentation=docs_text)
                st.success("Trained on docs.")
            except Exception as e:
                st.error(f"Failed: {e}")

st.divider()
st.subheader("Ask questions")

if not st.session_state.db_connected or st.session_state.vn is None:
    st.info("Connect your database to start chatting.")
    st.stop()

# chat input
prompt = st.chat_input("Ask a question about your data...")

if prompt:
    vn = st.session_state.vn

    with st.chat_message("user"):
        st.write(prompt)

    # Step 1: Generate SQL
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            try:
                sql = vn.generate_sql(question=prompt) or ""
            except Exception as e:
                st.error(f"Failed to generate SQL: {e}")
                sql = ""

        if not sql.strip():
            st.warning("No SQL generated.")
        else:
            st.code(sql, language="sql")

        # Guardrail
        if sql and not simple_sql_guard(sql):
            st.error("Blocked potentially unsafe SQL (DDL/DML). Edit your question and try again.")
            st.stop()

        # Step 2: Run SQL
        df = None
        if sql:
            with st.spinner("Running SQL..."):
                try:
                    df = vn.run_sql(sql=sql)
                except Exception as e:
                    st.error(f"SQL execution error: {e}")
                    try:
                        vn.flag_sql_for_review(question=prompt, sql=sql, error_msg=str(e))
                    except Exception:
                        pass

        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df, use_container_width=True)

            # Step 3: Chart
            with st.spinner("Plotting graph..."):
                try:
                    if safe_charts:
                        fig = vn.get_plotly_figure(plotly_code="", df=df)
                    else:
                        plotly_code = vn.generate_plotly_code(question=prompt, sql=sql, df=df)
                        fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Chart not available: {e}")

            st.session_state.history.append({"question": prompt, "sql": sql, "rows": len(df)})
        else:
            st.info("No rows returned.")

# History panel
with st.expander("History"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.caption("No questions asked yet.")
