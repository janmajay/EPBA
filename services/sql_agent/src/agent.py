import os
import sys
import sqlite3
import json
# Add parent dir to path to find services module
from shared.src.config import settings
from shared.src.logger import configure_logger, log_execution_time
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import Langfuse

# Configure Logger
logger = configure_logger("sql_agent")

# Initialize Langfuse
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

class SQLAgentService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0)
        
        # Optimization: Pre-load schema to avoid dynamic tool calls
        self.table_info = self.db.get_table_info()
        
        # Fast, single-shot SQL generator prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly efficient SQLite data extraction expert. 
Given the user's question and the database schema below, write ONLY a valid SQLite query to answer the question.
DO NOT wrap the query in markdown formatting like ```sql...```.
DO NOT include any explanations, preambles, or conversational text. Returning exactly the raw SQL string is CRITICAL.

Schema:
{schema}

Instructions:
1. Try to use JOINs to merge data (demographics, conditions, medications, encounters) if the user asks for "details" about a patient.
2. Group medications and conditions using GROUP_CONCAT if necessary, or just return the rows.
3. If looking up by name, use the `LIKE` operator with wildcard (e.g., `first_name || ' ' || last_name LIKE '%Abdul%'`).
4. Keep the query highly optimized and limited to exactly what the user asks.
"""),
            ("user", "{question}")
        ])
        
        # Single-shot pipeline: Prompt -> LLM -> Raw SQL String
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _execute_query(self, query: str) -> str:
        try:
            # Clean up LLM output in case it hallucinated markdown
            cleaned_query = query.replace("```sql", "").replace("```", "").strip()
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(cleaned_query)
            rows = cursor.fetchall()
            
            if not rows:
                return "No Records Found"
                
            # Construct a raw JSON output of the fetched rows
            result = [dict(row) for row in rows]
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error("sqlite_execution_failed", error=str(e), sql=cleaned_query)
            return f"Error executing database query: {str(e)}"
        finally:
            if 'conn' in locals():
                conn.close()

    def query(self, user_query: str, trace_id: str = None) -> str:
        with log_execution_time(logger, "sql_query_execution"):
            logger.info("received_query", query=user_query, trace_id=trace_id)
            
            # Start Langfuse Generation Span
            generation = None
            if trace_id:
                generation = langfuse.generation(
                    trace_id=trace_id,
                    name="SQL Agent Query Generation",
                    model=settings.LLM_MODEL_NAME,
                    model_parameters={"temperature": 0.0},
                    input={"question": user_query, "schema_provided": True},
                )
                
            try:
                # 1. 0-Shot generate the SQL query (Fast: ~1-2 seconds)
                sql_query = self.chain.invoke({
                    "schema": self.table_info,
                    "question": user_query
                })
                logger.info("generated_sql", sql=sql_query)
                
                if generation:
                    generation.update(output=sql_query)
                
                # 2. Execute SQL directly in Python (Fast: ~0.05 seconds)
                result = self._execute_query(sql_query)
                
                logger.info("query_success", output_preview=result[:100])
                
                # The Orchestrator is responsible for synthesis, so returning raw JSON rows is perfect.
                return result
                
            except Exception as e:
                logger.error("query_failed", error=str(e))
                if generation:
                    generation.update(level="ERROR", status_message=str(e))
                return f"Error executing SQL query: {str(e)}"

def get_sql_agent(db_path: str = None):
    final_path = db_path or settings.SQL_DB_PATH # Default from settings logic
    return SQLAgentService(final_path)
