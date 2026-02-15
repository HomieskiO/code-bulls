import os
import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
# Replace with your actual Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBp9TcVucK1tfINq23K2vfzLXtfWhNYgqs"
DB_DIR = "./chroma_db"


class TradingAgent:
    def __init__(self):
        print("🧠 Loading local rules database...")
        if not os.path.exists(DB_DIR):
            print(f"❌ Error: Database '{DB_DIR}' not found. Run 'build_brain.py' first.")
            return

        # Local embeddings (Zero tokens for retrieval)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)

        # Thinking Engine - Using 'gemini-1.5-flash-latest' for stability
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1
            )
            print("🚀 Agent initialized successfully.")
        except Exception as e:
            print(f"❌ Initialization Error: {e}")

        self.chain = self._create_chain()

    def _create_chain(self):
        # Retrieve the most relevant trading rules from your local vector DB
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

        template = """
        You are a Professional Trading Mentor. Analyze the input against the successful trader's rules provided.

        --- TRADING RULES CONTEXT ---
        {context}
        ------------------------------

        TRADER INPUT: {question}

        TASK:
        1. Compare actions/setups against the rules.
        2. Assign a Score (1-10) or Grade (A-F).
        3. Identify specific rule violations or correct alignments.
        4. Be direct, professional, and focus on discipline.
        """
        prompt = ChatPromptTemplate.from_template(template)
        return ({"context": retriever, "question": RunnablePassthrough()} | prompt | self.llm)

    def analyze_text(self, text):
        """Analyze a manual text setup."""
        return self.chain.invoke(text).content

    def analyze_csv(self, file_path):
        """Analyze a CSV trade log."""
        if not os.path.exists(file_path):
            return f"❌ Error: File '{file_path}' not found."

        try:
            print(f"📊 Processing report: {file_path}")
            df = pd.read_csv(file_path)

            # Select key columns for the mentor to focus on
            cols = ['data_name', 'side', 'entry_time', 'exit_time', 'pnl_pct', 'holding_period_days']
            df_clean = df[[c for c in cols if c in df.columns]].copy()

            # Formatting PnL and Time for better LLM comprehension
            if 'pnl_pct' in df_clean.columns:
                df_clean['pnl_pct'] = (df_clean['pnl_pct'] * 100).round(2).astype(str) + "%"

            for col in ['entry_time', 'exit_time']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).str.split('T').str[0]

            report_summary = df_clean.to_string(index=False)
            print("🧐 Analyzing trade history against the rules...")
            return self.chain.invoke(report_summary).content
        except Exception as e:
            return f"❌ Error processing CSV: {e}"


if __name__ == "__main__":
    agent = TradingAgent()
    while True:
        print("\n--- TRADING AGENT MODES ---")
        print("1. Manual Setup Analysis (Text)")
        print("2. Trade Log Review (CSV)")
        print("3. Exit")
        choice = input("Select (1/2/3): ")

        if choice == '1':
            setup = input("Describe setup: ")
            print("\n" + "=" * 50 + "\n" + agent.analyze_text(setup) + "\n" + "=" * 50)
        elif choice == '2':
            path = input("Enter CSV path: ")
            print("\n" + "=" * 50 + "\n" + agent.analyze_csv(path) + "\n" + "=" * 50)
        elif choice == '3':
            print("👋 Session ended.")
            break