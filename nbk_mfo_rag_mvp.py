"""
NBK MFO Analysis System - Production Grade Architecture

A scalable RAG system for microfinance institution analysis with:
- SOLID design principles
- Comprehensive logging
- Type safety
- Modular components
- Production-ready error handling

Author: [Your Name]
Date: 2025
"""

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
from dotenv import load_dotenv
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class SystemConfig:
    """Centralized configuration management"""
    input_dir: Path
    output_dir: Path
    embedding_model: str
    deepseek_api_key: Optional[str]
    leader_threshold: float = 0.80
    vector_db_path: str = "./chromadb"
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Initialize configuration from environment"""
        load_dotenv()
        
        input_dir = Path(os.getenv(
            'INPUT_DIR',
            r"C:\Users\m.zakaryanov\Desktop\fintech_analysis\rmc\npl_nbk\input"
        ))
        output_dir = Path(os.getenv(
            'OUTPUT_DIR',
            r"C:\Users\m.zakaryanov\Desktop\fintech_analysis\mfo-rag\output"
        ))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            deepseek_api_key=os.getenv("DEEPSEEK_API")
        )


# ============================================================================
# Domain Models
# ============================================================================

@dataclass
class MFORecord:
    """Domain model for MFO financial record"""
    mfo_name: str
    quarter: str
    year: int
    assets: float
    portfolio_gross: float
    del_31_60: float
    del_61_90: float
    del_90_plus: float
    format_type: str
    is_leader: bool = False
    
    @property
    def over30(self) -> float:
        return self.del_31_60 + self.del_61_90 + self.del_90_plus
    
    @property
    def npl(self) -> float:
        return self.del_90_plus
    
    @property
    def over30_pct(self) -> float:
        return (self.over30 / self.portfolio_gross * 100) if self.portfolio_gross > 0 else 0
    
    @property
    def npl_pct(self) -> float:
        return (self.npl / self.portfolio_gross * 100) if self.portfolio_gross > 0 else 0


@dataclass
class AggregationResult:
    """Container for aggregation results"""
    name: str
    data: pd.DataFrame
    metadata: Dict


# ============================================================================
# Interface Definitions (SOLID: Interface Segregation)
# ============================================================================

class DataParser(Protocol):
    """Interface for data parsing implementations"""
    def parse(self) -> pd.DataFrame:
        ...


class DataAggregator(Protocol):
    """Interface for data aggregation implementations"""
    def aggregate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ...


class VectorStore(Protocol):
    """Interface for vector storage implementations"""
    def index(self, documents: List[Dict]) -> None:
        ...
    
    def search(self, query: str, n_results: int) -> List[str]:
        ...


class LLMProvider(Protocol):
    """Interface for LLM implementations"""
    def generate(self, prompt: str, max_tokens: int) -> str:
        ...


# ============================================================================
# Data Parsing Layer (SOLID: Single Responsibility)
# ============================================================================

class QuarterParser:
    """Handles quarter string parsing logic"""
    
    @staticmethod
    def parse(sheet_name: str) -> Optional[str]:
        """Extract quarter date from sheet name"""
        match = re.search(r'01\.(\d{2})\.20(\d{2})', sheet_name)
        if match:
            month, year = match.groups()
            return f"20{year}-{int(month):02d}-01"
        return None


class ExcelSheetParser:
    """Parses individual Excel sheets with format detection"""
    
    def __init__(self, quarter_parser: QuarterParser):
        self.quarter_parser = quarter_parser
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse(self, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Parse single sheet with automatic format detection"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='openpyxl')
        except Exception as e:
            self.logger.warning(f"Failed to read sheet {sheet_name}: {e}")
            return pd.DataFrame()
        
        header_row = self._find_header_row(df)
        if header_row is None:
            return pd.DataFrame()
        
        has_buckets = self._detect_bucket_format(df, header_row)
        data_start = header_row + 2
        
        if data_start >= len(df):
            return pd.DataFrame()
        
        if has_buckets:
            return self._parse_new_format(df, data_start, file_path, sheet_name)
        else:
            return self._parse_old_format(df, data_start, file_path, sheet_name)
    
    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """Locate header row containing 'наименование'"""
        for idx, cell in df.iloc[:, 1].items():
            if pd.notna(cell) and 'наименование' in str(cell).lower():
                return idx
        return None
    
    def _detect_bucket_format(self, df: pd.DataFrame, header_row: int) -> bool:
        """Check if data uses bucket format (2023+)"""
        for col_idx in range(10):
            if col_idx >= df.shape[1]:
                break
            cell = df.iloc[header_row + 1, col_idx] if header_row + 1 < len(df) else None
            if pd.notna(cell) and any(term in str(cell).lower() for term in ['31', '60', '90', 'просрочен']):
                return True
        return False
    
    def _parse_new_format(self, df: pd.DataFrame, data_start: int, 
                         file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Parse 2023+ format with explicit buckets"""
        df_data = df.iloc[data_start:, :10].copy()
        mask = df_data.iloc[:, 0].astype(str).str.match(r'^\d+$', na=False)
        df_data = df_data[mask].reset_index(drop=True)
        
        if df_data.empty:
            return pd.DataFrame()
        
        df_data.columns = ['no', 'mfo_name', 'assets', 'portfolio_gross',
                           'del_31_60', 'del_61_90', 'del_90_plus',
                           'liabilities', 'equity', 'profit']
        
        for col in ['assets', 'portfolio_gross', 'del_31_60', 'del_61_90', 'del_90_plus']:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        
        df_data['format'] = 'new'
        df_data['quarter'] = self.quarter_parser.parse(sheet_name)
        df_data['year'] = pd.to_datetime(df_data['quarter']).dt.year
        df_data['file'] = file_path.name
        
        return df_data.dropna(subset=['mfo_name', 'assets', 'portfolio_gross'])
    
    def _parse_old_format(self, df: pd.DataFrame, data_start: int,
                         file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Parse 2018-2022 format with estimated buckets"""
        df_data = df.iloc[data_start:, :6].copy()
        mask = df_data.iloc[:, 0].astype(str).str.match(r'^\d+$', na=False)
        df_data = df_data[mask].reset_index(drop=True)
        
        if df_data.empty:
            return pd.DataFrame()
        
        df_data.columns = ['no', 'mfo_name', 'assets', 'net_loans', 'liabilities', 'equity']
        df_data['assets'] = pd.to_numeric(df_data['assets'], errors='coerce')
        df_data['net_loans'] = pd.to_numeric(df_data['net_loans'], errors='coerce')
        
        # Estimate buckets from reserves
        df_data['portfolio_gross'] = df_data['net_loans'] / 0.92
        reserves = df_data['portfolio_gross'] - df_data['net_loans']
        df_data['del_90_plus'] = reserves * 0.8
        df_data['del_31_60'] = reserves * 0.1
        df_data['del_61_90'] = reserves * 0.1
        df_data['format'] = 'old'
        
        df_data['quarter'] = self.quarter_parser.parse(sheet_name)
        df_data['year'] = pd.to_datetime(df_data['quarter']).dt.year
        df_data['file'] = file_path.name
        
        return df_data.dropna(subset=['mfo_name', 'assets', 'portfolio_gross'])


class MFODataParser:
    """Orchestrates full data parsing pipeline"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.sheet_parser = ExcelSheetParser(QuarterParser())
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse(self) -> pd.DataFrame:
        """Parse all Excel files in input directory"""
        file_paths = list(self.config.input_dir.glob("*.xlsx"))
        
        if not file_paths:
            raise ValueError(f"No Excel files found in {self.config.input_dir}")
        
        self.logger.info(f"Found {len(file_paths)} files to process")
        
        all_data = []
        for file_path in file_paths:
            self.logger.info(f"Processing: {file_path.name}")
            all_data.extend(self._parse_file(file_path))
        
        if not all_data:
            raise ValueError("No data extracted from any files")
        
        df = pd.concat(all_data, ignore_index=True)
        df = self._enrich_metrics(df)
        df = self._identify_leaders(df)
        
        self.logger.info(f"Parsed {len(df)} records from {df['mfo_name'].nunique()} MFOs")
        return df
    
    def _parse_file(self, file_path: Path) -> List[pd.DataFrame]:
        """Parse all sheets in single Excel file"""
        try:
            xl = pd.ExcelFile(file_path, engine='openpyxl')
        except Exception as e:
            self.logger.error(f"Failed to open {file_path.name}: {e}")
            return []
        
        sheets_data = []
        for sheet_name in xl.sheet_names:
            if not QuarterParser.parse(sheet_name):
                continue
            
            df_sheet = self.sheet_parser.parse(file_path, sheet_name)
            if not df_sheet.empty:
                sheets_data.append(df_sheet)
        
        return sheets_data
    
    def _enrich_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics"""
        df['over30'] = df['del_31_60'] + df['del_61_90'] + df['del_90_plus']
        df['npl'] = df['del_90_plus']
        
        df['over30_pct'] = np.where(
            df['portfolio_gross'] > 0,
            df['over30'] / df['portfolio_gross'] * 100,
            np.nan
        )
        df['npl_pct'] = np.where(
            df['portfolio_gross'] > 0,
            df['npl'] / df['portfolio_gross'] * 100,
            np.nan
        )
        
        return df
    
    def _identify_leaders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark market leaders by assets per quarter"""
        leaders_list = []
        
        for quarter in df['quarter'].unique():
            quarter_df = df[df['quarter'] == quarter].copy()
            quarter_df = quarter_df.sort_values('assets', ascending=False)
            
            total_assets = quarter_df['assets'].sum()
            quarter_df['cumulative_share'] = quarter_df['assets'].cumsum() / total_assets
            quarter_df['is_leader'] = quarter_df['cumulative_share'] <= self.config.leader_threshold
            
            leaders_list.append(quarter_df)
        
        return pd.concat(leaders_list, ignore_index=True)


# ============================================================================
# Aggregation Layer (SOLID: Single Responsibility)
# ============================================================================

class BaseAggregator(ABC):
    """Abstract base for aggregation strategies"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """Compute aggregation"""
        pass
    
    def _weighted_average_npl(self, group: pd.DataFrame) -> float:
        """Calculate portfolio-weighted NPL"""
        valid = group.dropna(subset=['npl_pct', 'portfolio_gross'])
        if len(valid) == 0:
            return np.nan
        return np.average(valid['npl_pct'], weights=valid['portfolio_gross'])


class YearlyMarketAggregator(BaseAggregator):
    """Yearly market-wide statistics"""
    
    def compute(self) -> pd.DataFrame:
        return self.df.groupby('year').apply(self._aggregate_year).reset_index(drop=True)
    
    def _aggregate_year(self, group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'year': group['year'].iloc[0],
            'mfo_count': group['mfo_name'].nunique(),
            'total_assets': group['assets'].sum(),
            'total_portfolio': group['portfolio_gross'].sum(),
            'total_npl': group['npl'].sum(),
            'avg_npl_pct': self._weighted_average_npl(group)
        })


class YearlySegmentAggregator(BaseAggregator):
    """Yearly statistics by market segment"""
    
    def __init__(self, df: pd.DataFrame, is_leader: bool):
        super().__init__(df[df['is_leader'] == is_leader])
        self.is_leader = is_leader
    
    def compute(self) -> pd.DataFrame:
        return self.df.groupby('year').apply(self._aggregate_year).reset_index(drop=True)
    
    def _aggregate_year(self, group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'year': group['year'].iloc[0],
            'segment': 'leaders' if self.is_leader else 'non_leaders',
            'mfo_count': group['mfo_name'].nunique(),
            'total_assets': group['assets'].sum(),
            'total_portfolio': group['portfolio_gross'].sum(),
            'avg_npl_pct': self._weighted_average_npl(group)
        })


class TopMFOsAggregator(BaseAggregator):
    """Top MFOs ranking by period"""
    
    def __init__(self, df: pd.DataFrame, top_n: int = 10):
        super().__init__(df)
        self.top_n = top_n
    
    def compute(self) -> pd.DataFrame:
        top_per_year = []
        
        for year in self.df['year'].unique():
            year_data = self.df[self.df['year'] == year]
            latest_quarter = year_data['quarter'].max()
            year_latest = year_data[year_data['quarter'] == latest_quarter]
            
            top_n = year_latest.nlargest(self.top_n, 'assets')[
                ['mfo_name', 'year', 'quarter', 'assets', 'portfolio_gross', 'npl_pct', 'is_leader']
            ].copy()
            top_n['rank'] = range(1, len(top_n) + 1)
            
            top_per_year.append(top_n)
        
        return pd.concat(top_per_year, ignore_index=True)


class AggregationEngine:
    """Manages all aggregation computations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.aggregations: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_all(self) -> Dict[str, pd.DataFrame]:
        """Execute all aggregation strategies"""
        self.logger.info("Computing aggregations")
        
        strategies = {
            'yearly_market': YearlyMarketAggregator(self.df),
            'yearly_leaders': YearlySegmentAggregator(self.df, is_leader=True),
            'yearly_non_leaders': YearlySegmentAggregator(self.df, is_leader=False),
            'top_mfos': TopMFOsAggregator(self.df, top_n=10)
        }
        
        for name, strategy in strategies.items():
            self.aggregations[name] = strategy.compute()
            self.logger.info(f"Computed: {name}")
        
        return self.aggregations
    
    def save(self, output_path: Path) -> None:
        """Save aggregations to Excel"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for name, df in self.aggregations.items():
                df.to_excel(writer, sheet_name=name, index=False)
        
        self.logger.info(f"Saved aggregations to {output_path}")


# ============================================================================
# RAG Components (SOLID: Dependency Inversion)
# ============================================================================

class ChromaVectorStore:
    """ChromaDB implementation of vector storage"""
    
    def __init__(self, collection_name: str, persist_path: str):
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.client = chromadb.PersistentClient(path=persist_path)
        
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def index(self, documents: List[Dict]) -> None:
        """Index documents with embeddings"""
        if not documents:
            return
        
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedder.encode(texts)
        
        self.collection.add(
            ids=[doc['id'] for doc in documents],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[doc.get('metadata', {}) for doc in documents]
        )
        
        self.logger.info(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, n_results: int = 3) -> List[str]:
        """Semantic search"""
        query_embedding = self.embedder.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results['documents'][0] if results['documents'] else []


class DeepSeekProvider:
    """DeepSeek LLM implementation"""
    
    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required for DeepSeek")
        
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate completion"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise


class RAGSystem:
    """Main RAG orchestration"""
    
    def __init__(self, 
                 vector_store: ChromaVectorStore,
                 llm_provider: Optional[DeepSeekProvider],
                 agg_engine: AggregationEngine):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.agg_engine = agg_engine
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> None:
        """Index aggregation summaries"""
        summary = self._generate_summary()
        chunks = self._chunk_summary(summary)
        
        documents = [
            {'id': f'summary_{i}', 'text': chunk, 'metadata': {'type': 'summary'}}
            for i, chunk in enumerate(chunks)
        ]
        
        self.vector_store.index(documents)
        self.logger.info("RAG system initialized")
    
    def query(self, question: str) -> str:
        """Answer question using RAG"""
        context_chunks = self.vector_store.search(question, n_results=2)
        context = "\n\n".join(context_chunks)
        
        if self.llm_provider:
            prompt = self._build_prompt(question, context)
            return self.llm_provider.generate(prompt)
        else:
            return f"Context:\n{context}"
    
    def _generate_summary(self) -> str:
        """Generate text summary from aggregations"""
        market = self.agg_engine.aggregations['yearly_market']
        
        lines = ["Market Overview"]
        for _, row in market.iterrows():
            lines.append(
                f"{int(row['year'])}: {int(row['mfo_count'])} MFOs, "
                f"Assets {row['total_assets']/1e9:.1f}B KZT, "
                f"NPL {row['avg_npl_pct']:.2f}%"
            )
        
        return "\n".join(lines)
    
    def _chunk_summary(self, summary: str) -> List[str]:
        """Split summary into chunks"""
        return [chunk.strip() for chunk in summary.split('\n') if chunk.strip()]
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construct LLM prompt"""
        return f"""You are an analyst for the National Bank of Kazakhstan.

Context:
{context}

Question: {question}

Provide a concise, data-driven answer in 2-3 sentences."""


# ============================================================================
# Application Facade (SOLID: Single Responsibility)
# ============================================================================

class MFOAnalysisSystem:
    """Main application facade"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.parser: Optional[MFODataParser] = None
        self.agg_engine: Optional[AggregationEngine] = None
        self.rag: Optional[RAGSystem] = None
        self.data: Optional[pd.DataFrame] = None
    
    def initialize(self) -> None:
        """Initialize all system components"""
        self.logger.info("Initializing MFO Analysis System")
        
        # Parse data
        self.parser = MFODataParser(self.config)
        self.data = self.parser.parse()
        
        # Compute aggregations
        self.agg_engine = AggregationEngine(self.data)
        self.agg_engine.compute_all()
        self.agg_engine.save(self.config.output_dir / "aggregations.xlsx")
        
        # Initialize RAG
        vector_store = ChromaVectorStore("nbk_mfo", self.config.vector_db_path)
        
        llm_provider = None
        if self.config.deepseek_api_key and OPENAI_AVAILABLE:
            llm_provider = DeepSeekProvider(self.config.deepseek_api_key)
        
        self.rag = RAGSystem(vector_store, llm_provider, self.agg_engine)
        self.rag.initialize()
        
        self.logger.info("System initialization complete")
    
    def query(self, question: str) -> str:
        """Query the system"""
        if not self.rag:
            raise RuntimeError("System not initialized")
        return self.rag.query(question)
    
    def get_aggregations(self) -> Dict[str, pd.DataFrame]:
        """Access aggregated data"""
        if not self.agg_engine:
            raise RuntimeError("System not initialized")
        return self.agg_engine.aggregations
    
    def get_raw_data(self) -> pd.DataFrame:
        """Access raw parsed data"""
        if self.data is None:
            raise RuntimeError("System not initialized")
        return self.data


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Application entry point"""
    config = SystemConfig.from_env()
    system = MFOAnalysisSystem(config)
    
    try:
        system.initialize()
        
        # Demo queries
        questions = [
            "What is the average NPL in 2024?",
            "Which MFOs are market leaders?",
            "How has the market evolved since 2020?"
        ]
        
        for question in questions:
            logger.info(f"Question: {question}")
            answer = system.query(question)
            logger.info(f"Answer: {answer}\n")
    
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()