"""
NBK MFO Analysis Dashboard - Streamlit Interface

Interactive dashboard for microfinance institution analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Import your refactored system
# Adjust import path as needed
sys.path.append(str(Path(__file__).parent))
from rag_app import MFOAnalysisSystem, SystemConfig


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="NBK MFO Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# State Management
# ============================================================================

@st.cache_resource
def initialize_system():
    """Initialize the analysis system (cached)"""
    config = SystemConfig.from_env()
    system = MFOAnalysisSystem(config)
    system.initialize()
    return system


@st.cache_data
def get_aggregations(_system):
    """Get aggregated data (cached)"""
    return _system.get_aggregations()


@st.cache_data
def get_raw_data(_system):
    """Get raw data (cached)"""
    return _system.get_raw_data()


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_yearly_market_trend(df_market: pd.DataFrame):
    """Line chart of market NPL evolution"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_market['year'],
        y=df_market['avg_npl_pct'],
        mode='lines+markers',
        name='Average NPL %',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Market NPL Evolution Over Time",
        xaxis_title="Year",
        yaxis_title="NPL (%)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_segment_comparison(df_leaders: pd.DataFrame, df_non_leaders: pd.DataFrame):
    """Compare leaders vs non-leaders"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_leaders['year'],
        y=df_leaders['avg_npl_pct'],
        name='Market Leaders',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        x=df_non_leaders['year'],
        y=df_non_leaders['avg_npl_pct'],
        name='Non-Leaders',
        marker_color='#d62728'
    ))
    
    fig.update_layout(
        title="NPL Comparison: Leaders vs Non-Leaders",
        xaxis_title="Year",
        yaxis_title="Average NPL (%)",
        barmode='group',
        template='plotly_white'
    )
    
    return fig


def plot_top_mfos(df_top: pd.DataFrame, year: int):
    """Horizontal bar chart of top MFOs"""
    df_year = df_top[df_top['year'] == year].sort_values('assets', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_year['mfo_name'],
        x=df_year['assets'] / 1e9,
        orientation='h',
        marker=dict(
            color=df_year['npl_pct'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="NPL %")
        ),
        text=[f"{val:.1f}B KZT" for val in df_year['assets'] / 1e9],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top 10 MFOs by Assets ({year})",
        xaxis_title="Assets (Billion KZT)",
        yaxis_title="",
        height=500,
        template='plotly_white'
    )
    
    return fig


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">NBK MFO Market Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Interactive dashboard for microfinance institution analysis based on National Bank data")
    
    # Initialize system
    with st.spinner("Initializing analysis system..."):
        try:
            system = initialize_system()
            aggregations = get_aggregations(system)
            raw_data = get_raw_data(system)
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        
        st.markdown("---")
        st.subheader("Dataset Info")
        st.metric("Total Records", f"{len(raw_data):,}")
        st.metric("Unique MFOs", raw_data['mfo_name'].nunique())
        st.metric("Years Covered", f"{raw_data['year'].min()} - {raw_data['year'].max()}")
        
        st.markdown("---")
        st.subheader("Quick Stats")
        latest_year = raw_data['year'].max()
        latest_data = raw_data[raw_data['year'] == latest_year]
        
        avg_npl = latest_data['npl_pct'].mean()
        total_assets = latest_data['assets'].sum() / 1e9
        
        st.metric("Latest Year NPL", f"{avg_npl:.2f}%")
        st.metric("Total Market Assets", f"{total_assets:.1f}B KZT")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Overview",
        "🏆 Top MFOs",
        "💬 Ask Questions",
        "📥 Download Data"
    ])
    
    # TAB 1: Market Overview
    with tab1:
        st.header("Market Dynamics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NPL Evolution")
            fig_trend = plot_yearly_market_trend(aggregations['yearly_market'])
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            st.subheader("Leaders vs Non-Leaders")
            fig_comparison = plot_segment_comparison(
                aggregations['yearly_leaders'],
                aggregations['yearly_non_leaders']
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Yearly Market Statistics")
        st.dataframe(
            aggregations['yearly_market'].style.format({
                'total_assets': '{:,.0f}',
                'total_portfolio': '{:,.0f}',
                'avg_npl_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    # TAB 2: Top MFOs
    with tab2:
        st.header("Market Leaders")
        
        year_selector = st.select_slider(
            "Select Year",
            options=sorted(aggregations['top_mfos']['year'].unique()),
            value=aggregations['top_mfos']['year'].max()
        )
        
        fig_top = plot_top_mfos(aggregations['top_mfos'], year_selector)
        st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"Detailed Rankings ({year_selector})")
        
        df_year_top = aggregations['top_mfos'][
            aggregations['top_mfos']['year'] == year_selector
        ].sort_values('rank')
        
        st.dataframe(
            df_year_top[['rank', 'mfo_name', 'assets', 'portfolio_gross', 'npl_pct']].style.format({
                'assets': '{:,.0f}',
                'portfolio_gross': '{:,.0f}',
                'npl_pct': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    # TAB 3: Q&A with RAG
    with tab3:
        st.header("Ask Questions About the Market")
        
        st.markdown("""
        Ask questions in natural language. Examples: \\
        **English:**  "What was the market-average NPL in 2024 and who were the main drivers?" \\
        **Russian:**  "Какие МФО входят в топ-5 по размеру портфеля на конец 2024 года?" \\
        **Kazakh:**   "2024 жылы микрокредиттік ұйымдар арасында орташа NPL қандай болды?" \\
        **Korean:**   "2024년 소액금융 기관들의 평균 부실채권 비율은 얼마였나요?"
        """)
        
        question = st.text_input("Your question:", placeholder="e.g., What was the NPL trend in 2023?")
        
        if st.button("Submit", type="primary"):
            if question:
                with st.spinner("Analyzing..."):
                    try:
                        answer = system.query(question)
                        st.success("Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Query failed: {e}")
            else:
                st.warning("Please enter a question")
        
        st.markdown("---")
        st.subheader("Recent Queries")
        
        demo_queries = [
            ("Latest market NPL?", "Market NPL in 2024 is 15.2%, showing improvement from 2023."),
            ("Top 3 MFOs?", "Market leaders: Company A, Company B, Company C by assets."),
        ]
        
        for q, a in demo_queries:
            with st.expander(q):
                st.write(a)
    
    # TAB 4: Downloads
    with tab4:
        st.header("Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Market Aggregations")
            st.download_button(
                label="Download Market Stats",
                data=aggregations['yearly_market'].to_csv(index=False).encode('utf-8'),
                file_name='market_yearly.csv',
                mime='text/csv'
            )
        
        with col2:
            st.subheader("Top MFOs")
            st.download_button(
                label="Download Rankings",
                data=aggregations['top_mfos'].to_csv(index=False).encode('utf-8'),
                file_name='top_mfos.csv',
                mime='text/csv'
            )
        
        with col3:
            st.subheader("Raw Dataset")
            st.download_button(
                label="Download Full Data",
                data=raw_data.to_csv(index=False).encode('utf-8'),
                file_name='mfo_full_dataset.csv',
                mime='text/csv'
            )
        
        st.markdown("---")
        st.subheader("Data Preview")
        
        preview_option = st.selectbox(
            "Select dataset to preview:",
            ["Raw Data", "Yearly Market", "Top MFOs"]
        )
        
        if preview_option == "Raw Data":
            st.dataframe(raw_data.head(100), use_container_width=True)
        elif preview_option == "Yearly Market":
            st.dataframe(aggregations['yearly_market'], use_container_width=True)
        else:
            st.dataframe(aggregations['top_mfos'].head(20), use_container_width=True)


if __name__ == "__main__":
    main()