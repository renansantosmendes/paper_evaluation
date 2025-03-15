from langchain_core.runnables import (
    RunnablePassthrough, RunnableLambda, RunnableBranch, 
    RunnableParallel, RunnableSequence
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, Document
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_experimental.text_splitter import SemanticChunker
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Literal
import numpy as np
import re
import json
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM models
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_smart = ChatOpenAI(model="gpt-4", temperature=0.2)
embeddings_model = OpenAIEmbeddings()

# Define Pydantic schemas for structured outputs
class QueryAnalysis(BaseModel):
    original_query: str
    intent: Literal["factual", "conceptual", "procedural", "comparative"]
    complexity: Literal["low", "medium", "high"]
    domain_specificity: Literal["general", "domain_specific"]
    temporal_nature: Literal["current", "historical", "future", "timeless"]
    requires_structured_data: bool
    requires_context_from_history: bool
    reformulations: List[str] = Field(description="Alternative formulations of the query")
    entities: List[str] = Field(description="Key entities mentioned in the query")
    keywords: List[str] = Field(description="Important keywords for search")

class RetrievalStrategy(BaseModel):
    retrieval_type: Literal["semantic", "keyword", "hybrid", "multi_query"]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_mmr: bool
    use_compression: bool
    reranking_strategy: Optional[str] = None
    custom_preprocessing: Optional[str] = None

class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    timestamp: Optional[str] = None
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievedContext(BaseModel):
    chunks: List[str]
    sources: List[DocumentMetadata]
    combined_relevance_score: float
    strategy_used: str

class ResponseGeneration(BaseModel):
    response_type: Literal["direct_answer", "explanation", "step_by_step", "comparison", "clarification_needed"]
    tone: Literal["neutral", "technical", "simplified", "conversational"]
    confidence: float
    citations_needed: bool
    follow_up_suggestions: Optional[List[str]] = None

# Memory system
conversation_memory = ConversationBufferMemory(
    return_messages=True,
    output_key="response",
    input_key="query"
)

# Document Database Mock
sample_docs = [
    Document(page_content="Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. They are a series of algorithms that recognize underlying relationships in a set of data. Neural networks can adapt to changing inputs; so the network generates the best possible result without needing to redesign the output criteria.", 
             metadata={"source": "machine_learning_textbook.pdf", "page": 42, "category": "technical"}),
    Document(page_content="A Transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing. Like recurrent neural networks (RNNs), Transformers are designed to handle sequential data, such as natural language, for tasks such as translation and text summarization.", 
             metadata={"source": "nlp_advances.pdf", "page": 78, "category": "technical"}),
    Document(page_content="Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.", 
             metadata={"source": "optimization_algorithms.pdf", "page": 21, "category": "technical"}),
    Document(page_content="Python is a high-level, interpreted, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected.", 
             metadata={"source": "programming_languages.pdf", "page": 14, "category": "general"})
]

# Create vector database
vector_db = Chroma.from_documents(sample_docs, embeddings_model)
bm25_retriever = BM25Retriever.from_documents(sample_docs)
bm25_retriever.k = 3

# 1. QUERY UNDERSTANDING COMPONENT
query_analyzer_prompt = ChatPromptTemplate.from_template("""
You are an advanced query understanding system. Analyze the following query in detail:

USER QUERY: {query}

CONVERSATION HISTORY: {history}

Provide a detailed analysis including:
1. The intent behind the query (factual, conceptual, procedural, comparative)
2. Complexity level (low, medium, high)
3. Domain specificity (general or domain specific)
4. Temporal nature (current, historical, future, timeless)
5. Whether it requires structured data
6. Whether it requires context from conversation history
7. Generate 2-3 alternative formulations of the query
8. Extract key entities mentioned
9. Extract important keywords for search

Format your response as a JSON object.
""")

query_analyzer = (
    query_analyzer_prompt 
    | llm_smart 
    | JsonOutputParser(pydantic_model=QueryAnalysis)
)

# 2. RETRIEVAL STRATEGY SELECTOR
strategy_selector_prompt = ChatPromptTemplate.from_template("""
As a retrieval strategy expert, determine the optimal document retrieval approach based on this query analysis:

{query_analysis}

Consider:
- For factual/specific queries, prefer semantic search with smaller chunks
- For conceptual questions, use larger chunks with more overlap
- For procedural queries, consider hybrid retrieval
- For complex queries, consider multi-query retrieval
- For queries needing historical context, incorporate conversation history

Recommend specific parameters including:
- Retrieval type (semantic, keyword, hybrid, multi_query)
- Chunk size (in characters)
- Chunk overlap percentage
- Number of documents to retrieve (top_k)
- Whether to use Maximum Marginal Relevance (use_mmr)
- Whether to use context compression
- Any reranking strategy
- Any custom preprocessing needed

Return as JSON.
""")

strategy_selector = (
    strategy_selector_prompt 
    | llm_fast 
    | JsonOutputParser(pydantic_model=RetrievalStrategy)
)

# 3. DOCUMENT CHUNKING SYSTEM
def create_text_splitter(strategy: RetrievalStrategy):
    """Dynamically create text splitter based on strategy"""
    if strategy.custom_preprocessing == "semantic_chunking":
        return SemanticChunker(embeddings_model)
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=strategy.chunk_size,
            chunk_overlap=int(strategy.chunk_size * 0.1),  # 10% overlap by default
            separators=["\n\n", "\n", ". ", " ", ""]
        )

# 4. RETRIEVAL COMPONENTS

# 4.1 Semantic Search
def semantic_search(query: str, db, top_k: int) -> List[Document]:
    return db.similarity_search(query, k=top_k)

# 4.2 Keyword Search
def keyword_search(query: str, retriever, top_k: int) -> List[Document]:
    return retriever.get_relevant_documents(query)[:top_k]

# 4.3 Hybrid Search
def hybrid_search(query: str, vector_db, keyword_retriever, top_k: int) -> List[Document]:
    semantic_results = semantic_search(query, vector_db, top_k)
    keyword_results = keyword_search(query, keyword_retriever, top_k)
    
    # Combine and deduplicate
    all_results = semantic_results + keyword_results
    seen_content = set()
    unique_results = []
    
    for doc in all_results:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_results.append(doc)
            if len(unique_results) >= top_k:
                break
                
    return unique_results

# 4.4 Multi-query Retrieval
def setup_multi_query_retriever(query: str, db, top_k: int) -> List[Document]:
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": top_k}),
        llm=llm_fast
    )
    return retriever.get_relevant_documents(query)

# 4.5 Context Compression
def apply_context_compression(documents: List[Document], query: str) -> List[Document]:
    compressor = LLMChainExtractor.from_llm(llm_fast)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=lambda q: documents
    )
    return compression_retriever.get_relevant_documents(query)

# 5. DOCUMENT SELECTION AND PREPARATION
def prepare_retrieved_context(documents: List[Document], strategy: RetrievalStrategy, query: str) -> RetrievedContext:
    """Process retrieved documents into structured context"""
    chunks = [doc.page_content for doc in documents]
    
    sources = []
    for i, doc in enumerate(documents):
        relevance = 1.0 - (i * 0.1)  # Simple decay function for relevance
        if relevance < 0.3:
            relevance = 0.3  # Minimum relevance threshold
            
        source_meta = DocumentMetadata(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            timestamp=doc.metadata.get("timestamp", datetime.now().isoformat()),
            relevance_score=round(relevance, 2),
            metadata=doc.metadata
        )
        sources.append(source_meta)
    
    # Calculate combined relevance
    combined_score = sum(s.relevance_score for s in sources) / len(sources) if sources else 0
    
    return RetrievedContext(
        chunks=chunks,
        sources=sources,
        combined_relevance_score=round(combined_score, 2),
        strategy_used=strategy.retrieval_type
    )

# 6. RESPONSE PLANNING
response_planner_prompt = ChatPromptTemplate.from_template("""
Based on the query analysis and retrieved information, plan the optimal response strategy:

QUERY ANALYSIS: {query_analysis}

RETRIEVED CONTEXT:
{retrieved_context}

Determine:
1. What type of response would be most appropriate (direct_answer, explanation, step_by_step, comparison, or clarification_needed)
2. What tone should be used (neutral, technical, simplified, conversational)
3. How confident we can be in answering based on retrieved information (0.0-1.0)
4. Whether citations should be included
5. Any relevant follow-up questions that might be helpful

Return your analysis as JSON.
""")

response_planner = (
    response_planner_prompt 
    | llm_fast 
    | JsonOutputParser(pydantic_model=ResponseGeneration)
)

# 7. RESPONSE GENERATION
response_generator_prompt = ChatPromptTemplate.from_template("""
Generate a comprehensive response to the user's query.

USER QUERY: {query}

CONVERSATION HISTORY: {history}

QUERY ANALYSIS: {query_analysis}

RETRIEVED CONTEXT:
{retrieved_context}

RESPONSE STRATEGY:
{response_plan}

Based on this information, provide a {response_plan.response_type} response with a {response_plan.tone} tone.
{citations_instruction}

USER QUERY: {query}
YOUR RESPONSE:
""")

def get_citations_instruction(response_plan):
    if response_plan.citations_needed:
        return "Include citations to sources where appropriate using [Source: name, page]."
    return "No need to include formal citations."

# 8. MAIN ORCHESTRATION RUNNABLE

# 8.1 Get conversation history
def get_conversation_history(inputs):
    messages = conversation_memory.chat_memory.messages
    return {"history": messages}

# 8.2 Run query analysis
def run_query_analysis(inputs):
    query = inputs["query"]
    history = inputs["history"]
    analysis = query_analyzer.invoke({"query": query, "history": history})
    return {"query_analysis": analysis}

# 8.3 Select retrieval strategy
def select_retrieval_strategy(inputs):
    query_analysis = inputs["query_analysis"]
    strategy = strategy_selector.invoke({"query_analysis": query_analysis})
    return {"retrieval_strategy": strategy}

# 8.4 Document retrieval
def retrieve_documents(inputs):
    query = inputs["query"]
    query_analysis = inputs["query_analysis"]
    strategy = inputs["retrieval_strategy"]
    
    # Select retrieval method based on strategy
    if strategy.retrieval_type == "semantic":
        docs = semantic_search(query, vector_db, strategy.top_k)
    elif strategy.retrieval_type == "keyword":
        docs = keyword_search(query, bm25_retriever, strategy.top_k)
    elif strategy.retrieval_type == "hybrid":
        docs = hybrid_search(query, vector_db, bm25_retriever, strategy.top_k)
    elif strategy.retrieval_type == "multi_query":
        docs = setup_multi_query_retriever(query, vector_db, strategy.top_k)
    else:
        # Default to semantic
        docs = semantic_search(query, vector_db, 3)
    
    # Apply compression if specified
    if strategy.use_compression:
        docs = apply_context_compression(docs, query)
        
    # Prepare context
    context = prepare_retrieved_context(docs, strategy, query)
    return {"retrieved_context": context}

# 8.5 Plan response
def plan_response(inputs):
    plan = response_planner.invoke({
        "query_analysis": inputs["query_analysis"],
        "retrieved_context": inputs["retrieved_context"]
    })
    
    citations_instruction = get_citations_instruction(plan)
    return {"response_plan": plan, "citations_instruction": citations_instruction}

# 8.6 Generate response
def generate_response(inputs):
    # Select appropriate LLM based on complexity
    if inputs["query_analysis"].complexity == "high" or inputs["retrieved_context"].combined_relevance_score < 0.5:
        generation_llm = llm_smart
    else:
        generation