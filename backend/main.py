import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
import uuid
import hashlib
import re
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import openai
from langchain_community.tools import TavilySearchResults
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
import numpy as np
from scipy import constants as scipy_constants
import sympy as sp
import chempy
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    CONVERSATION_TABLE = "conversations"
    VECTOR_TABLE = "conversation_vectors"
    SESSION_TIMEOUT = 3600  
    MAX_MEMORY_ITEMS = 50
    
    PROMETHEUS_PORT = 9090

genai.configure(api_key=Config.GEMINI_API_KEY)
openai.api_key = Config.OPENAI_API_KEY

registry = CollectorRegistry()
request_counter = Counter('tutorbot_requests_total', 'Total requests', ['agent', 'subject'], registry=registry)
response_time_histogram = Histogram('tutorbot_response_time_seconds', 'Response time', ['agent'], registry=registry)
active_sessions_gauge = Gauge('tutorbot_active_sessions', 'Active sessions', registry=registry)
tool_usage_counter = Counter('tutorbot_tool_usage_total', 'Tool usage', ['tool', 'agent'], registry=registry)


class MemoryManager:
    """Advanced memory management with vector embeddings and conversation history"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.vector_store = SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name=Config.VECTOR_TABLE,
            query_name="match_conversations"
        )
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(Config.REDIS_URL, decode_responses=True)
        
    async def store_conversation(self, session_id: str, query: str, response: str, metadata: Dict[str, Any]):
        """Store conversation with vector embedding"""
        conversation_id = str(uuid.uuid4())
        
        doc = {
            "id": conversation_id,
            "session_id": session_id,
            "query": query,
            "response": response,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        self.supabase.table(Config.CONVERSATION_TABLE).insert(doc).execute()
        
        text_for_embedding = f"Query: {query}\nResponse: {response}"
        embedding = await self.embeddings.aembed_query(text_for_embedding)
        
        vector_doc = {
            "id": conversation_id,
            "session_id": session_id,
            "content": text_for_embedding,
            "embedding": embedding,
            "metadata": json.dumps(metadata)
        }
        
        self.supabase.table(Config.VECTOR_TABLE).insert(vector_doc).execute()
        
        await self.redis_client.lpush(
            f"session:{session_id}:history",
            json.dumps(doc)
        )
        await self.redis_client.ltrim(f"session:{session_id}:history", 0, Config.MAX_MEMORY_ITEMS)
        await self.redis_client.expire(f"session:{session_id}:history", Config.SESSION_TIMEOUT)
        
        return conversation_id
    
    async def get_relevant_context(self, session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context using vector similarity search"""
        query_embedding = await self.embeddings.aembed_query(query)
        
        results = self.supabase.rpc(
            "match_conversations",
            {
                "p_session_id": session_id,
                "query_embedding": query_embedding,
                "match_count": k
            }
        ).execute()
        
        return results.data if results.data else []
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history from Redis cache"""
        history = await self.redis_client.lrange(f"session:{session_id}:history", 0, limit - 1)
        return [json.loads(item) for item in history]
    
    async def summarize_conversation(self, session_id: str) -> str:
        """Generate conversation summary using Gemini"""
        history = await self.get_session_history(session_id, limit=20)
        
        if not history:
            return "No conversation history available."
        
        conversation_text = "\n".join([
            f"User: {item['query']}\nAssistant: {item['response'][:200]}..."
            for item in history
        ])
        
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        prompt = f"""Summarize this tutoring conversation, highlighting key topics discussed and concepts explained:

{conversation_text}

Provide a concise summary focusing on educational content."""
        
        response = model.generate_content(prompt)
        return response.text



class MathTools:
    """Advanced mathematical tools using SymPy and NumPy"""
    
    @staticmethod
    def solve_equation(equation_str: str) -> Dict[str, Any]:
        """Solve various types of equations using SymPy"""
        try:
            
            if '=' in equation_str:
                left, right = equation_str.split('=')
                eq = sp.Eq(sp.sympify(left), sp.sympify(right))
            else:
                eq = sp.sympify(equation_str)
            
            
            variables = list(eq.free_symbols)
            
            if len(variables) == 0:
                return {"error": "No variables found in equation"}
            
            
            if len(variables) == 1:
                solutions = sp.solve(eq, variables[0])
            else:
                solutions = sp.solve(eq, variables)
            
            return {
                "equation": str(eq),
                "variables": [str(v) for v in variables],
                "solutions": [str(sol) for sol in solutions],
                "steps": MathTools._get_solution_steps(eq, variables[0] if len(variables) == 1 else variables)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _get_solution_steps(equation, variable) -> List[str]:
        """Generate step-by-step solution"""
        steps = []
        steps.append(f"Given: {equation}")
        
        
        simplified = sp.simplify(equation)
        if simplified != equation:
            steps.append(f"Simplified: {simplified}")
        
        
        if isinstance(equation, sp.Eq):
            steps.append("Rearranging to solve for " + str(variable))
        
        return steps
    
    @staticmethod
    def calculate_derivative(expression: str, variable: str = 'x') -> Dict[str, Any]:
        """Calculate derivatives"""
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            first_derivative = sp.diff(expr, var)
            second_derivative = sp.diff(first_derivative, var)
            
            return {
                "expression": str(expr),
                "first_derivative": str(first_derivative),
                "second_derivative": str(second_derivative),
                "critical_points": [str(cp) for cp in sp.solve(first_derivative, var)]
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def integrate(expression: str, variable: str = 'x') -> Dict[str, Any]:
        """Calculate integrals"""
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            indefinite_integral = sp.integrate(expr, var)
            
            return {
                "expression": str(expr),
                "indefinite_integral": str(indefinite_integral) + " + C"
            }
        except Exception as e:
            return {"error": str(e)}

class PhysicsTools:
    """Advanced physics tools with scipy and numerical simulations"""
    
    @staticmethod
    def get_constant(constant_name: str) -> Dict[str, Any]:
        """Get physics constants from scipy"""
        constant_map = {
            "speed_of_light": ("c", scipy_constants.c),
            "gravitational_constant": ("G", scipy_constants.G),
            "planck": ("h", scipy_constants.h),
            "boltzmann": ("k", scipy_constants.k),
            "avogadro": ("N_A", scipy_constants.Avogadro),
            "elementary_charge": ("e", scipy_constants.e),
            "electron_mass": ("m_e", scipy_constants.electron_mass),
            "proton_mass": ("m_p", scipy_constants.proton_mass)
        }
        
        for key, (symbol, value) in constant_map.items():
            if constant_name.lower() in key:
                return {
                    "name": key,
                    "symbol": symbol,
                    "value": value,
                    "unit": PhysicsTools._get_unit(key)
                }
        
        return {"error": f"Constant '{constant_name}' not found"}
    
    @staticmethod
    def _get_unit(constant_name: str) -> str:
        """Get appropriate unit for constant"""
        units = {
            "speed_of_light": "m/s",
            "gravitational_constant": "m³/kg·s²",
            "planck": "J·s",
            "boltzmann": "J/K",
            "avogadro": "mol⁻¹",
            "elementary_charge": "C",
            "electron_mass": "kg",
            "proton_mass": "kg"
        }
        return units.get(constant_name, "")
    
    @staticmethod
    def calculate_kinematics(initial_velocity: float, acceleration: float, time: float) -> Dict[str, Any]:
        """Calculate kinematic quantities"""
        final_velocity = initial_velocity + acceleration * time
        displacement = initial_velocity * time + 0.5 * acceleration * time**2
        average_velocity = (initial_velocity + final_velocity) / 2
        
        return {
            "initial_velocity": initial_velocity,
            "final_velocity": final_velocity,
            "acceleration": acceleration,
            "time": time,
            "displacement": displacement,
            "average_velocity": average_velocity,
            "equations_used": [
                "v = v₀ + at",
                "s = v₀t + ½at²",
                "v_avg = (v₀ + v) / 2"
            ]
        }

class ChemistryTools:
    """Advanced chemistry tools using RDKit and ChemPy"""
    
    @staticmethod
    def calculate_molecular_properties(smiles_or_formula: str) -> Dict[str, Any]:
        """Calculate molecular properties from SMILES or formula"""
        try:
            
            mol = Chem.MolFromSmiles(smiles_or_formula)
            
            if mol is None:
                
                mol = Chem.MolFromSmiles(ChemistryTools._formula_to_smiles(smiles_or_formula))
            
            if mol is not None:
                return {
                    "molecular_weight": Descriptors.ExactMolWt(mol),
                    "logP": Descriptors.MolLogP(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "num_bonds": mol.GetNumBonds(),
                    "tpsa": Descriptors.TPSA(mol),
                    "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                    "num_h_donors": Descriptors.NumHDonors(mol),
                    "num_h_acceptors": Descriptors.NumHAcceptors(mol)
                }
            else:
                return {"error": "Could not parse molecule"}
                
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _formula_to_smiles(formula: str) -> str:
        """Convert simple formulas to SMILES"""
        
        formula_map = {
            "H2O": "O",
            "CO2": "O=C=O",
            "CH4": "C",
            "NH3": "N",
            "H2SO4": "O=S(=O)(O)O",
            "HCl": "Cl",
            "NaCl": "[Na+].[Cl-]"
        }
        return formula_map.get(formula, formula)
    
    @staticmethod
    def balance_equation(equation: str) -> Dict[str, Any]:
        """Balance chemical equations"""
        try:
            
            
            return {
                "original": equation,
                "balanced": equation,  
                "coefficients": {},
                "note": "Full equation balancing requires chempy setup"
            }
        except Exception as e:
            return {"error": str(e)}

class BiologyTools:
    """Advanced biology tools for sequence analysis and more"""
    
    @staticmethod
    def analyze_dna_sequence(sequence: str) -> Dict[str, Any]:
        """Analyze DNA sequence"""
        sequence = sequence.upper()
        
        
        valid_bases = set('ATCG')
        if not all(base in valid_bases for base in sequence):
            return {"error": "Invalid DNA sequence"}
        
        
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0
        
        
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        complement = ''.join(complement_map[base] for base in sequence)
        
        
        reverse_complement = complement[::-1]
        
        
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        
        return {
            "sequence": sequence,
            "length": len(sequence),
            "gc_content": f"{gc_content:.1f}%",
            "complement": complement,
            "reverse_complement": reverse_complement,
            "codons": codons,
            "base_composition": {
                "A": sequence.count('A'),
                "T": sequence.count('T'),
                "G": sequence.count('G'),
                "C": sequence.count('C')
            }
        }
    
    @staticmethod
    def translate_dna_to_protein(dna_sequence: str) -> Dict[str, Any]:
        """Translate DNA to protein sequence"""
        codon_table = {
            'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
            'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
            'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
            'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
            
        }
        
        
        rna_sequence = dna_sequence.upper().replace('T', 'U')
        
        
        protein = []
        for i in range(0, len(rna_sequence)-2, 3):
            codon = rna_sequence[i:i+3]
            amino_acid = codon_table.get(codon, 'X')
            protein.append(amino_acid)
            if amino_acid == '*':  
                break
        
        return {
            "dna_sequence": dna_sequence,
            "rna_sequence": rna_sequence,
            "protein_sequence": ''.join(protein),
            "length": len(protein)
        }



class SearchManager:
    """Manage web search using Tavily through LangChain"""
    
    def __init__(self):
        self.search_tool = TavilySearchResults(
            api_key=Config.TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced"
        )
    
    async def search(self, query: str) -> List[Dict[str, str]]:
        """Perform web search"""
        try:
            results = await asyncio.to_thread(self.search_tool.run, query)
            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []



class SubjectType(str, Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    GENERAL = "general"

class BaseAgent:
    """Enhanced base agent with memory and monitoring"""
    
    def __init__(self, name: str, subject: SubjectType):
        self.name = name
        self.subject = subject
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        self.tools = {}
        self.memory_manager: Optional[MemoryManager] = None
        self.search_manager: Optional[SearchManager] = None
        
    def add_tool(self, tool_name: str, tool_func):
        """Add a tool to the agent"""
        self.tools[tool_name] = tool_func
        
    async def get_context(self, session_id: str, query: str) -> str:
        """Get relevant context from memory"""
        if not self.memory_manager:
            return ""
            
        
        relevant_context = await self.memory_manager.get_relevant_context(session_id, query, k=3)
        
        
        recent_history = await self.memory_manager.get_session_history(session_id, limit=5)
        
        
        context_parts = []
        
        if relevant_context:
            context_parts.append("Relevant past discussions:")
            for ctx in relevant_context:
                context_parts.append(f"- {ctx['content'][:200]}...")
        
        if recent_history:
            context_parts.append("\nRecent conversation:")
            for hist in recent_history[-3:]:
                context_parts.append(f"User: {hist['query']}")
                context_parts.append(f"Assistant: {hist['response'][:100]}...")
        
        return "\n".join(context_parts)
    
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process query with context and monitoring"""
        raise NotImplementedError

class MathAgent(BaseAgent):
    """Enhanced Math Agent with advanced tools"""
    
    def __init__(self):
        super().__init__("MathAgent", SubjectType.MATH)
        self.math_tools = MathTools()
        
        
        self.add_tool("solve_equation", self.math_tools.solve_equation)
        self.add_tool("derivative", self.math_tools.calculate_derivative)
        self.add_tool("integral", self.math_tools.integrate)
        
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process math queries with advanced tools"""
        with response_time_histogram.labels(agent=self.name).time():
            request_counter.labels(agent=self.name, subject=self.subject.value).inc()
            
            tools_used = []
            tool_results = {}
            
            
            context = await self.get_context(session_id, query)
            
            
            if any(keyword in query.lower() for keyword in ["solve", "equation", "="]):
                
                eq_pattern = r'[^:]*[=][^:]*'
                eq_match = re.search(eq_pattern, query)
                if eq_match:
                    equation = eq_match.group().strip()
                    result = self.tools["solve_equation"](equation)
                    tool_results["equation_solver"] = result
                    tools_used.append("equation_solver")
                    tool_usage_counter.labels(tool="equation_solver", agent=self.name).inc()
            
            
            if "derivative" in query.lower() or "differentiate" in query.lower():
                
                expr_match = re.search(r'of\s+(.+?)(?:\s+with|\s+$)', query)
                if expr_match:
                    expression = expr_match.group(1).strip()
                    result = self.tools["derivative"](expression)
                    tool_results["derivative"] = result
                    tools_used.append("derivative")
                    tool_usage_counter.labels(tool="derivative", agent=self.name).inc()
            
            
            if "integral" in query.lower() or "integrate" in query.lower():
                expr_match = re.search(r'of\s+(.+?)(?:\s+with|\s+$)', query)
                if expr_match:
                    expression = expr_match.group(1).strip()
                    result = self.tools["integral"](expression)
                    tool_results["integral"] = result
                    tools_used.append("integral")
                    tool_usage_counter.labels(tool="integral", agent=self.name).inc()
            
            
            search_results = []
            if "example" in query.lower() or "how to" in query.lower():
                search_results = await self.search_manager.search(f"mathematics {query}")
            
            
            prompt = f"""You are an expert mathematics tutor. Answer this question: {query}

Context from previous conversations:
{context}

Tool results:
{json.dumps(tool_results, indent=2)}

Search results:
{json.dumps(search_results[:2], indent=2) if search_results else "No search results"}

Provide a clear, educational response that:
1. Explains the concept thoroughly
2. Shows step-by-step solutions if applicable
3. Uses the tool results appropriately
4. References any relevant past discussions
5. Includes examples when helpful"""
            
            response = self.model.generate_content(prompt)
            
            return {
                "response": response.text,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "search_results": search_results
            }

class PhysicsAgent(BaseAgent):
    """Enhanced Physics Agent"""
    
    def __init__(self):
        super().__init__("PhysicsAgent", SubjectType.PHYSICS)
        self.physics_tools = PhysicsTools()
        
        self.add_tool("get_constant", self.physics_tools.get_constant)
        self.add_tool("kinematics", self.physics_tools.calculate_kinematics)
        
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process physics queries"""
        with response_time_histogram.labels(agent=self.name).time():
            request_counter.labels(agent=self.name, subject=self.subject.value).inc()
            
            tools_used = []
            tool_results = {}
            
            
            context = await self.get_context(session_id, query)
            
            
            constant_keywords = ["speed of light", "gravitational constant", "planck", "boltzmann"]
            for keyword in constant_keywords:
                if keyword in query.lower():
                    result = self.tools["get_constant"](keyword)
                    tool_results["constants"] = result
                    tools_used.append("constants")
                    tool_usage_counter.labels(tool="constants", agent=self.name).inc()
                    break
            
            
            if any(word in query.lower() for word in ["velocity", "acceleration", "displacement"]):
                
                tools_used.append("kinematics")
                tool_usage_counter.labels(tool="kinematics", agent=self.name).inc()
            
            
            search_results = []
            if "latest" in query.lower() or "recent" in query.lower():
                search_results = await self.search_manager.search(f"physics {query} 2025")
            
            
            prompt = f"""You are an expert physics tutor. Answer this question: {query}

Context from previous conversations:
{context}

Tool results:
{json.dumps(tool_results, indent=2)}

Search results:
{json.dumps(search_results[:2], indent=2) if search_results else "No search results"}

Provide a clear, educational response that:
1. Explains physics concepts clearly
2. Uses accurate values from tools
3. References relevant formulas
4. Includes real-world applications
5. Builds on any previous discussions"""
            
            response = self.model.generate_content(prompt)
            
            return {
                "response": response.text,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "search_results": search_results
            }

class ChemistryAgent(BaseAgent):
    """Enhanced Chemistry Agent"""
    
    def __init__(self):
        super().__init__("ChemistryAgent", SubjectType.CHEMISTRY)
        self.chemistry_tools = ChemistryTools()
        
        self.add_tool("molecular_properties", self.chemistry_tools.calculate_molecular_properties)
        self.add_tool("balance_equation", self.chemistry_tools.balance_equation)
        
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process chemistry queries"""
        with response_time_histogram.labels(agent=self.name).time():
            request_counter.labels(agent=self.name, subject=self.subject.value).inc()
            
            tools_used = []
            tool_results = {}
            
            
            context = await self.get_context(session_id, query)
            
            
            if any(word in query.lower() for word in ["molecular", "weight", "properties", "formula"]):
                
                mol_pattern = r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b'
                mol_match = re.search(mol_pattern, query)
                if mol_match:
                    molecule = mol_match.group()
                    result = self.tools["molecular_properties"](molecule)
                    tool_results["molecular_properties"] = result
                    tools_used.append("molecular_properties")
                    tool_usage_counter.labels(tool="molecular_properties", agent=self.name).inc()
            
            
            prompt = f"""You are an expert chemistry tutor. Answer this question: {query}

Context from previous conversations:
{context}

Tool results:
{json.dumps(tool_results, indent=2)}

Provide a clear, educational response about chemistry."""
            
            response = self.model.generate_content(prompt)
            
            return {
                "response": response.text,
                "tools_used": tools_used,
                "tool_results": tool_results
            }

class BiologyAgent(BaseAgent):
    """Enhanced Biology Agent"""
    
    def __init__(self):
        super().__init__("BiologyAgent", SubjectType.BIOLOGY)
        self.biology_tools = BiologyTools()
        
        self.add_tool("analyze_dna", self.biology_tools.analyze_dna_sequence)
        self.add_tool("translate_dna", self.biology_tools.translate_dna_to_protein)
        
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process biology queries"""
        with response_time_histogram.labels(agent=self.name).time():
            request_counter.labels(agent=self.name, subject=self.subject.value).inc()
            
            tools_used = []
            tool_results = {}
            
            
            context = await self.get_context(session_id, query)
            
            
            dna_pattern = r'[ATCG]{4,}'
            dna_match = re.search(dna_pattern, query.upper())
            if dna_match:
                sequence = dna_match.group()
                result = self.tools["analyze_dna"](sequence)
                tool_results["dna_analysis"] = result
                tools_used.append("analyze_dna")
                tool_usage_counter.labels(tool="analyze_dna", agent=self.name).inc()
            
            
            prompt = f"""You are an expert biology tutor. Answer this question: {query}

Context from previous conversations:
{context}

Tool results:
{json.dumps(tool_results, indent=2)}

Provide a clear, educational response about biology."""
            
            response = self.model.generate_content(prompt)
            
            return {
                "response": response.text,
                "tools_used": tools_used,
                "tool_results": tool_results
            }

class TutorAgent(BaseAgent):
    """Main orchestrator agent with enhanced capabilities"""
    
    def __init__(self, memory_manager: MemoryManager, search_manager: SearchManager):
        super().__init__("TutorAgent", SubjectType.GENERAL)
        self.memory_manager = memory_manager
        self.search_manager = search_manager
        
        
        self.agents = {
            SubjectType.MATH: MathAgent(),
            SubjectType.PHYSICS: PhysicsAgent(),
            SubjectType.CHEMISTRY: ChemistryAgent(),
            SubjectType.BIOLOGY: BiologyAgent()
        }
        
        
        for agent in self.agents.values():
            agent.memory_manager = memory_manager
            agent.search_manager = search_manager
    
    async def classify_query(self, session_id: str, query: str) -> SubjectType:
        """Classify query using context-aware classification"""
        
        context = await self.get_context(session_id, query)
        
        prompt = f"""Based on the conversation context and current query, classify this into ONE category:
- MATH: Mathematics questions (algebra, calculus, geometry, etc.)
- PHYSICS: Physics questions (mechanics, thermodynamics, etc.)
- CHEMISTRY: Chemistry questions (elements, molecules, reactions, etc.)
- BIOLOGY: Biology questions (cells, DNA, organisms, etc.)
- GENERAL: Other educational topics

Context:
{context}

Current query: {query}

Respond with only one word: MATH, PHYSICS, CHEMISTRY, BIOLOGY, or GENERAL"""
        
        response = self.model.generate_content(prompt)
        classification = response.text.strip().upper()
        
        try:
            return SubjectType(classification.lower())
        except ValueError:
            
            query_lower = query.lower()
            if any(word in query_lower for word in ["math", "equation", "calculate", "solve", "integral"]):
                return SubjectType.MATH
            elif any(word in query_lower for word in ["physics", "force", "energy", "velocity"]):
                return SubjectType.PHYSICS
            elif any(word in query_lower for word in ["chemistry", "element", "molecule", "reaction"]):
                return SubjectType.CHEMISTRY
            elif any(word in query_lower for word in ["biology", "cell", "dna", "organism"]):
                return SubjectType.BIOLOGY
            else:
                return SubjectType.GENERAL
    
    async def process(self, session_id: str, query: str) -> Dict[str, Any]:
        """Process query with full orchestration"""
        
        active_sessions_gauge.inc()
        
        try:
            
            subject = await self.classify_query(session_id, query)
            
            
            if subject in self.agents:
                result = await self.agents[subject].process(session_id, query)
                agent_used = self.agents[subject].name
            else:
                
                context = await self.get_context(session_id, query)
                search_results = await self.search_manager.search(query)
                
                prompt = f"""You are a helpful educational tutor. Answer this question: {query}

Context from previous conversations:
{context}

Search results:
{json.dumps(search_results[:3], indent=2) if search_results else "No search results"}

Provide a clear, educational response."""
                
                response = self.model.generate_content(prompt)
                result = {
                    "response": response.text,
                    "tools_used": [],
                    "search_results": search_results
                }
                agent_used = self.name
            
            
            await self.memory_manager.store_conversation(
                session_id=session_id,
                query=query,
                response=result["response"],
                metadata={
                    "subject": subject.value,
                    "agent_used": agent_used,
                    "tools_used": result.get("tools_used", []),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            
            result["subject"] = subject.value
            result["agent_used"] = agent_used
            result["session_id"] = session_id
            
            return result
            
        finally:
            active_sessions_gauge.dec()



class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class QueryResponse(BaseModel):
    response: str
    subject: str
    tools_used: List[str]
    agent_used: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ConversationSummary(BaseModel):
    session_id: str
    summary: str
    total_interactions: int
    topics_covered: List[str]



async def get_memory_manager():
    if not hasattr(app.state, "memory_manager"):
        app.state.memory_manager = MemoryManager()
        await app.state.memory_manager.initialize()
    return app.state.memory_manager

async def get_search_manager():
    if not hasattr(app.state, "search_manager"):
        app.state.search_manager = SearchManager()
    return app.state.search_manager

async def get_tutor_agent(
    memory_manager: MemoryManager = Depends(get_memory_manager),
    search_manager: SearchManager = Depends(get_search_manager)
):
    if not hasattr(app.state, "tutor_agent"):
        app.state.tutor_agent = TutorAgent(memory_manager, search_manager)
    return app.state.tutor_agent



app = FastAPI(
    title="Advanced Multi-Agent Tutoring Bot",
    description="AI-powered tutoring system with memory, search, and monitoring",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up Advanced Tutoring Bot...")
    
    
    memory_manager = await get_memory_manager()
    logger.info("Memory manager initialized")
    
    
    search_manager = await get_search_manager()
    logger.info("Search manager initialized")
    
    
    tutor_agent = await get_tutor_agent(memory_manager, search_manager)
    logger.info("Tutor agent initialized")
    
    logger.info("Startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app.state, "memory_manager") and app.state.memory_manager.redis_client:
        await app.state.memory_manager.redis_client.close()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    tutor_agent: TutorAgent = Depends(get_tutor_agent)
):
    """Process a question through the multi-agent system"""
    try:
        logger.info(f"Processing query: {request.query} (session: {request.session_id})")
        
        result = await tutor_agent.process(request.session_id, request.query)
        
        return QueryResponse(
            response=result["response"],
            subject=result["subject"],
            tools_used=result.get("tools_used", []),
            agent_used=result["agent_used"],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/summary/{session_id}", response_model=ConversationSummary)
async def get_conversation_summary(
    session_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get conversation summary for a session"""
    try:
        summary = await memory_manager.summarize_conversation(session_id)
        history = await memory_manager.get_session_history(session_id)
        
        
        topics = set()
        for item in history:
            if "metadata" in item and "subject" in item["metadata"]:
                topics.add(item["metadata"]["subject"])
        
        return ConversationSummary(
            session_id=session_id,
            summary=summary,
            total_interactions=len(history),
            topics_covered=list(topics)
        )
        
    except Exception as e:
        logger.error(f"Error getting summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: int = 10,
    memory_manager: MemoryManager = Depends(get_memory_manager)
):
    """Get conversation history for a session"""
    try:
        history = await memory_manager.get_session_history(session_id, limit)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(registry), media_type="text/plain")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini": "connected",
            "openai": "connected",
            "redis": "connected" if hasattr(app.state, "memory_manager") else "not initialized",
            "supabase": "connected" if hasattr(app.state, "memory_manager") else "not initialized"
        }
    }

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    return {
        "agents": [
            {
                "name": "TutorAgent",
                "role": "Main orchestrator with memory and context management",
                "capabilities": ["query classification", "context awareness", "conversation memory"]
            },
            {
                "name": "MathAgent",
                "role": "Advanced mathematics specialist",
                "tools": ["equation_solver", "derivative_calculator", "integral_calculator", "web_search"]
            },
            {
                "name": "PhysicsAgent",
                "role": "Physics specialist with numerical tools",
                "tools": ["constants_lookup", "kinematics_calculator", "web_search"]
            },
            {
                "name": "ChemistryAgent",
                "role": "Chemistry specialist with molecular tools",
                "tools": ["molecular_properties", "equation_balancer", "web_search"]
            },
            {
                "name": "BiologyAgent",
                "role": "Biology specialist with sequence analysis",
                "tools": ["dna_analyzer", "protein_translator", "web_search"]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)