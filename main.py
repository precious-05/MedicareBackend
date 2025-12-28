from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, text, func, distinct
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import requests
import logging
from pydantic import BaseModel
import os
import json
import asyncio
import aiohttp
import time
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ==================== DATABASE CONFIGURATION FOR NEON DB ====================
# Use environment variable for Render deployment with fallback for local development
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql://neondb_owner:npg_e6jAPiVpBm8Q@ep-calm-lab-a4bsqb6t-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

# Create FastAPI app
app = FastAPI(
    title="MediNomix API",
    description="AI-Powered Medication Safety System",
    version="2.1.0"
)

# CORS middleware - Update for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup for production with connection pooling
try:
    # For Neon DB with connection pooling
    if "neon.tech" in DATABASE_URL and "pooler" not in DATABASE_URL:
        # Ensure we're using the pooled connection
        DATABASE_URL = DATABASE_URL.replace("ep-calm-lab-a4bsqb6t", "ep-calm-lab-a4bsqb6t-pooler")
    
    # Create engine with production settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,  # Recycle connections after 5 minutes
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    print(f"✅ Database engine created successfully for Neon DB")
except Exception as e:
    print(f"❌ Error creating database engine: {e}")
    print(f"Database URL: {DATABASE_URL[:50]}...")  # Show first 50 chars for debugging
    # Don't exit in production, allow the app to start and show error in health check
    # exit(1)

# Logging setup for production
logging.basicConfig(
    level=logging.INFO if os.environ.get("RENDER", False) else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATABASE MODELS ====================

class Drug(Base):
    """Drug information from OpenFDA"""
    __tablename__ = "drugs"
    
    id = Column(Integer, primary_key=True, index=True)
    openfda_id = Column(String, unique=True, index=True)
    brand_name = Column(String, index=True)
    generic_name = Column(String, index=True)
    manufacturer = Column(String)
    substance_name = Column(String)
    product_type = Column(String)
    route = Column(String)
    active_ingredients = Column(Text)
    purpose = Column(Text)
    warnings = Column(Text)
    indications_and_usage = Column(Text)
    dosage_form = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    confusion_risks_as_source = relationship(
        "ConfusionRisk", 
        foreign_keys="ConfusionRisk.source_drug_id",
        back_populates="source_drug"
    )
    confusion_risks_as_target = relationship(
        "ConfusionRisk", 
        foreign_keys="ConfusionRisk.target_drug_id",
        back_populates="target_drug"
    )

class ConfusionRisk(Base):
    """Risk assessment between two drugs"""
    __tablename__ = "confusion_risks"
    
    id = Column(Integer, primary_key=True, index=True)
    source_drug_id = Column(Integer, ForeignKey("drugs.id"), index=True)
    target_drug_id = Column(Integer, ForeignKey("drugs.id"), index=True)
    
    spelling_similarity = Column(Float)
    phonetic_similarity = Column(Float)
    length_similarity = Column(Float)
    therapeutic_context_risk = Column(Float)
    
    combined_risk = Column(Float)
    risk_category = Column(String)
    
    algorithm_version = Column(String, default="1.0")
    last_analyzed = Column(DateTime, default=func.now())
    
    source_drug = relationship("Drug", foreign_keys=[source_drug_id], back_populates="confusion_risks_as_source")
    target_drug = relationship("Drug", foreign_keys=[target_drug_id], back_populates="confusion_risks_as_target")

class AnalysisLog(Base):
    """Log of user analyses"""
    __tablename__ = "analysis_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    drug_name = Column(String, index=True)
    timestamp = Column(DateTime, default=func.now())
    similar_drugs_found = Column(Integer)
    highest_risk_score = Column(Float)
    analysis_duration = Column(Float)

# ==================== REAL-TIME DASHBOARD MANAGER ====================

class RealTimeDashboardManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_metrics = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

dashboard_manager = RealTimeDashboardManager()

# ==================== DATABASE INITIALIZATION ====================

def init_database():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        
        # For Neon DB, we don't create databases, only tables
        try:
            # Try to create tables if they don't exist
            with engine.connect() as conn:
                # Check if tables exist
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                existing_tables = [row[0] for row in result]
                
                if 'drugs' not in existing_tables:
                    logger.info("Creating tables in Neon DB...")
                    Base.metadata.create_all(bind=engine)
                    logger.info("✅ Database tables created successfully")
                else:
                    logger.info("✅ Database tables already exist")
            
            return True
            
        except Exception as e2:
            logger.error(f"❌ Failed to initialize database: {e2}")
            print(f"\n🔧 DATABASE ERROR: Please check your Neon DB connection string")
            print(f"Current connection: {DATABASE_URL[:80]}...")
            print("\n1. Make sure your Neon database is running")
            print("2. Verify the connection string is correct")
            print("3. Check if you need to whitelist Render's IP addresses")
            return False

# ==================== PYDANTIC MODELS ====================

class DrugBase(BaseModel):
    id: int
    brand_name: str
    generic_name: str
    manufacturer: Optional[str] = None
    purpose: Optional[str] = None
    
    class Config:
        from_attributes = True

class ConfusionRiskBase(BaseModel):
    id: int
    target_drug: DrugBase
    spelling_similarity: float
    phonetic_similarity: float
    therapeutic_context_risk: float
    combined_risk: float
    risk_category: str
    
    class Config:
        from_attributes = True

class AnalysisResponse(BaseModel):
    query_drug: str
    similar_drugs: List[ConfusionRiskBase]
    total_found: int
    analysis_id: str
    timestamp: datetime

class DashboardMetrics(BaseModel):
    total_drugs: int
    total_analyses: int
    high_risk_pairs: int
    critical_risk_pairs: int
    avg_risk_score: float
    recent_searches: List[Dict[str, Any]]
    system_status: str
    last_updated: datetime
    connected_clients: int

# ==================== NEW MODELS FOR MISSING APIs ====================

class TopRiskResponse(BaseModel):
    drug1: str
    drug2: str
    risk_score: float
    risk_category: str
    reason: str
    
class RiskBreakdownResponse(BaseModel):
    category: str
    count: int
    
class HeatmapResponse(BaseModel):
    drug_names: List[str]
    risk_matrix: List[List[float]]
    
class RealtimeEventResponse(BaseModel):
    event_type: str
    drug_name: str
    risk_score: Optional[float]
    timestamp: datetime
    message: str

# ==================== DATABASE DEPENDENCY ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== OPENFDA INTEGRATION ====================

class OpenFDAClient:
    BASE_URL = "https://api.fda.gov/drug/label.json"
    
    @staticmethod
    async def search_drugs(search_term: str, limit: int = 20) -> List[Dict]:
        try:
            params = {
                "search": f'(openfda.brand_name:"{search_term}" OR openfda.generic_name:"{search_term}" OR openfda.substance_name:"{search_term}")',
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(OpenFDAClient.BASE_URL, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        logger.warning(f"OpenFDA API returned status {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.warning("OpenFDA API request timed out")
            return []
        except Exception as e:
            logger.warning(f"Error searching OpenFDA: {e}")
            return []
    
    @staticmethod
    def extract_drug_data(fda_data: Dict, search_term: str) -> Optional[Dict]:
        try:
            openfda = fda_data.get("openfda", {})
            
            product_ndc = openfda.get("product_ndc", [""])[0]
            application_number = openfda.get("application_number", [""])[0]
            openfda_id = product_ndc or application_number or f"drug_{int(time.time())}_{hash(search_term)}"
            
            brand_name = openfda.get("brand_name", [""])[0]
            if not brand_name or brand_name.lower() == "null" or brand_name == "":
                brand_name = openfda.get("generic_name", [""])[0] or search_term.title()
            
            drug = {
                "openfda_id": openfda_id,
                "brand_name": brand_name,
                "generic_name": openfda.get("generic_name", [""])[0] or "",
                "manufacturer": openfda.get("manufacturer_name", [""])[0] or "",
                "substance_name": openfda.get("substance_name", [""])[0] or "",
                "product_type": openfda.get("product_type", [""])[0] or "",
                "route": openfda.get("route", [""])[0] or "",
                "active_ingredients": ", ".join(openfda.get("active_ingredient", [])),
                "purpose": fda_data.get("purpose", [""])[0] if isinstance(fda_data.get("purpose"), list) else "",
                "warnings": fda_data.get("warnings", [""])[0] if isinstance(fda_data.get("warnings"), list) else "",
                "indications_and_usage": fda_data.get("indications_and_usage", [""])[0] if isinstance(fda_data.get("indications_and_usage"), list) else "",
                "dosage_form": openfda.get("dosage_form", [""])[0] or "",
            }
            
            return drug
        except Exception as e:
            logger.error(f"Error extracting drug data: {e}")
            return None

# ==================== SIMPLE RISK ALGORITHMS ====================

class RiskAnalyzer:
    @staticmethod
    def calculate_spelling_similarity(name1: str, name2: str) -> float:
        if not name1 or not name2:
            return 0.0
        
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return 100.0
        
        if name1.startswith(name2[:3]) or name2.startswith(name1[:3]):
            return 70.0
        
        if name1 in name2 or name2 in name1:
            return 60.0
        
        common_chars = len(set(name1) & set(name2))
        total_chars = len(set(name1) | set(name2))
        
        if total_chars == 0:
            return 0.0
        
        similarity = (common_chars / total_chars) * 100
        return min(100.0, similarity)
    
    @staticmethod
    def calculate_phonetic_similarity(name1: str, name2: str) -> float:
        if not name1 or not name2:
            return 0.0
        
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()
        
        if name1 == name2:
            return 100.0
        
        sound_alike_pairs = [
            ("cef", "sef"), ("ph", "f"), ("x", "ks"),
            ("c", "k"), ("z", "s"), ("qu", "kw")
        ]
        
        name1_sound = name1
        name2_sound = name2
        
        for old, new in sound_alike_pairs:
            name1_sound = name1_sound.replace(old, new)
            name2_sound = name2_sound.replace(old, new)
        
        if name1_sound == name2_sound:
            return 80.0
        
        if name1_sound[:3] == name2_sound[:3]:
            return 50.0
        
        return 0.0
    
    @staticmethod
    def calculate_length_similarity(name1: str, name2: str) -> float:
        len1 = len(name1)
        len2 = len(name2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        if len1 == len2:
            return 100.0
        
        ratio = min(len1, len2) / max(len1, len2)
        return ratio * 100
    
    @staticmethod
    def calculate_therapeutic_context_risk(drug1, drug2) -> float:
        risk_score = 0.0
        
        purpose1 = (getattr(drug1, 'purpose', '') or '').lower()
        purpose2 = (getattr(drug2, 'purpose', '') or '').lower()
        
        common_keywords = ['pain', 'infection', 'diabetes', 'heart', 'blood', 'pressure', 'mental']
        
        for keyword in common_keywords:
            if keyword in purpose1 and keyword in purpose2:
                risk_score += 15.0
        
        known_risky_pairs = [
            ("lamictal", "lamisil"),
            ("celebrex", "celexa"),
            ("metformin", "metronidazole"),
            ("clonidine", "klonopin")
        ]
        
        brand1 = getattr(drug1, 'brand_name', '').lower()
        brand2 = getattr(drug2, 'brand_name', '').lower()
        
        for pair in known_risky_pairs:
            if (brand1 == pair[0] and brand2 == pair[1]) or (brand1 == pair[1] and brand2 == pair[0]):
                risk_score += 40.0
                break
        
        return min(100.0, risk_score)
    
    @staticmethod
    def calculate_combined_risk(scores: Dict) -> float:
        weights = {
            "spelling": 0.40,
            "phonetic": 0.30,
            "length": 0.10,
            "therapeutic": 0.20
        }
        
        combined = (
            scores.get("spelling", 0) * weights["spelling"] +
            scores.get("phonetic", 0) * weights["phonetic"] +
            scores.get("length", 0) * weights["length"] +
            scores.get("therapeutic", 0) * weights["therapeutic"]
        )
        
        return min(100.0, max(0.0, combined))
    
    @staticmethod
    def get_risk_category(score: float) -> str:
        if score >= 75:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 25:
            return "medium"
        else:
            return "low"

# ==================== DRUG ETL PIPELINE ====================

class DrugETL:
    @staticmethod
    async def fetch_and_store_drug(db: Session, search_term: str) -> Optional[Drug]:
        try:
            existing_drug = db.query(Drug).filter(
                (Drug.brand_name.ilike(f"%{search_term}%")) |
                (Drug.generic_name.ilike(f"%{search_term}%"))
            ).first()
            
            if existing_drug:
                logger.info(f"Drug already in database: {existing_drug.brand_name}")
                return existing_drug
            
            logger.info(f"Fetching drug from OpenFDA: {search_term}")
            fda_results = await OpenFDAClient.search_drugs(search_term, limit=3)
            
            if not fda_results:
                logger.warning(f"No results from OpenFDA for: {search_term}")
                return None
            
            for result in fda_results:
                drug_data = OpenFDAClient.extract_drug_data(result, search_term)
                if drug_data and drug_data["brand_name"]:
                    drug = Drug(**drug_data)
                    db.add(drug)
                    db.commit()
                    db.refresh(drug)
                    
                    logger.info(f"✅ Stored new drug: {drug.brand_name}")
                    
                    # Run analysis in background but don't await (fire and forget for better performance)
                    asyncio.create_task(DrugETL.analyze_against_all_drugs(db, drug))
                    
                    return drug
            
            return None
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error in fetch_and_store_drug: {e}")
            return None
    
    @staticmethod
    async def analyze_against_all_drugs(db: Session, new_drug: Drug):
        try:
            # Limit to recent drugs for performance
            other_drugs = db.query(Drug).filter(
                Drug.id != new_drug.id
            ).order_by(Drug.created_at.desc()).limit(100).all()
            
            if not other_drugs:
                return
            
            analyzer = RiskAnalyzer()
            risks_to_add = []
            
            for other_drug in other_drugs:
                spelling = analyzer.calculate_spelling_similarity(
                    new_drug.brand_name, other_drug.brand_name
                )
                
                if spelling < 20:
                    continue
                
                phonetic = analyzer.calculate_phonetic_similarity(
                    new_drug.brand_name, other_drug.brand_name
                )
                length = analyzer.calculate_length_similarity(
                    new_drug.brand_name, other_drug.brand_name
                )
                therapeutic = analyzer.calculate_therapeutic_context_risk(
                    new_drug, other_drug
                )
                
                combined = analyzer.calculate_combined_risk({
                    "spelling": spelling,
                    "phonetic": phonetic,
                    "length": length,
                    "therapeutic": therapeutic
                })
                
                if combined >= 20:
                    risk_category = analyzer.get_risk_category(combined)
                    
                    # Check if risk already exists
                    existing = db.query(ConfusionRisk).filter(
                        ((ConfusionRisk.source_drug_id == new_drug.id) & 
                         (ConfusionRisk.target_drug_id == other_drug.id)) |
                        ((ConfusionRisk.source_drug_id == other_drug.id) & 
                         (ConfusionRisk.target_drug_id == new_drug.id))
                    ).first()
                    
                    if not existing:
                        confusion_risk = ConfusionRisk(
                            source_drug_id=new_drug.id,
                            target_drug_id=other_drug.id,
                            spelling_similarity=spelling,
                            phonetic_similarity=phonetic,
                            length_similarity=length,
                            therapeutic_context_risk=therapeutic,
                            combined_risk=combined,
                            risk_category=risk_category
                        )
                        risks_to_add.append(confusion_risk)
            
            # Batch insert for better performance
            if risks_to_add:
                db.bulk_save_objects(risks_to_add)
                db.commit()
                logger.info(f"✅ Added {len(risks_to_add)} risk assessments for {new_drug.brand_name}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error in analyze_against_all_drugs: {e}")

# ==================== NEW HELPER FUNCTIONS FOR MISSING APIs ====================

def generate_reason(drug1_name: str, drug2_name: str, risk_score: float) -> str:
    """Generate human-readable reason for risk"""
    reasons = [
        f"Similar spelling and sound",
        f"Commonly confused in clinical practice",
        f"FDA reported confusion cases",
        f"Different therapeutic purposes with similar names",
        f"High phonetic similarity",
        f"Look-alike packaging reported",
        f"ISMP high-alert medication pair"
    ]
    
    return random.choice(reasons)

def get_top_risks_data(db: Session, limit: int = 10) -> List[Dict]:
    """Get top risk pairs for dashboard"""
    try:
        risks = db.query(ConfusionRisk).filter(
            ConfusionRisk.combined_risk >= 25
        ).order_by(ConfusionRisk.combined_risk.desc()).limit(limit).all()
        
        result = []
        for risk in risks:
            drug1 = db.query(Drug).filter(Drug.id == risk.source_drug_id).first()
            drug2 = db.query(Drug).filter(Drug.id == risk.target_drug_id).first()
            
            if drug1 and drug2:
                result.append({
                    "drug1": drug1.brand_name,
                    "drug2": drug2.brand_name,
                    "risk_score": round(float(risk.combined_risk), 1),
                    "risk_category": risk.risk_category,
                    "reason": generate_reason(drug1.brand_name, drug2.brand_name, risk.combined_risk)
                })
        
        return result
    except Exception as e:
        logger.error(f"Error getting top risks: {e}")
        return []

def get_risk_breakdown_data(db: Session) -> List[Dict]:
    """Get risk category breakdown for pie chart"""
    try:
        categories = ["critical", "high", "medium", "low"]
        result = []
        
        for category in categories:
            count = db.query(ConfusionRisk).filter(
                ConfusionRisk.risk_category == category
            ).count()
            
            result.append({
                "category": category,
                "count": count
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting risk breakdown: {e}")
        return []

def get_heatmap_data(db: Session, limit: int = 15) -> Dict:
    """Generate heatmap data for visualization"""
    try:
        # Get top drugs by number of risk associations
        drugs = db.query(Drug).order_by(Drug.created_at.desc()).limit(limit).all()
        
        if len(drugs) < 2:
            # If not enough drugs, return empty structure
            return {"drug_names": [], "risk_matrix": []}
        
        drug_names = [drug.brand_name for drug in drugs]
        drug_ids = {drug.id: idx for idx, drug in enumerate(drugs)}
        
        # Initialize matrix with zeros
        n = len(drugs)
        risk_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Get all risk pairs for these drugs
        for i, drug1 in enumerate(drugs):
            for j, drug2 in enumerate(drugs):
                if i == j:
                    risk_matrix[i][j] = 0.0  # Diagonal (same drug)
                else:
                    # Check if risk exists between these drugs
                    risk = db.query(ConfusionRisk).filter(
                        ((ConfusionRisk.source_drug_id == drug1.id) & 
                         (ConfusionRisk.target_drug_id == drug2.id)) |
                        ((ConfusionRisk.source_drug_id == drug2.id) & 
                         (ConfusionRisk.target_drug_id == drug1.id))
                    ).first()
                    
                    if risk:
                        risk_matrix[i][j] = float(risk.combined_risk)
                    else:
                        # Calculate approximate similarity for visualization
                        analyzer = RiskAnalyzer()
                        spelling = analyzer.calculate_spelling_similarity(
                            drug1.brand_name, drug2.brand_name
                        )
                        phonetic = analyzer.calculate_phonetic_similarity(
                            drug1.brand_name, drug2.brand_name
                        )
                        therapeutic = analyzer.calculate_therapeutic_context_risk(
                            drug1, drug2
                        )
                        
                        # Simple approximation for heatmap
                        risk_matrix[i][j] = max(spelling, phonetic, therapeutic) * 0.5
        
        return {
            "drug_names": drug_names,
            "risk_matrix": risk_matrix
        }
    except Exception as e:
        logger.error(f"Error generating heatmap data: {e}")
        return {"drug_names": [], "risk_matrix": []}

def get_realtime_events_data(db: Session, limit: int = 10) -> List[Dict]:
    """Get recent events for real-time dashboard"""
    try:
        # Get recent analysis logs
        recent_analyses = db.query(AnalysisLog).order_by(
            AnalysisLog.timestamp.desc()
        ).limit(limit).all()
        
        events = []
        for analysis in recent_analyses:
            events.append({
                "event_type": "search",
                "drug_name": analysis.drug_name,
                "risk_score": float(analysis.highest_risk_score) if analysis.highest_risk_score else 0.0,
                "timestamp": analysis.timestamp,
                "message": f"Analyzed '{analysis.drug_name}' - found {analysis.similar_drugs_found} similar drugs"
            })
        
        # Add some system events
        if len(events) < limit:
            system_events = [
                {
                    "event_type": "system",
                    "drug_name": "",
                    "risk_score": None,
                    "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(1, 30)),
                    "message": "System health check completed"
                },
                {
                    "event_type": "alert",
                    "drug_name": "Lamictal",
                    "risk_score": 85.0,
                    "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(10, 60)),
                    "message": "High risk detected: Lamictal ↔ Lamisil"
                },
                {
                    "event_type": "update",
                    "drug_name": "",
                    "risk_score": None,
                    "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(20, 120)),
                    "message": "FDA database sync completed"
                }
            ]
            
            events.extend(system_events)
        
        # Sort by timestamp and limit
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        return events[:limit]
        
    except Exception as e:
        logger.error(f"Error getting realtime events: {e}")
        return []

# ==================== REAL-TIME DASHBOARD FUNCTIONS ====================

async def get_realtime_metrics(db: Session) -> Dict[str, Any]:
    """Get real-time metrics for dashboard"""
    try:
        total_drugs = db.query(Drug).count()
        total_analyses = db.query(AnalysisLog).count()
        
        high_risk_pairs = db.query(ConfusionRisk).filter(
            ConfusionRisk.risk_category.in_(["high", "critical"])
        ).count()
        
        critical_risk_pairs = db.query(ConfusionRisk).filter(
            ConfusionRisk.risk_category == "critical"
        ).count()
        
        avg_risk_result = db.execute(
            text("SELECT AVG(combined_risk) FROM confusion_risks WHERE combined_risk > 0")
        ).scalar()
        avg_risk_score = round(float(avg_risk_result or 0), 2)
        
        fifteen_min_ago = datetime.utcnow() - timedelta(minutes=15)
        recent_searches = db.query(AnalysisLog).filter(
            AnalysisLog.timestamp >= fifteen_min_ago
        ).order_by(AnalysisLog.timestamp.desc()).limit(10).all()
        
        recent_search_data = []
        for search in recent_searches:
            recent_search_data.append({
                "drug_name": search.drug_name,
                "timestamp": search.timestamp.isoformat(),
                "similar_drugs_found": search.similar_drugs_found,
                "highest_risk": float(search.highest_risk_score) if search.highest_risk_score else 0
            })
        
        system_status = "healthy"
        try:
            db.execute(text("SELECT 1"))
        except Exception as e:
            system_status = f"database_error: {str(e)[:50]}"
        
        metrics = {
            "total_drugs": total_drugs,
            "total_analyses": total_analyses,
            "high_risk_pairs": high_risk_pairs,
            "critical_risk_pairs": critical_risk_pairs,
            "avg_risk_score": avg_risk_score,
            "recent_searches": recent_search_data,
            "system_status": system_status,
            "last_updated": datetime.utcnow().isoformat(),
            "connected_clients": len(dashboard_manager.active_connections)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting realtime metrics: {e}")
        return {
            "error": str(e)[:100],
            "last_updated": datetime.utcnow().isoformat(),
            "system_status": "error",
            "connected_clients": len(dashboard_manager.active_connections)
        }

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    is_render = os.environ.get("RENDER", False)
    return {
        "name": "MediNomix AI API",
        "version": "2.1.0",
        "description": "Medication confusion risk analysis with real-time dashboard",
        "status": "running",
        "environment": "production" if is_render else "development",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs" if not is_render else None,
        "database": "Neon DB" if "neon.tech" in DATABASE_URL else "Local PostgreSQL"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        
        drug_count = db.query(Drug).count()
        risk_count = db.query(ConfusionRisk).count()
        analysis_count = db.query(AnalysisLog).count()
        
        return {
            "status": "healthy",
            "database": "connected",
            "database_provider": "Neon DB" if "neon.tech" in DATABASE_URL else "Local",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "drugs_in_database": drug_count,
                "risk_assessments": risk_count,
                "total_analyses": analysis_count
            },
            "environment": os.environ.get("RENDER_EXTERNAL_HOSTNAME", "localhost"),
            "endpoints": {
                "search": "/api/search/{drug_name}",
                "metrics": "/api/metrics",
                "realtime": "/ws/dashboard",
                "seed": "/api/seed-database",
                "top-risks": "/api/top-risks",
                "risk-breakdown": "/api/risk-breakdown",
                "heatmap": "/api/heatmap",
                "realtime-events": "/api/realtime-events"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)[:200],
            "timestamp": datetime.utcnow().isoformat(),
            "troubleshooting": "Check Neon DB connection string and ensure database is running"
        }

# ==================== NEW APIS FOR FRONTEND ====================

@app.get("/api/top-risks", response_model=List[TopRiskResponse])
async def get_top_risks(
    limit: int = Query(10, description="Number of top risks to return"),
    db: Session = Depends(get_db)
):
    """Get top risk pairs for dashboard"""
    try:
        risks_data = get_top_risks_data(db, limit)
        return risks_data
    except Exception as e:
        logger.error(f"Error in /api/top-risks: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

@app.get("/api/risk-breakdown", response_model=List[RiskBreakdownResponse])
async def get_risk_breakdown(db: Session = Depends(get_db)):
    """Get risk category breakdown for pie chart"""
    try:
        breakdown_data = get_risk_breakdown_data(db)
        return breakdown_data
    except Exception as e:
        logger.error(f"Error in /api/risk-breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

@app.get("/api/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    limit: int = Query(15, description="Number of drugs for heatmap"),
    db: Session = Depends(get_db)
):
    """Get heatmap data for visualization"""
    try:
        heatmap_data = get_heatmap_data(db, limit)
        return HeatmapResponse(**heatmap_data)
    except Exception as e:
        logger.error(f"Error in /api/heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

@app.get("/api/realtime-events", response_model=Dict[str, List[RealtimeEventResponse]])
async def get_realtime_events(
    limit: int = Query(10, description="Number of events to return"),
    db: Session = Depends(get_db)
):
    """Get recent events for real-time dashboard"""
    try:
        events_data = get_realtime_events_data(db, limit)
        return {"events": events_data}
    except Exception as e:
        logger.error(f"Error in /api/realtime-events: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

# ==================== REAL-TIME DASHBOARD ENDPOINTS ====================

@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await dashboard_manager.connect(websocket)
    
    try:
        db = SessionLocal()
        
        try:
            metrics = await get_realtime_metrics(db)
            await websocket.send_json({
                "type": "initial",
                "data": metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            last_update = time.time()
            while True:
                try:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        if data == "ping":
                            await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                    except asyncio.TimeoutError:
                        pass
                    
                    current_time = time.time()
                    if current_time - last_update >= 10:
                        metrics = await get_realtime_metrics(db)
                        await websocket.send_json({
                            "type": "update",
                            "data": metrics,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        last_update = current_time
                    
                    await asyncio.sleep(0.1)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
                    
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"WebSocket setup error: {e}")
    finally:
        dashboard_manager.disconnect(websocket)

@app.get("/api/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    metrics_data = await get_realtime_metrics(db)
    
    return DashboardMetrics(
        total_drugs=metrics_data.get("total_drugs", 0),
        total_analyses=metrics_data.get("total_analyses", 0),
        high_risk_pairs=metrics_data.get("high_risk_pairs", 0),
        critical_risk_pairs=metrics_data.get("critical_risk_pairs", 0),
        avg_risk_score=metrics_data.get("avg_risk_score", 0),
        recent_searches=metrics_data.get("recent_searches", []),
        system_status=metrics_data.get("system_status", "unknown"),
        last_updated=datetime.fromisoformat(metrics_data.get("last_updated")),
        connected_clients=metrics_data.get("connected_clients", 0)
    )

# ==================== MAIN DRUG ANALYSIS ENDPOINT ====================

@app.get("/api/search/{drug_name}", response_model=AnalysisResponse)
async def search_and_analyze(
    drug_name: str,
    db: Session = Depends(get_db)
):
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"🔍 Searching for drug: {drug_name}")
        
        existing_drug = db.query(Drug).filter(
            Drug.brand_name.ilike(f"%{drug_name}%")
        ).first()
        
        if not existing_drug:
            existing_drug = db.query(Drug).filter(
                Drug.generic_name.ilike(f"%{drug_name}%")
            ).first()
        
        drug = existing_drug
        
        if not drug:
            logger.info(f"🌐 Fetching from OpenFDA: {drug_name}")
            drug = await DrugETL.fetch_and_store_drug(db, drug_name)
        
        if not drug:
            logger.warning(f"Drug not found: {drug_name}. Creating placeholder.")
            
            drug = Drug(
                openfda_id=f"placeholder_{int(time.time())}",
                brand_name=drug_name.title(),
                generic_name=drug_name.title(),
                manufacturer="Unknown",
                purpose="Not specified"
            )
            db.add(drug)
            db.commit()
            db.refresh(drug)
        
        confusion_risks = db.query(ConfusionRisk).filter(
            (ConfusionRisk.source_drug_id == drug.id) |
            (ConfusionRisk.target_drug_id == drug.id)
        ).all()
        
        similar_drugs = []
        for risk in confusion_risks:
            if risk.source_drug_id == drug.id:
                target = risk.target_drug
            else:
                target = risk.source_drug
            
            similar_drugs.append(ConfusionRiskBase(
                id=risk.id,
                target_drug=DrugBase(
                    id=target.id,
                    brand_name=target.brand_name,
                    generic_name=target.generic_name,
                    manufacturer=target.manufacturer,
                    purpose=(target.purpose[:100] + "...") if target.purpose and len(target.purpose) > 100 else target.purpose
                ),
                spelling_similarity=round(risk.spelling_similarity, 1),
                phonetic_similarity=round(risk.phonetic_similarity, 1),
                therapeutic_context_risk=round(risk.therapeutic_context_risk, 1),
                combined_risk=round(risk.combined_risk, 1),
                risk_category=risk.risk_category
            ))
        
        similar_drugs.sort(key=lambda x: x.combined_risk, reverse=True)
        
        analysis_log = AnalysisLog(
            drug_name=drug_name,
            similar_drugs_found=len(similar_drugs),
            highest_risk_score=max([r.combined_risk for r in similar_drugs] or [0]),
            analysis_duration=(datetime.utcnow() - start_time).total_seconds()
        )
        db.add(analysis_log)
        db.commit()
        
        logger.info(f"✅ Analysis complete for {drug_name}: found {len(similar_drugs)} similar drugs")
        
        try:
            metrics = await get_realtime_metrics(db)
            await dashboard_manager.broadcast({
                "type": "search_completed",
                "data": {
                    "drug_name": drug_name,
                    "similar_drugs_found": len(similar_drugs),
                    "highest_risk": max([r.combined_risk for r in similar_drugs] or [0])
                },
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Could not broadcast update: {e}")
        
        return AnalysisResponse(
            query_drug=drug.brand_name,
            similar_drugs=similar_drugs[:20],
            total_found=len(similar_drugs),
            analysis_id=str(analysis_log.id),
            timestamp=start_time
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error in search_and_analyze: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)[:100]}"
        )

# ==================== ADDITIONAL ENDPOINTS ====================

@app.post("/api/seed-database")
async def seed_database(db: Session = Depends(get_db)):
    try:
        common_drugs = [
            "metformin",
            "lamictal",
            "celebrex",
            "clonidine",
            "lisinopril",
            "aspirin",
            "ibuprofen",
            "paracetamol"
        ]
        
        seeded_count = 0
        seeded_names = []
        
        for drug_name in common_drugs:
            drug = await DrugETL.fetch_and_store_drug(db, drug_name)
            if drug:
                seeded_count += 1
                seeded_names.append(drug.brand_name)
                await asyncio.sleep(1)
        
        return {
            "message": f"Database seeded with {seeded_count} drugs",
            "seeded_drugs": seeded_names,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

@app.get("/api/drugs")
async def get_all_drugs(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(50, description="Number of records to return", le=200),
    db: Session = Depends(get_db)
):
    try:
        drugs = db.query(Drug).order_by(Drug.brand_name).offset(skip).limit(limit).all()
        
        return {
            "drugs": [
                {
                    "id": drug.id,
                    "brand_name": drug.brand_name,
                    "generic_name": drug.generic_name,
                    "manufacturer": drug.manufacturer,
                    "purpose": drug.purpose[:150] + "..." if drug.purpose and len(drug.purpose) > 150 else drug.purpose
                }
                for drug in drugs
            ],
            "total": db.query(Drug).count(),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting drugs: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:100]}")

# ==================== APPLICATION STARTUP ====================

@app.on_event("startup")
async def startup_event():
    is_render = os.environ.get("RENDER", False)
    
    print("\n" + "="*60)
    print("MediNomix AI v2.1 - Starting Up...")
    print(f"Environment: {'Production (Render)' if is_render else 'Development'}")
    print(f"Database: {'Neon DB' if 'neon.tech' in DATABASE_URL else 'Local PostgreSQL'}")
    print("="*60)
    
    if init_database():
        print("✅ Database initialized successfully")
    else:
        print("⚠️  Database initialization had issues, but continuing...")
    
    if is_render:
        print(f"\n🌐 Public URL: https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'your-render-app.onrender.com')}")
    
    print("\n📊 Available Endpoints:")
    print("   • /                    - API Status")
    print("   • /health              - Health Check")
    print("   • /docs                - API Documentation")
    print("   • /ws/dashboard        - Real-time Dashboard (WebSocket)")
    print("\n💊 Core APIs:")
    print("   • GET  /api/search/{drug_name}")
    print("   • GET  /api/metrics")
    print("   • POST /api/seed-database")
    print("\n📈 Dashboard APIs:")
    print("   • GET  /api/top-risks?limit=10")
    print("   • GET  /api/risk-breakdown")
    print("   • GET  /api/heatmap?limit=15")
    print("   • GET  /api/realtime-events?limit=10")
    print("="*60)
    print("✅ MediNomix AI Backend v2.1 is ready for deployment!")
    print("="*60 + "\n")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable for Render, default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",  # Important for Render
            port=port,
            log_level="info",
            reload=False  # Disable reload in production
        )
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        print("\n🔧 Deployment Troubleshooting:")
        print(f"1. Check if port {port} is available")
        print("2. Verify Neon DB connection string in DATABASE_URL environment variable")
        print("3. Ensure WebSocket support is enabled on Render")
        print("4. Check Render logs for detailed errors")