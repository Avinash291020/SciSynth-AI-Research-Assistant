# -*- coding: utf-8 -*-
"""
Production-Level FastAPI Application for SciSynth AI Research Assistant.

This module demonstrates:
- Production-ready API with authentication and authorization
- Rate limiting and request validation
- Comprehensive monitoring and logging
- Multi-user support and session management
- Health checks and metrics
- Docker and Kubernetes deployment ready
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import (
    FastAPI, HTTPException, Depends, status, Request, BackgroundTasks,
    WebSocket, WebSocketDisconnect
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import redis
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import jwt
from passlib.context import CryptContext
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.orchestrator import ResearchOrchestrator

# Configure structured logging
logger = structlog.get_logger()

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Redis configuration - Use sync Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client: Optional[redis.Redis] = None
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Test connection
    redis_client.ping()  # type: ignore
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# For async Redis operations, we'll use a sync client but handle it properly
# In production, consider using aioredis for true async support

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_USERS = Counter('active_users_total', 'Total active users')

# Global state
orchestrator = None
user_sessions = {}
executor = ThreadPoolExecutor(max_workers=10)


# --- Pydantic Models ---

class UserCreate(BaseModel):
    """User registration model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str

class ResearchQuery(BaseModel):
    """Enhanced research query model."""
    question: str = Field(..., min_length=10, max_length=1000)
    use_all_systems: bool = True
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    max_tokens: int = Field(default=1000, ge=100, le=5000)
    include_citations: bool = True
    analysis_depth: str = Field(default="standard", pattern="^(basic|standard|comprehensive)$")

class AnalysisResult(BaseModel):
    """Analysis result model."""
    query_id: str
    question: str
    result: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    status: str

class HealthCheck(BaseModel):
    """Enhanced health check model."""
    status: str
    orchestrator_status: str
    redis_status: str
    memory_usage: Dict[str, float]
    active_connections: int
    uptime: float

class UserSession(BaseModel):
    """User session model."""
    user_id: str
    username: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    request_count: int

# --- Authentication and Authorization ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = verify_token(token)
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Check if user exists in Redis - handle sync Redis operations
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        user_data: Dict[str, str] = redis_client.hgetall(f"user:{username}")  # type: ignore
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
        
        return {"username": username, "user_id": user_data.get("user_id", "")}
    except Exception as e:
        logger.error("Redis operation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Database error")

# --- Rate Limiting ---

class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_client, max_requests: int = 100, window_seconds: int = 3600):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        key = f"rate_limit:{user_id}"
        current = self.redis.get(key)
        
        if current is None:
            self.redis.setex(key, self.window_seconds, 1)
            return True
        
        current_count = int(current)
        if current_count >= self.max_requests:
            return False
        
        self.redis.incr(key)
        return True

rate_limiter = RateLimiter(redis_client)

# --- Application Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global orchestrator
    logger.info("Starting SciSynth AI Research Assistant API")
    
    # Initialize orchestrator
    try:
        results_path = Path("results/all_papers_results.json")
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            orchestrator = ResearchOrchestrator(all_results)
            logger.info("ResearchOrchestrator initialized successfully")
        else:
            logger.warning("Results file not found, orchestrator not initialized")
    except Exception as e:
        logger.error("Failed to initialize orchestrator", error=str(e))
    
    # Test Redis connection
    try:
        redis_client.ping()  # type: ignore
        logger.info("Redis connection established")
    except Exception as e:
        logger.error("Redis connection failed", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down SciSynth AI Research Assistant API")
    executor.shutdown(wait=True)

# --- FastAPI Application ---

app = FastAPI(
    title="SciSynth AI Research Assistant API",
    description="Production-ready API for multi-paradigm AI research assistant",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# --- Middleware for Monitoring ---

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor and log all requests."""
    start_time = time.time()
    
    # Extract user info if authenticated
    user_id = "anonymous"
    try:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            user_id = payload.get("sub", "unknown")
    except:
        pass
    
    # Process request
    response = await call_next(request)
    
    # Calculate metrics
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.observe(process_time)
    
    # Log request
    logger.info(
        "HTTP request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time=process_time,
        user_id=user_id,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    return response

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the SciSynth AI Research Assistant API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthCheck, tags=["Status"])
async def health_check():
    """Comprehensive health check endpoint."""
    import psutil
    
    # Check orchestrator status
    orchestrator_status = "ready" if orchestrator else "not_available"
    
    # Check Redis status
    try:
        redis_client.ping()  # type: ignore
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "percent_used": memory.percent
    }
    
    # Get active connections
    active_connections = len(user_sessions)
    
    # Calculate uptime
    uptime = time.time() - os.path.getctime(__file__)
    
    return HealthCheck(
        status="ok",
        orchestrator_status=orchestrator_status,
        redis_status=redis_status,
        memory_usage=memory_usage,
        active_connections=active_connections,
        uptime=uptime
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/register", tags=["Authentication"])
async def register_user(user: UserCreate):
    """Register a new user."""
    # Check if user already exists
    if redis_client.exists(f"user:{user.username}"):  # type: ignore
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user.password)
    
    user_data = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat(),
        "request_count": 0
    }
    
    # Store in Redis
    redis_client.hmset(f"user:{user.username}", user_data)  # type: ignore
    
    logger.info("User registered", username=user.username, user_id=user_id)
    
    return {"message": "User registered successfully", "user_id": user_id}

@app.post("/login", tags=["Authentication"])
async def login_user(user: UserLogin):
    """Authenticate user and return access token."""
    # Get user data
    user_data: Dict[str, str] = redis_client.hgetall(f"user:{user.username}")  # type: ignore
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    hashed_password = user_data["hashed_password"]
    if not verify_password(user.password, hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Update last login
    redis_client.hset(f"user:{user.username}", "last_login", datetime.utcnow().isoformat())  # type: ignore
    
    logger.info("User logged in", username=user.username)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/analyze", tags=["AI Core"])
async def analyze_research_question(
    query: ResearchQuery,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform comprehensive research analysis with authentication and rate limiting."""
    # Rate limiting
    if not rate_limiter.is_allowed(current_user["user_id"]):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator is not available or still initializing."
        )
    
    # Generate query ID
    query_id = str(uuid.uuid4())
    
    # Create session
    session = UserSession(
        user_id=current_user["user_id"],
        username=current_user["username"],
        session_id=query_id,
        created_at=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        request_count=1
    )
    user_sessions[query_id] = session
    
    try:
        start_time = time.time()
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            orchestrator.ask_question,
            query.question,
            query.use_all_systems
        )
        
        processing_time = time.time() - start_time
        
        # Create analysis result
        analysis_result = AnalysisResult(
            query_id=query_id,
            question=query.question,
            result=result,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            status="completed"
        )
        
        # Store result in Redis
        redis_client.setex(f"analysis:{query_id}", 3600, json.dumps(analysis_result.dict(), default=str))  # type: ignore
        
        # Update user stats
        redis_client.hincrby(f"user:{current_user['username']}", "request_count", 1)  # type: ignore
        
        logger.info(
            "Analysis completed",
            query_id=query_id,
            user_id=current_user["user_id"],
            processing_time=processing_time
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(
            "Analysis failed",
            query_id=query_id,
            user_id=current_user["user_id"],
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during analysis: {str(e)}"
        )
    finally:
        # Clean up session
        if query_id in user_sessions:
            del user_sessions[query_id]

@app.get("/analysis/{query_id}", tags=["AI Core"])
async def get_analysis_result(
    query_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analysis result by query ID."""
    # Get result from Redis
    result_data: Optional[str] = redis_client.get(f"analysis:{query_id}")  # type: ignore
    if not result_data:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    result = json.loads(result_data)
    return result

@app.get("/user/stats", tags=["User"])
async def get_user_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get user statistics."""
    user_data: Dict[str, str] = redis_client.hgetall(f"user:{current_user['username']}")  # type: ignore
    
    stats = {
        "username": current_user["username"],
        "user_id": current_user["user_id"],
        "total_requests": int(user_data.get("request_count", 0)),
        "created_at": user_data.get("created_at", ""),
        "last_login": user_data.get("last_login", "")
    }
    
    return stats

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_text(json.dumps({
                "type": "status",
                "timestamp": datetime.utcnow().isoformat(),
                "active_users": len(user_sessions)
            }))
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", user_id=user_id)

# --- Main entry point ---

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    ) 