# -*- coding: utf-8 -*-
"""
System Design and Scaling for Production-Level AI Engineering.

This module demonstrates:
- Memory management and optimization
- Latency optimization techniques
- Multi-user support and session management
- Distributed computing with Ray
- Load balancing and caching strategies
- Performance monitoring and profiling
- Scalable architecture patterns
"""

import os
import sys
import time
import json
import asyncio
import threading
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import gc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import structlog
from functools import lru_cache, wraps
import hashlib
import pickle
from contextlib import contextmanager
import tracemalloc
import cProfile
import pstats
import io

# Configure structured logging first
logger = structlog.get_logger()

# Optional imports with fallbacks
try:
    import ray  # type: ignore
    from ray import serve  # type: ignore
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available - distributed processing will be disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring will be limited")

@dataclass
class SystemConfig:
    """Configuration for system design and scaling."""
    max_memory_gb: float = 8.0
    max_concurrent_users: int = 100
    cache_size_mb: int = 512
    max_request_timeout: int = 30
    enable_caching: bool = True
    enable_load_balancing: bool = True
    enable_monitoring: bool = True
    worker_processes: int = multiprocessing.cpu_count()
    redis_url: str = "redis://localhost:6379"
    ray_address: str = "auto"


class MemoryManager:
    """Advanced memory management and optimization."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.memory_usage = deque(maxlen=1000)
        self.large_objects = set()
        self.memory_threshold = 0.8  # 80% of max memory
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            stats = {
                "rss_gb": memory_info.rss / 1024**3,
                "vms_gb": memory_info.vms / 1024**3,
                "percent": process.memory_percent(),
                "available_gb": psutil.virtual_memory().available / 1024**3,
                "total_gb": psutil.virtual_memory().total / 1024**3
            }
        else:
            # Fallback memory monitoring without psutil
            import resource
            try:
                memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                stats = {
                    "rss_gb": memory_usage / 1024**3 if hasattr(resource, 'RUSAGE_SELF') else 0.0,
                    "vms_gb": 0.0,
                    "percent": 0.0,
                    "available_gb": 0.0,
                    "total_gb": 0.0
                }
            except (ImportError, AttributeError):
                # Windows fallback
                stats = {
                    "rss_gb": 0.0,
                    "vms_gb": 0.0,
                    "percent": 0.0,
                    "available_gb": 0.0,
                    "total_gb": 0.0
                }
        
        self.memory_usage.append(stats)
        return stats
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_usage()
        return stats["rss_gb"] > self.max_memory_bytes * self.memory_threshold
    
    def optimize_memory(self):
        """Perform memory optimization."""
        logger.info("Starting memory optimization")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info("Garbage collection completed", collected_objects=collected)
        
        # Clear large objects cache if under pressure
        if self.is_memory_pressure():
            self.clear_large_objects()
        
        # Monitor memory usage
        stats = self.get_memory_usage()
        logger.info("Memory optimization completed", memory_stats=stats)
    
    def clear_large_objects(self):
        """Clear large objects from memory."""
        logger.info("Clearing large objects from memory")
        self.large_objects.clear()
    
    def track_large_object(self, obj: Any, size_mb: float):
        """Track large objects for potential cleanup."""
        if size_mb > 10:  # Track objects larger than 10MB
            self.large_objects.add(obj)
            logger.info("Tracking large object", size_mb=size_mb)
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory usage during operations."""
        start_stats = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_stats = self.get_memory_usage()
            end_time = time.time()
            
            memory_delta = end_stats["rss_gb"] - start_stats["rss_gb"]
            duration = end_time - start_time
            
            logger.info(
                "Memory operation completed",
                operation=operation_name,
                memory_delta_gb=memory_delta,
                duration_seconds=duration,
                final_memory_gb=end_stats["rss_gb"]
            )


class CacheManager:
    """Advanced caching system with multiple strategies."""
    
    def __init__(self, max_size_mb: int = 512, redis_url: str = "redis://localhost:6379"):
        self.max_size_bytes = max_size_mb * 1024**2
        self.current_size = 0
        self.cache = {}
        self.access_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize Redis client with error handling
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(redis_url)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning("Redis cache initialization failed, using local cache only", error=str(e))
            self.redis_client = None
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return sys.getsizeof(obj)
    
    def _evict_lru(self, required_size: int):
        """Evict least recently used items."""
        while self.current_size + required_size > self.max_size_bytes and self.cache:
            # Find LRU item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            lru_size = self._get_object_size(self.cache[lru_key])
            
            del self.cache[lru_key]
            del self.access_times[lru_key]
            self.current_size -= lru_size
            
            logger.debug("Evicted LRU item", key=lru_key, size=lru_size)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        # Try local cache first
        if key in self.cache:
            self.access_times[key] = time.time()
            self.cache_hits += 1
            return self.cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                redis_value: Optional[Union[str, bytes]] = self.redis_client.get(key)  # type: ignore
                if redis_value:
                    # Handle both string and bytes responses from Redis
                    if isinstance(redis_value, bytes):
                        value = pickle.loads(redis_value)
                    else:
                        value = pickle.loads(redis_value.encode('utf-8'))
                    self.cache_hits += 1
                    return value
            except Exception as e:
                logger.warning("Redis cache access failed", error=str(e))
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in cache."""
        size = self._get_object_size(value)
        
        # Evict if necessary
        self._evict_lru(size)
        
        # Store in local cache
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.current_size += size
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception as e:
                logger.warning("Redis cache storage failed", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "current_size_mb": self.current_size / 1024**2,
            "max_size_mb": self.max_size_bytes / 1024**2,
            "items_count": len(self.cache)
        }


class LoadBalancer:
    """Simple load balancer for distributing requests."""
    
    def __init__(self, workers: List[str]):
        self.workers = workers
        self.current_index = 0
        self.worker_loads = defaultdict(int)
        self.worker_health = {worker: True for worker in workers}
        
    def get_next_worker(self) -> str:
        """Get next available worker using round-robin."""
        available_workers = [w for w in self.workers if self.worker_health[w]]
        
        if not available_workers:
            raise RuntimeError("No healthy workers available")
        
        worker = available_workers[self.current_index % len(available_workers)]
        self.current_index += 1
        self.worker_loads[worker] += 1
        
        return worker
    
    def get_least_loaded_worker(self) -> str:
        """Get worker with least load."""
        available_workers = [w for w in self.workers if self.worker_health[w]]
        
        if not available_workers:
            raise RuntimeError("No healthy workers available")
        
        worker = min(available_workers, key=lambda w: self.worker_loads[w])
        self.worker_loads[worker] += 1
        
        return worker
    
    def mark_worker_unhealthy(self, worker: str):
        """Mark worker as unhealthy."""
        self.worker_health[worker] = False
        logger.warning("Worker marked as unhealthy", worker=worker)
    
    def mark_worker_healthy(self, worker: str):
        """Mark worker as healthy."""
        self.worker_health[worker] = True
        logger.info("Worker marked as healthy", worker=worker)


class PerformanceProfiler:
    """Performance profiling and monitoring."""
    
    def __init__(self):
        self.profiles = {}
        self.tracemalloc_enabled = False
        self.profile_counter = 0
        
    def start_profiling(self, name: str):
        """Start profiling for a specific operation."""
        # Make profile name unique to avoid conflicts
        unique_name = f"{name}_{self.profile_counter}"
        self.profile_counter += 1
        
        if not self.tracemalloc_enabled:
            try:
                tracemalloc.start()
                self.tracemalloc_enabled = True
            except RuntimeError:
                # tracemalloc already started
                pass
        
        self.profiles[unique_name] = {
            'start_time': time.time(),
            'start_snapshot': tracemalloc.take_snapshot() if self.tracemalloc_enabled else None,
            'profiler': cProfile.Profile(),
            'original_name': name
        }
        self.profiles[unique_name]['profiler'].enable()
        return unique_name
    
    def stop_profiling(self, name: str) -> Dict[str, Any]:
        """Stop profiling and return results."""
        if name not in self.profiles:
            raise ValueError(f"No profiling session found for {name}")
        
        profile_data = self.profiles[name]
        profile_data['profiler'].disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profile_data['profiler'], stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Get memory snapshot if available
        memory_stats = []
        peak_memory_mb = 0.0
        if self.tracemalloc_enabled and profile_data['start_snapshot']:
            try:
                end_snapshot = tracemalloc.take_snapshot()
                memory_stats = end_snapshot.compare_to(profile_data['start_snapshot'], 'lineno')
                memory_stats = memory_stats[:10]  # Top 10 memory differences
                peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1024**2
            except Exception as e:
                logger.warning("Memory profiling failed", error=str(e))
        
        # Calculate timing
        duration = time.time() - profile_data['start_time']
        
        results = {
            'duration': duration,
            'profile_stats': s.getvalue(),
            'memory_stats': memory_stats,
            'peak_memory_mb': peak_memory_mb
        }
        
        del self.profiles[name]
        return results


class DistributedProcessor:
    """Distributed processing using Ray."""
    
    def __init__(self, ray_address: str = "auto"):
        self.ray_available = RAY_AVAILABLE
        if self.ray_available:
            try:
                if not ray.is_initialized():
                    ray.init(address=ray_address)
                self.cluster_resources = ray.cluster_resources()
                logger.info("Ray initialized", resources=self.cluster_resources)
            except Exception as e:
                logger.warning("Ray initialization failed", error=str(e))
                self.ray_available = False
        else:
            logger.info("Ray not available - using local processing")
            self.cluster_resources = {}
    
    def process_chunk(self, data_chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a chunk of data locally or remotely."""
        if self.ray_available and RAY_AVAILABLE:
            # Use Ray remote processing
            try:
                remote_func = ray.remote(self._process_chunk_remote)
                return ray.get(remote_func.remote(data_chunk, processor_func))
            except Exception as e:
                logger.warning("Ray processing failed, falling back to local", error=str(e))
                return [processor_func(item) for item in data_chunk]
        else:
            # Local processing
            return [processor_func(item) for item in data_chunk]
    
    def _process_chunk_remote(self, data_chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a chunk of data remotely."""
        return [processor_func(item) for item in data_chunk]
    
    def process_distributed(self, data: List[Any], processor_func: Callable, chunk_size: int = 100) -> List[Any]:
        """Process data in parallel using Ray or local processing."""
        # Split data into chunks
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        if self.ray_available:
            try:
                # Apply ray.remote decorator conditionally
                if RAY_AVAILABLE:
                    remote_func = ray.remote(self._process_chunk_remote)
                    futures = [remote_func.remote(chunk, processor_func) for chunk in chunks]
                else:
                    return self._process_locally(chunks, processor_func)
                
                # Collect results
                results = ray.get(futures)
                
                # Flatten results
                return [item for sublist in results for item in sublist]
            except Exception as e:
                logger.warning("Ray distributed processing failed, using local", error=str(e))
                # Fallback to local processing
                return self._process_locally(chunks, processor_func)
        else:
            # Local processing
            return self._process_locally(chunks, processor_func)
    
    def _process_locally(self, chunks: List[List[Any]], processor_func: Callable) -> List[Any]:
        """Process chunks locally using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
            futures = [executor.submit(self.process_chunk, chunk, processor_func) for chunk in chunks]
            for future in futures:
                results.extend(future.result())
        return results
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get Ray cluster statistics or local stats."""
        if self.ray_available:
            try:
                return {
                    "nodes": len(ray.nodes()),
                    "resources": ray.cluster_resources(),
                    "available_resources": ray.available_resources()
                }
            except Exception as e:
                logger.warning("Failed to get Ray cluster stats", error=str(e))
                return {"error": "Ray cluster stats unavailable"}
        else:
            return {
                "nodes": 1,
                "resources": {"CPU": multiprocessing.cpu_count()},
                "available_resources": {"CPU": multiprocessing.cpu_count()},
                "mode": "local_processing"
            }


class SessionManager:
    """Multi-user session management."""
    
    def __init__(self, max_sessions: int = 1000):
        self.sessions = {}
        self.max_sessions = max_sessions
        self.session_locks = {}
        
    def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create a new user session."""
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_expired_sessions()
        
        session_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()
        
        session = {
            'user_id': user_id,
            'data': session_data,
            'created_at': time.time(),
            'last_activity': time.time(),
            'request_count': 0
        }
        
        self.sessions[session_id] = session
        self.session_locks[session_id] = threading.Lock()
        
        logger.info("Session created", session_id=session_id, user_id=user_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session['last_activity'] = time.time()
            session['request_count'] += 1
            return session
        return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        """Update session data."""
        if session_id in self.sessions:
            with self.session_locks[session_id]:
                self.sessions[session_id]['data'].update(data)
                self.sessions[session_id]['last_activity'] = time.time()
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.session_locks[session_id]
            logger.info("Session deleted", session_id=session_id)
    
    def _cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > max_age_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info("Cleaned up expired sessions", count=len(expired_sessions))
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "total_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "active_users": len(set(s['user_id'] for s in self.sessions.values()))
        }


class ScalableSystem:
    """Main scalable system orchestrator."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_gb)
        self.cache_manager = CacheManager(config.cache_size_mb, config.redis_url)
        self.load_balancer = LoadBalancer([f"worker_{i}" for i in range(config.worker_processes)])
        self.performance_profiler = PerformanceProfiler()
        self.distributed_processor = DistributedProcessor(config.ray_address)
        self.session_manager = SessionManager(config.max_concurrent_users)
        
        # Thread pools
        self.io_executor = ThreadPoolExecutor(max_workers=20)
        self.cpu_executor = ProcessPoolExecutor(max_workers=config.worker_processes)
        
        # Monitoring
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        logger.info("Scalable system initialized", config=config.__dict__)
    
    async def process_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user request with full system optimization."""
        start_time = time.time()
        
        try:
            # Create or get session
            session_id = self._get_or_create_session(user_id)
            
            # Check cache first
            cache_key = self._generate_cache_key(request_data)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info("Cache hit", cache_key=cache_key)
                return cached_result
            
            # Memory pressure check
            if self.memory_manager.is_memory_pressure():
                self.memory_manager.optimize_memory()
            
            # Profile performance
            profile_name = self.performance_profiler.start_profiling(f"request_{user_id}")
            
            # Process request
            with self.memory_manager.memory_monitor("request_processing"):
                result = await self._process_request_core(request_data)
            
            # Cache result
            self.cache_manager.set(cache_key, result, ttl=3600)
            
            # Update session
            session_data = self.session_manager.get_session(session_id)
            if session_data:
                self.session_manager.update_session(session_id, {
                    'last_request': request_data,
                    'total_requests': session_data['request_count']
                })
            
            # Record metrics
            processing_time = time.time() - start_time
            self.request_times.append(processing_time)
            
            # Stop profiling
            profile_results = self.performance_profiler.stop_profiling(profile_name)
            
            logger.info(
                "Request processed successfully",
                user_id=user_id,
                processing_time=processing_time,
                memory_peak_gb=profile_results['peak_memory_mb'] / 1024
            )
            
            return result
            
        except Exception as e:
            self.error_counts[type(e).__name__] += 1
            logger.error("Request processing failed", user_id=user_id, error=str(e))
            raise
    
    async def _process_request_core(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core request processing logic."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate CPU-intensive work
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.cpu_executor,
            self._cpu_intensive_work,
            request_data
        )
        
        return result
    
    def _cpu_intensive_work(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate CPU-intensive work."""
        # Simulate some computation
        result = 0
        for i in range(1000):
            result += i * np.sin(i)
        
        return {
            "result": result,
            "processed_at": time.time(),
            "data_size": len(str(data))
        }
    
    def _get_or_create_session(self, user_id: str) -> str:
        """Get existing session or create new one."""
        # Check for existing session
        for session_id, session in self.session_manager.sessions.items():
            if session['user_id'] == user_id:
                return session_id
        
        # Create new session
        return self.session_manager.create_session(user_id, {})
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from request data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        memory_stats = self.memory_manager.get_memory_usage()
        cache_stats = self.cache_manager.get_stats()
        session_stats = self.session_manager.get_session_stats()
        cluster_stats = self.distributed_processor.get_cluster_stats()
        
        # Calculate average request time
        avg_request_time = np.mean(self.request_times) if self.request_times else 0
        
        return {
            "memory": memory_stats,
            "cache": cache_stats,
            "sessions": session_stats,
            "cluster": cluster_stats,
            "performance": {
                "avg_request_time": avg_request_time,
                "total_requests": len(self.request_times),
                "error_counts": dict(self.error_counts)
            },
            "system_health": {
                "memory_pressure": self.memory_manager.is_memory_pressure(),
                "cache_efficiency": cache_stats["hit_rate"],
                "session_utilization": session_stats["total_sessions"] / session_stats["max_sessions"]
            }
        }
    
    def optimize_system(self):
        """Perform system-wide optimization."""
        logger.info("Starting system optimization")
        
        # Memory optimization
        self.memory_manager.optimize_memory()
        
        # Session cleanup
        self.session_manager._cleanup_expired_sessions()
        
        # Cache optimization
        cache_stats = self.cache_manager.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            logger.warning("Low cache hit rate detected", hit_rate=cache_stats["hit_rate"])
        
        logger.info("System optimization completed")


# Example usage and demonstration
async def demonstrate_scalable_system():
    """Demonstrate the scalable system capabilities."""
    config = SystemConfig(
        max_memory_gb=4.0,
        max_concurrent_users=50,
        cache_size_mb=256,
        worker_processes=4
    )
    
    system = ScalableSystem(config)
    
    # Simulate multiple concurrent requests
    tasks = []
    for i in range(10):
        user_id = f"user_{i}"
        request_data = {"query": f"analysis_{i}", "parameters": {"depth": "comprehensive"}}
        task = system.process_request(user_id, request_data)
        tasks.append(task)
    
    # Wait for all requests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Get system statistics
    stats = system.get_system_stats()
    
    print("=== Scalable System Demonstration ===")
    print(f"Processed {len(results)} requests")
    print(f"Average request time: {stats['performance']['avg_request_time']:.3f}s")
    print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
    print(f"Memory usage: {stats['memory']['rss_gb']:.2f}GB")
    print(f"Active sessions: {stats['sessions']['total_sessions']}")
    
    return system, stats


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_scalable_system()) 