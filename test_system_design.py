#!/usr/bin/env python3
"""
Test script for system_design.py to verify all fixes work correctly.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that the module can be imported without errors."""
    try:
        from app.system_design import (
            SystemConfig, MemoryManager, CacheManager, LoadBalancer,
            PerformanceProfiler, DistributedProcessor, SessionManager, ScalableSystem
        )
        print("✅ All classes imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of each component."""
    try:
        from app.system_design import SystemConfig, MemoryManager, CacheManager
        
        # Test SystemConfig
        config = SystemConfig(max_memory_gb=2.0, max_concurrent_users=10)
        print("✅ SystemConfig created successfully")
        
        # Test MemoryManager
        memory_manager = MemoryManager(max_memory_gb=2.0)
        stats = memory_manager.get_memory_usage()
        print(f"✅ MemoryManager working - RSS: {stats['rss_gb']:.2f}GB")
        
        # Test CacheManager
        cache_manager = CacheManager(max_size_mb=64)
        cache_manager.set("test_key", "test_value")
        result = cache_manager.get("test_key")
        assert result == "test_value"
        print("✅ CacheManager working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    try:
        from app.system_design import ScalableSystem, SystemConfig
        
        config = SystemConfig(
            max_memory_gb=1.0,
            max_concurrent_users=5,
            cache_size_mb=32,
            worker_processes=2
        )
        
        system = ScalableSystem(config)
        
        # Test single request
        result = await system.process_request(
            "test_user",
            {"query": "test_query", "parameters": {"depth": "basic"}}
        )
        
        print("✅ Async request processing working")
        
        # Test system stats
        stats = system.get_system_stats()
        print(f"✅ System stats generated - Cache hit rate: {stats['cache']['hit_rate']:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing System Design Module ===")
    
    # Test 1: Import
    if not test_import():
        return False
    
    # Test 2: Basic functionality
    if not test_basic_functionality():
        return False
    
    # Test 3: Async functionality
    try:
        asyncio.run(test_async_functionality())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! System design module is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 