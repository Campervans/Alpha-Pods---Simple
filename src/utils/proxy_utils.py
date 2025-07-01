#!/usr/bin/env python3
"""
Proxy utilities for loading and managing the proxy list from CSV.
"""

import csv
import os
import random
from typing import List, Dict, Optional
from pathlib import Path

# Proxy credentials - these remain the same as before
PROXY_USERNAME = 'sp7lr99xhd'
PROXY_PASSWORD = '7Xtywa2k3o0oxoViLX'

def load_proxies_from_csv(csv_path: str = None) -> List[int]:
    """
    Load proxy ports from the CSV file.
    
    Args:
        csv_path: Path to the proxy CSV file. If None, uses default location.
        
    Returns:
        List of proxy ports
    """
    if csv_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        csv_path = project_root / "data" / "proxies" / "proxies.csv"
    
    proxy_ports = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0]:  # Skip empty rows
                    # Format: dc.decodo.com:port::<password>
                    parts = row[0].split(':')
                    if len(parts) >= 2:
                        try:
                            port = int(parts[1])
                            proxy_ports.append(port)
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Warning: Proxy CSV file not found at {csv_path}")
        # Fallback to original 30 proxies
        proxy_ports = list(range(10001, 10031))
    except Exception as e:
        print(f"Warning: Error reading proxy CSV: {e}")
        # Fallback to original 30 proxies
        proxy_ports = list(range(10001, 10031))
    
    if not proxy_ports:
        print("Warning: No proxies loaded, using fallback")
        proxy_ports = list(range(10001, 10031))
    
    print(f"Loaded {len(proxy_ports)} proxies from {csv_path}")
    return proxy_ports


def get_proxy_ports() -> List[int]:
    """Get the current list of proxy ports."""
    return load_proxies_from_csv()


def get_random_proxy() -> Dict[str, str]:
    """Get a random proxy configuration from the loaded list."""
    ports = get_proxy_ports()
    port = random.choice(ports)
    proxy_url = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@dc.decodo.com:{port}"
    return {
        'http': proxy_url,
        'https': proxy_url
    }


def get_proxy_for_port(port: int) -> Dict[str, str]:
    """Get proxy configuration for a specific port."""
    proxy_url = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@dc.decodo.com:{port}"
    return {
        'http': proxy_url,
        'https': proxy_url
    }


def test_proxy(proxy_dict: Dict[str, str]) -> bool:
    """Test if a proxy is working."""
    try:
        import requests
        response = requests.get('https://ip.decodo.com/json', 
                              proxies=proxy_dict, 
                              timeout=5)
        return response.status_code == 200
    except:
        return False


# Cache the proxy ports to avoid reading the file multiple times
_CACHED_PROXY_PORTS = None

def get_cached_proxy_ports() -> List[int]:
    """Get cached proxy ports, loading from CSV on first call."""
    global _CACHED_PROXY_PORTS
    if _CACHED_PROXY_PORTS is None:
        _CACHED_PROXY_PORTS = load_proxies_from_csv()
    return _CACHED_PROXY_PORTS


def refresh_proxy_cache():
    """Force refresh of the proxy cache."""
    global _CACHED_PROXY_PORTS
    _CACHED_PROXY_PORTS = None 