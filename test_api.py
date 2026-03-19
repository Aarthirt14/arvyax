#!/usr/bin/env python3
"""
Test script for  API and integrated system
Tests all endpoints and functionality
"""

import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8000"
TIMEOUT = 10


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_health():
    """Test health check endpoint"""
    print_section("TEST 1: HEALTH CHECK")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_info():
    """Test info endpoint"""
    print_section("TEST 2: SYSTEM INFO")
    try:
        response = requests.get(f"{API_BASE}/info", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"System: {data.get('name')}")
        print(f"Version: {data.get('version')}")
        print(f"Tagline: {data.get('tagline')}")
        print(f"\nFeatures:")
        for feature in data.get('features', []):
            print(f"  • {feature}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print_section("TEST 3: SINGLE PREDICTION")
    
    test_cases = [
        {
            "name": "Positive case - energized",
            "data": {
                "journal_text": "Had an amazing morning workout! Feeling energized and ready to tackle the day. My mind is clear and I feel motivated.",
                "stress_level": 1,
                "energy_level": 5,
                "sleep_hours": 8,
                "time_of_day": "morning",
                "ambience_type": "forest",
                "reflection_quality": "high"
            }
        },
        {
            "name": "Negative case - overwhelmed",
            "data": {
                "journal_text": "So much work piling up and I can't seem to focus. Feeling drowsy and overwhelmed by everything happening at once.",
                "stress_level": 5,
                "energy_level": 1,
                "sleep_hours": 4,
                "time_of_day": "evening",
                "ambience_type": "urban",
                "reflection_quality": "low"
            }
        },
        {
            "name": "Mixed case - calm but tired",
            "data": {
                "journal_text": "Just relaxing at home. Feeling peaceful but a bit tired from the day. Need some quiet time to recharge.",
                "stress_level": 2,
                "energy_level": 2,
                "sleep_hours": 6,
                "time_of_day": "night",
                "ambience_type": "home",
                "reflection_quality": "medium"
            }
        },
        {
            "name": "Short ambiguous case",
            "data": {
                "journal_text": "OK",
                "stress_level": 3,
                "energy_level": 3,
                "sleep_hours": 6,
                "time_of_day": "afternoon",
                "ambience_type": "café",
                "reflection_quality": "low"
            }
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(
                f"{API_BASE}/predict",
                json=test_case['data'],
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                pred = response.json()
                print(f"  ✓ REQUEST SUCCESSFUL")
                print(f"  State: {pred['predicted_state']}")
                print(f"  Intensity: {pred['predicted_intensity']}/5")
                print(f"  Confidence: {pred['confidence']*100:.1f}%")
                print(f"  Action: {pred['what_to_do']}")
                print(f"  Timing: {pred['when_to_do']}")
                print(f"  Message: \"{pred['supportive_message'][:100]}...\"")
                print(f"  Uncertain: {'Yes' if pred['uncertain_flag'] else 'No'}")
                results.append(True)
            else:
                print(f"  ✗ FAILED with status {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                results.append(False)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(False)
    
    return all(results)


def test_stats():
    """Test stats endpoint"""
    print_section("TEST 4: STATISTICS")
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        
        print(f"\nTotal Predictions: {data.get('total_predictions')}")
        print(f"Average Confidence: {data.get('average_confidence')*100:.1f}%")
        print(f"Uncertain Rate: {data.get('uncertain_rate')*100:.1f}%")
        
        print(f"\nState Distribution:")
        for state, count in data.get('state_distribution', {}).items():
            print(f"  • {state}: {count}")
        
        print(f"\nTop Recommendations:")
        for action, count in data.get('top_recommendations', {}).items():
            print(f"  • {action}: {count}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ui():
    """Test UI endpoint"""
    print_section("TEST 5: WEB UI")
    try:
        response = requests.get(f"{API_BASE}/", timeout=TIMEOUT)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            # Check if HTML content is returned
            if "<!DOCTYPE" in response.text or "<html" in response.text:
                print(f"✓ HTML UI served successfully")
                print(f"Content length: {len(response.text)} bytes")
                # Check for key elements
                checks = [
                    ("Form", "predictionForm" in response.text),
                    ("Sliders", "stressSlider" in response.text),
                    ("Textarea", "journalText" in response.text),
                    ("Results", "resultValue" in response.text),
                    ("API Integration", "API_URL" in response.text),
                ]
                print("\nUI Components:")
                for component, found in checks:
                    status = "✓" if found else "✗"
                    print(f"  {status} {component}")
                return all(found for _, found in checks)
            else:
                print(f"✗ Response is not HTML")
                return False
        else:
            print(f"✗ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_cors():
    """Test CORS headers"""
    print_section("TEST 6: CORS SUPPORT")
    try:
        response = requests.options(f"{API_BASE}/predict", timeout=TIMEOUT)
        
        headers = response.headers
        print("CORS Headers:")
        cors_headers = [
            ("Access-Control-Allow-Origin", headers.get("access-control-allow-origin")),
            ("Access-Control-Allow-Methods", headers.get("access-control-allow-methods")),
            ("Access-Control-Allow-Headers", headers.get("access-control-allow-headers")),
        ]
        
        for header, value in cors_headers:
            if value:
                print(f"  ✓ {header}: {value[:60]}...")
            else:
                print(f"  ? {header}: Not set")
        
        return bool(headers.get("access-control-allow-origin"))
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("   API TEST SUITE")
    print("="*70)
    print(f"\nAPI Base URL: {API_BASE}")
    print(f"Timeout: {TIMEOUT}s")
    
    # Wait a moment for server detection
    print("\nAttempting to reach server...")
    
    tests = [
        ("Health Check", test_health),
        ("System Info", test_info),
        ("Single Predictions", test_single_prediction),
        ("Statistics", test_stats),
        ("Web UI", test_ui),
        ("CORS Support", test_cors),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            results[name] = False
    
    # Print summary
    print_section("TEST SUMMARY")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(1 for p in results.values() if p)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Server is ready for use.")
    else:
        print(f"\n✗ {total - passed} tests failed. Please check server configuration.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
