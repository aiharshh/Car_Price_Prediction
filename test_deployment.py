#!/usr/bin/env python3
"""
Pre-deployment test script for Render
Run this locally to ensure everything works before deploying
"""

import os
import sys
import requests
import subprocess
import time
import threading
from datetime import datetime

def test_model_training():
    """Test if model can be trained successfully"""
    print("🧪 Testing model training...")
    try:
        result = subprocess.run([sys.executable, 'train_model_safe.py'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Model training: PASSED")
            return True
        else:
            print(f"❌ Model training: FAILED\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Model training: ERROR - {str(e)}")
        return False

def test_app_startup():
    """Test if Flask app starts successfully"""
    print("🧪 Testing Flask app startup...")
    try:
        # Start Flask app in background
        process = subprocess.Popen([sys.executable, 'app.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Flask app startup: PASSED")
            process.terminate()
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Flask app startup: FAILED\n{stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Flask app startup: ERROR - {str(e)}")
        return False

def test_health_endpoint():
    """Test health check endpoint"""
    print("🧪 Testing health endpoint...")
    try:
        # Start Flask app in background for testing
        process = subprocess.Popen([sys.executable, 'app.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get('http://localhost:5000/health', timeout=10)
        
        if response.status_code == 200:
            print("✅ Health endpoint: PASSED")
            process.terminate()
            return True
        else:
            print(f"❌ Health endpoint: FAILED - Status {response.status_code}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Health endpoint: ERROR - {str(e)}")
        if 'process' in locals():
            process.terminate()
        return False

def test_prediction_api():
    """Test prediction API endpoint"""
    print("🧪 Testing prediction API...")
    try:
        # Start Flask app
        process = subprocess.Popen([sys.executable, 'app.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(5)
        
        # Test data
        test_data = {
            "kms_driven": 50000,
            "mileage_kmpl": 15.0,
            "engine_cc": 1500,
            "max_power_bhp": 120,
            "torque_nm": 200
        }
        
        # Make API call
        response = requests.post('http://localhost:5000/api/predict', 
                               json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'predicted_price' in result:
                print(f"✅ Prediction API: PASSED - Price: ₹{result['predicted_price']} lakhs")
                process.terminate()
                return True
            else:
                print("❌ Prediction API: FAILED - No prediction in response")
                process.terminate()
                return False
        else:
            print(f"❌ Prediction API: FAILED - Status {response.status_code}")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Prediction API: ERROR - {str(e)}")
        if 'process' in locals():
            process.terminate()
        return False

def test_file_existence():
    """Test if all required files exist"""
    print("🧪 Testing required files...")
    
    required_files = [
        'app.py',
        'train_model.py',
        'requirements.txt',
        'Procfile',
        'build.sh',
        'used_cars_filled.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}: EXISTS")
        else:
            print(f"❌ {file}: MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all pre-deployment tests"""
    print("🚀 Pre-Deployment Test Suite for Render")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("File Existence", test_file_existence),
        ("Model Training", test_model_training),
        ("App Startup", test_app_startup),
        ("Health Endpoint", test_health_endpoint),
        ("Prediction API", test_prediction_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for Render deployment!")
        print("\nNext steps:")
        print("1. Push to GitHub: git add . && git commit -m 'Ready for deployment' && git push")
        print("2. Deploy on Render: https://render.com")
        print("3. Monitor deployment logs")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
