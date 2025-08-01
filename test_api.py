"""
API Testing Script for Car Price Prediction
Tests all API endpoints and validates responses
"""

import requests
import json
import time
from datetime import datetime

class APITester:
    """Comprehensive API testing for car price prediction service"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.test_results = {}
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            self.test_results['health_check'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'response': response.json()
            }
            
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            self.test_results['health_check'] = {'error': str(e)}
            return False
    
    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        print("ℹ️ Testing Model Info Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/model_info", timeout=10)
            
            self.test_results['model_info'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'response': response.json()
            }
            
            if response.status_code == 200:
                print("✅ Model info retrieved successfully")
                info = response.json()
                print(f"   Model loaded: {info.get('model_loaded', 'Unknown')}")
                print(f"   Features: {info.get('total_features', 'Unknown')}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model info error: {str(e)}")
            self.test_results['model_info'] = {'error': str(e)}
            return False
    
    def test_single_prediction_api(self):
        """Test single prediction API endpoint"""
        print("🔮 Testing Single Prediction API...")
        
        test_data = {
            "kms_driven": 45000,
            "mileage_kmpl": 18.5,
            "engine_cc": 1500,
            "max_power_bhp": 120,
            "torque_nm": 200
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            end_time = time.time()
            
            self.test_results['single_prediction_api'] = {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'test_data': test_data,
                'response': response.json()
            }
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result.get('predicted_price')
                print(f"✅ Prediction successful: ₹{predicted_price} lakhs")
                print(f"   Response time: {end_time - start_time:.3f} seconds")
                
                # Validate response structure
                required_fields = ['predicted_price', 'confidence_interval', 'input_features', 'timestamp']
                missing_fields = [field for field in required_fields if field not in result]
                
                if missing_fields:
                    print(f"⚠️ Missing fields in response: {missing_fields}")
                else:
                    print("✅ Response structure is complete")
                
                return True
            else:
                print(f"❌ Single prediction failed: {response.status_code}")
                print(f"   Error: {response.json().get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Single prediction error: {str(e)}")
            self.test_results['single_prediction_api'] = {'error': str(e)}
            return False
    
    def test_batch_prediction_api(self):
        """Test batch prediction API endpoint"""
        print("📦 Testing Batch Prediction API...")
        
        test_data = {
            "predictions": [
                {
                    "kms_driven": 25000,
                    "mileage_kmpl": 20.0,
                    "engine_cc": 1200,
                    "max_power_bhp": 85,
                    "torque_nm": 150
                },
                {
                    "kms_driven": 55000,
                    "mileage_kmpl": 15.5,
                    "engine_cc": 1800,
                    "max_power_bhp": 140,
                    "torque_nm": 250
                },
                {
                    "kms_driven": 35000,
                    "mileage_kmpl": 18.0,
                    "engine_cc": 1400,
                    "max_power_bhp": 110,
                    "torque_nm": 180
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/batch_predict",
                json=test_data,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            end_time = time.time()
            
            self.test_results['batch_prediction_api'] = {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'test_data': test_data,
                'response': response.json()
            }
            
            if response.status_code == 200:
                result = response.json()
                processed = result.get('total_processed', 0)
                errors = result.get('total_errors', 0)
                
                print(f"✅ Batch prediction successful")
                print(f"   Processed: {processed} predictions")
                print(f"   Errors: {errors} predictions")
                print(f"   Response time: {end_time - start_time:.3f} seconds")
                
                if processed > 0:
                    print("   Sample predictions:")
                    for i, pred in enumerate(result.get('results', [])[:2]):
                        print(f"     {i+1}. ₹{pred['predicted_price']} lakhs")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Error: {response.json().get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            self.test_results['batch_prediction_api'] = {'error': str(e)}
            return False
    
    def test_web_interface(self):
        """Test web interface accessibility"""
        print("🌐 Testing Web Interface...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            self.test_results['web_interface'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'content_length': len(response.text)
            }
            
            if response.status_code == 200:
                print("✅ Web interface accessible")
                print(f"   Content length: {len(response.text)} characters")
                
                # Check for key elements
                html_content = response.text.lower()
                key_elements = ['car price prediction', 'form', 'input', 'predict']
                found_elements = [elem for elem in key_elements if elem in html_content]
                
                print(f"   Found elements: {found_elements}")
                return True
            else:
                print(f"❌ Web interface failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Web interface error: {str(e)}")
            self.test_results['web_interface'] = {'error': str(e)}
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid data"""
        print("🚨 Testing Error Handling...")
        
        # Test with invalid data
        invalid_data_tests = [
            {
                'name': 'Missing fields',
                'data': {"kms_driven": 45000},
                'expected_status': 400
            },
            {
                'name': 'Negative values',
                'data': {
                    "kms_driven": -1000,
                    "mileage_kmpl": 18.5,
                    "engine_cc": 1500,
                    "max_power_bhp": 120,
                    "torque_nm": 200
                },
                'expected_status': 400
            },
            {
                'name': 'Unrealistic values',
                'data': {
                    "kms_driven": 2000000,  # Too high
                    "mileage_kmpl": 100,    # Too high
                    "engine_cc": 1500,
                    "max_power_bhp": 120,
                    "torque_nm": 200
                },
                'expected_status': 400
            }
        ]
        
        error_tests_passed = 0
        
        for test in invalid_data_tests:
            try:
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=test['data'],
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == test['expected_status']:
                    print(f"✅ {test['name']}: Properly handled error")
                    error_tests_passed += 1
                else:
                    print(f"❌ {test['name']}: Expected {test['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {test['name']}: Exception occurred - {str(e)}")
        
        self.test_results['error_handling'] = {
            'tests_passed': error_tests_passed,
            'total_tests': len(invalid_data_tests)
        }
        
        return error_tests_passed == len(invalid_data_tests)
    
    def test_performance(self):
        """Test API performance with multiple requests"""
        print("⚡ Testing Performance...")
        
        test_data = {
            "kms_driven": 45000,
            "mileage_kmpl": 18.5,
            "engine_cc": 1500,
            "max_power_bhp": 120,
            "torque_nm": 200
        }
        
        num_requests = 10
        response_times = []
        successful_requests = 0
        
        print(f"   Sending {num_requests} concurrent requests...")
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=test_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
            except Exception as e:
                print(f"   Request {i+1} failed: {str(e)}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"✅ Performance Test Results:")
            print(f"   Successful requests: {successful_requests}/{num_requests}")
            print(f"   Average response time: {avg_response_time:.3f} seconds")
            print(f"   Min response time: {min_response_time:.3f} seconds")
            print(f"   Max response time: {max_response_time:.3f} seconds")
            
            self.test_results['performance'] = {
                'successful_requests': successful_requests,
                'total_requests': num_requests,
                'avg_response_time': avg_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time
            }
            
            return successful_requests >= num_requests * 0.8  # 80% success rate
        else:
            print("❌ No successful responses for performance testing")
            return False
    
    def run_comprehensive_tests(self):
        """Run all tests and generate report"""
        print("🧪 STARTING COMPREHENSIVE API TESTING")
        print("=" * 60)
        
        start_time = datetime.now()
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_model_info_endpoint),
            ("Web Interface", self.test_web_interface),
            ("Single Prediction API", self.test_single_prediction_api),
            ("Batch Prediction API", self.test_batch_prediction_api),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * 40)
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"❌ Test failed with exception: {str(e)}")
                results.append((test_name, False))
        
        end_time = datetime.now()
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY REPORT")
        print("=" * 60)
        
        passed_tests = sum(1 for _, success in results if success)
        total_tests = len(results)
        
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Tests failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total time: {(end_time - start_time).total_seconds():.2f} seconds")
        
        print(f"\nDetailed Results:")
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name:<25} {status}")
        
        # Save detailed results
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'test_duration': (end_time - start_time).total_seconds(),
            'test_timestamp': start_time.isoformat()
        }
        
        # Save to file
        with open('api_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: api_test_results.json")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! API is ready for production.")
        else:
            print("⚠️ Some tests failed. Please review the issues before deployment.")
        
        return passed_tests == total_tests

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Car Price Prediction API')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Base URL of the API (default: http://localhost:5000)')
    
    args = parser.parse_args()
    
    print(f"🎯 Testing API at: {args.url}")
    
    tester = APITester(args.url)
    success = tester.run_comprehensive_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
