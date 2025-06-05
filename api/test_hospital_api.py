import requests

def test_hospital_api():
    """Test the hospital API endpoints"""
    # Base URL - change this if your server runs on a different port/host
    base_url = "http://localhost:8135/api/hospitals"
    
    # Test 1: Basic endpoint with no parameters
    response = requests.get(base_url)
    print("Test 1: Basic endpoint")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {data['total_results']} hospitals")
        if data['total_results'] > 0:
            print(f"First hospital: {data['hospitals'][0]['name']}")
    else:
        print(f"Failed with status code: {response.status_code}")
    
    # Test 2: Filter by location
    response = requests.get(f"{base_url}?location=san+diego")
    print("\nTest 2: Filter by location 'san diego'")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {data['total_results']} hospitals in San Diego")
    else:
        print(f"Failed with status code: {response.status_code}")
    
    # Test 3: Filter by specialty
    response = requests.get(f"{base_url}?specialty=oncology")
    print("\nTest 3: Filter by specialty 'oncology'")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {data['total_results']} hospitals with oncology specialty")
    else:
        print(f"Failed with status code: {response.status_code}")
    
    # Test 4: Multiple filters
    response = requests.get(f"{base_url}?location=ca&specialty=cardiology&rating=4.5")
    print("\nTest 4: Multiple filters")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Found {data['total_results']} hospitals matching all criteria")
    else:
        print(f"Failed with status code: {response.status_code}")

if __name__ == "__main__":
    test_hospital_api()