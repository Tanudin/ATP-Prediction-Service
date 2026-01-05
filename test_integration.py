"""
Quick test script to verify upcoming matches integration.
Run this to test the setup before deploying to GitHub Actions.
"""

import sys
from pathlib import Path

print("🎾 ATP Prediction Service - Integration Test\n")
print("=" * 60)

# Test 1: Check dependencies
print("\n1️⃣ Testing Dependencies...")
try:
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    print("   ✅ requests installed")
    print("   ✅ pandas installed")
    print("   ✅ beautifulsoup4 installed")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Check scripts exist
print("\n2️⃣ Testing Script Files...")
required_files = [
    "fetch_upcoming_matches.py",
    "tennis-inference-enhanced.py",
    ".github/workflows/daily_update.yml",
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ Missing: {file_path}")

# Test 3: Test API Connection
print("\n3️⃣ Testing SofaScore API Connection...")
try:
    url = "https://api.sofascore.com/api/v1/sport/tennis/events/live"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=5)
    
    if response.status_code == 200:
        print(f"   ✅ SofaScore API accessible (status: {response.status_code})")
        data = response.json()
        events = data.get("events", [])
        print(f"   ℹ️  Currently {len(events)} live ATP matches")
    else:
        print(f"   ⚠️  API returned status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Connection failed: {e}")

# Test 4: Import custom modules
print("\n4️⃣ Testing Custom Module Imports...")
try:
    from fetch_upcoming_matches import UpcomingMatchesFetcher
    print("   ✅ fetch_upcoming_matches imported successfully")
    
    # Try to instantiate
    fetcher = UpcomingMatchesFetcher()
    print("   ✅ UpcomingMatchesFetcher instantiated")
except Exception as e:
    print(f"   ❌ Import error: {e}")

# Test 5: Quick fetch test (optional)
print("\n5️⃣ Testing Upcoming Matches Fetch (Sample)...")
try:
    from datetime import datetime, timedelta
    from fetch_upcoming_matches import UpcomingMatchesFetcher
    
    fetcher = UpcomingMatchesFetcher()
    
    # Try to fetch just one day
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    
    url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        events = data.get("events", [])
        atp_events = [
            e for e in events 
            if e.get("tournament", {}).get("category", {}).get("name") == "ATP"
        ]
        print(f"   ✅ Found {len(atp_events)} ATP matches for {date_str}")
    else:
        print(f"   ⚠️  No data for {date_str} (status: {response.status_code})")
        
except Exception as e:
    print(f"   ❌ Fetch test failed: {e}")

# Test 6: Check environment variables
print("\n6️⃣ Checking Environment Variables...")
import os
from dotenv import load_dotenv

load_dotenv()

env_vars = {
    "HOPSWORKS_API_KEY": os.getenv("HOPSWORKS_API_KEY"),
    "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
    "KAGGLE_KEY": os.getenv("KAGGLE_KEY"),
}

for var_name, var_value in env_vars.items():
    if var_value:
        print(f"   ✅ {var_name} is set")
    else:
        print(f"   ⚠️  {var_name} is NOT set (required for full pipeline)")

# Summary
print("\n" + "=" * 60)
print("📊 Test Summary")
print("=" * 60)
print("""
Next Steps:
1. If all tests passed ✅, you're ready to deploy!
2. Push changes to GitHub:
   git add .
   git commit -m "Add upcoming matches integration"
   git push

3. The GitHub Action will run automatically:
   - Daily at 3 AM UTC (post-match update)
   - Daily at 9 AM UTC (upcoming matches)
   - Or trigger manually via GitHub Actions tab

4. Monitor the first run in GitHub Actions tab

5. Check for output files:
   - upcoming_matches.csv
   - tennis_predictions_enhanced.csv
   - last_prediction_update.txt
""")

print("\n✅ Integration test complete!")
