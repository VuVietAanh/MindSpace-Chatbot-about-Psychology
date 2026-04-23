from dotenv import load_dotenv

load_dotenv()

import os

from db.models import init_db

url = os.getenv("DATABASE_URL")
if not url:
    print("❌ DATABASE_URL not found in .env")
    exit(1)

if "sqlite" in url:
    print("❌ Still using SQLite! Check your .env file")
    print(f"   Current URL: {url}")
    exit(1)

print(f"✅ Connecting to: {url[:50]}...")
init_db(url)
print("🎉 Supabase migration done!")