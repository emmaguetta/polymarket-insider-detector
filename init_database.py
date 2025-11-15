"""
Initialize database tables
"""
from database.models import init_db

print("Initializing database...")
engine = init_db()
print("✅ Database tables created successfully!")
print("\nTables created:")
print("- markets")
print("- wallets")
print("- trades")
print("- suspicious_transactions")
print("- known_insider_cases")
