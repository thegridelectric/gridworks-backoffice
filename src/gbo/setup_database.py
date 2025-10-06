#!/usr/bin/env python3
"""
Script to set up the backofficedb database with Alembic migrations.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to set up the database."""
    print("Setting up backofficedb database...")
    
    # Check if GBO_DB_URL is set
    db_url = os.getenv("GBO_DB_URL")
    if not db_url:
        print("Error: GBO_DB_URL environment variable is not set.")
        print("Please set it to your PostgreSQL connection string, e.g.:")
        print("export GBO_DB_URL='postgresql://postgres:password@localhost/backofficedb'")
        sys.exit(1)
    
    print(f"Using database URL: {db_url}")
    
    # Create the database if it doesn't exist
    try:
        from database import create_database
        create_database()
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)
    
    print("Database setup complete!")
    print("\nNext steps:")
    print("1. Run: cd src/gbo && uv run alembic revision --autogenerate -m 'Initial migration'")
    print("2. Run: cd src/gbo && uv run alembic upgrade head")
    print("\nTo create new migrations in the future:")
    print("1. Modify your models in models.py")
    print("2. Run: cd src/gbo && uv run alembic revision --autogenerate -m 'Description of changes'")
    print("3. Run: cd src/gbo && uv run alembic upgrade head")

if __name__ == "__main__":
    main()
