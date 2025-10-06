"""
Database configuration and connection setup for the GridWorks Backoffice application.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("GBO_DB_URL", "postgresql://postgres:password@localhost/backofficedb")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

def get_db():
    """
    Dependency to get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_database():
    """
    Create the database if it doesn't exist.
    This function should be called before running migrations.
    """
    # For PostgreSQL, we need to connect to the default 'postgres' database first
    # to create our target database
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    # Parse the database URL to extract components
    from urllib.parse import urlparse
    parsed = urlparse(DATABASE_URL)
    
    # Connect to postgres database to create our target database
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database='postgres'  # Connect to default postgres database
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{parsed.path[1:]}'")
    exists = cursor.fetchone()
    
    if not exists:
        # Create the database
        cursor.execute(f"CREATE DATABASE {parsed.path[1:]}")
        print(f"Database '{parsed.path[1:]}' created successfully.")
    else:
        print(f"Database '{parsed.path[1:]}' already exists.")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_database()
