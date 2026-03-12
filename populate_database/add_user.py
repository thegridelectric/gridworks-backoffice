from sqlalchemy import Table, create_engine, MetaData
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import declarative_base, sessionmaker
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()
database_url = os.getenv("GBO_DB_URL_NO_ASYNC")
engine_gbo = create_engine(database_url)

Base = declarative_base()
metadata = MetaData()

users = Table('users', metadata, autoload_with=engine_gbo)

def get_password_hash(password):
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

Session = sessionmaker(bind=engine_gbo)

def create_user():
    # Drop and recreate the users table
    # metadata.drop_all(engine_gbo, tables=[users])
    # metadata.create_all(engine_gbo, tables=[users])
    
    session = Session()
    
    user = {
        "username": "user",
        "email": "test@example.com",
        "hashed_password": get_password_hash("password"),
        "is_active": True
    }
    
    stmt = insert(users).values(**user)
    # If a row with this email exists, overwrite all provided fields.
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["email"],
        set_={key: getattr(stmt.excluded, key) for key in user.keys()},
    )

    session.execute(upsert_stmt)
    session.commit()
    print("User upserted successfully")
    session.close()

if __name__ == "__main__":
    create_user() 