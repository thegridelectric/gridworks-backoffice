from sqlalchemy import insert
from sqlalchemy import Table, create_engine, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.context import CryptContext
import dotenv

engine_gbo = create_engine(dotenv.get("GBO_DB_URL"))
Base = declarative_base()
metadata = MetaData()

users = Table('users', metadata, autoload_with=engine_gbo)

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
    bcrypt__ident="2b"
)

def get_password_hash(password):
    return pwd_context.hash(password)

Session = sessionmaker(bind=engine_gbo)

def create_user():
    # Drop and recreate the users table
    # metadata.drop_all(engine_gbo, tables=[users])
    # metadata.create_all(engine_gbo, tables=[users])
    
    session = Session()
    
    user = {
        "username": "USERNAME HERE",
        "email": "EMAIL HERE",
        "password_hash": get_password_hash("PASSWORD HERE"),
        "is_active": True
    }
    
    session.execute(insert(users).values(**user))
    session.commit()
    print("User table recreated and test user created successfully")
    session.close()

if __name__ == "__main__":
    create_user() 