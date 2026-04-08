from sqlalchemy import Table, Column, String, JSON, Enum, create_engine, MetaData, text, DateTime
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
import bcrypt
import os
from dotenv import load_dotenv
from enum import StrEnum, auto
from datetime import datetime, timezone


class userType(StrEnum):
    admin = auto()
    owner = auto()
    viewer = auto()

# ------------------------------------------------------------
USERNAME = ''
PASSWORD = ''
EMAIL = ''
USER_TYPE = userType.admin
USER_INSTALLATIONS = []
DELETE_EXISTING_TABLE = False
# ------------------------------------------------------------

load_dotenv()
database_url = os.getenv("GBO_DB_URL_NO_ASYNC")
engine_gbo = create_engine(database_url)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("username", String, nullable=False, unique=True, primary_key=True),
    Column("hashed_password", String, nullable=False),
    Column("email", String, nullable=False, unique=True),
    Column("user_type", Enum(userType, name="user_type_enum", native_enum=False), nullable=False),
    Column("user_installations", JSON, nullable=False, default=list, server_default=text("'[]'")),
    Column("last_login", DateTime, nullable=True),
)

def get_password_hash(password):
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

Session = sessionmaker(bind=engine_gbo)

def create_user():
    # Drop and recreate the users table
    if DELETE_EXISTING_TABLE:
        metadata.drop_all(engine_gbo, tables=[users], checkfirst=True)
        metadata.create_all(engine_gbo, tables=[users], checkfirst=True)
        
    session = Session()
    
    user = {
        "username": USERNAME,
        "hashed_password": get_password_hash(PASSWORD),
        "email": EMAIL,
        "user_type": USER_TYPE,
        "user_installations": USER_INSTALLATIONS,
        "last_login": datetime.now(timezone.utc),
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