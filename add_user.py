from archive.auth_server import Session, users, get_password_hash, metadata, engine_gbo
from sqlalchemy import insert

def create_user():
    # Drop and recreate the users table
    metadata.drop_all(engine_gbo, tables=[users])
    metadata.create_all(engine_gbo, tables=[users])
    
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