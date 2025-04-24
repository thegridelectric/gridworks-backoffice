import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from gbo.named_types import HouseAddress, HouseContact, HouseStatus
from gjk.named_types import LayoutLite

Base = declarative_base()

class House(Base):
    __tablename__ = 'homes'
    short_alias = Column(String, nullable=False)
    address = Column(JSON, nullable=False)
    owner_contact = Column(JSON, nullable=False)
    hardware_layout = Column(JSON, nullable=True)
    unique_id = Column(Integer, primary_key=True)
    g_node_alias = Column(String, nullable=False)
    status = Column(JSON, nullable=False)
    scada_git_commit = Column(String, nullable=True)

def update_house(short_alias, new_house_data):
    
    load_dotenv()
    backoffice_db_url = os.getenv("GBO_DB_URL")
    engine = create_engine(backoffice_db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Find the house by short_alias
        house = session.query(House).filter(House.short_alias == short_alias).first()
        
        if not house:
            print(f"House with short_alias '{short_alias}' not found.")
            return False
        
        # Update the house with new data
        for key, value in new_house_data.items():
            if hasattr(house, key):
                setattr(house, key, value)
        
        # Commit the changes
        session.commit()
        print(f"Successfully updated house with short_alias '{short_alias}'.")
        return True
        
    except Exception as e:
        print(f"Error updating house: {e}")
        session.rollback()
        return False
    
    finally:
        session.close()

if __name__ == "__main__":

    new_house_data = {
        "status": HouseStatus(
            status="alert",
            message="zone2-garage is significantly below setpoint",
            acked=False
        ).to_dict()
    }

    # new_house_data = {
    #     "status": HouseStatus(
    #         status="ok",
    #     ).to_dict()
    # }
    
    update_house("oak", new_house_data) 