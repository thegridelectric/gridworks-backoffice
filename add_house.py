import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
backoffice_db_url = os.getenv("GBO_DB_URL")
engine = create_engine(backoffice_db_url)
Base = declarative_base()

class House(Base):
    __tablename__ = 'homes'
    short_alias = Column(String, nullable=False)
    address = Column(JSON, nullable=False)
    owner_contact = Column(JSON, nullable=False)
    hardware_layout = Column(JSON, nullable=True)
    unique_id = Column(Integer, primary_key=True)
    g_node_alias = Column(String, nullable=False)
    status = Column(String, nullable=False)

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

beech = House(
    short_alias="beech",
    address={
        "street": "45 Somerset St",
        "city": "Millinocket",
        "state": "ME",
        "zip": "04462",
        "country": "USA",
        "latitude": 45.65032,
        "longitude": -68.71284
    },
    owner_contact={
        "name": "Paul Moscone",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    },
    hardware_layout=None,
    g_node_alias="beech",
    status="ok"
)

oak = House(
    short_alias="oak",
    address={
        "street": "114 Lincoln St",
        "city": "Millinocket",
        "state": "ME",
        "zip": "04462",
        "country": "USA",
        "latitude": 45.65306,
        "longitude": -68.71281
    },
    owner_contact={
        "name": "Brennan Turner",
        "email": "example@example.com",
        "phone": "123-456-7890"
    },
    hardware_layout=None,
    g_node_alias="oak",
    status="ok"
)

fir = House(
    short_alias="fir",
    address={
        "street": "1230 Medway Rd",
        "city": "Millinocket",
        "state": "ME",
        "zip": "00130",
        "country": "USA",
        "latitude": 45.64405,
        "longitude": -68.64849
    },
    owner_contact={
        "name": "Renee St. Jean",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    },
    hardware_layout=None,
    g_node_alias="fir",
    status="ok"
)

maple = House(
    short_alias="maple",
    address={
        "street": "123 Main St",
        "city": "Millinocket",
        "state": "ME",
        "zip": "12345",
        "country": "USA",
        "latitude": 45.6387,
        "longitude": -68.6719
    },
    owner_contact={
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    },
    hardware_layout=None,
    g_node_alias="maple",
    status="ok"
)

session.add_all([beech, oak, fir, maple])
session.commit()

print("Inserted 4 homes into backofficedb.")