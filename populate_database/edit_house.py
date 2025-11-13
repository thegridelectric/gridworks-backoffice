import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from gbo.named_types import HouseAddress, HouseContact, HouseStatus
from gjk.named_types import LayoutLite
from gbo.enums.representation_status import RepresentationStatus

load_dotenv()
backoffice_db_url = os.getenv("GBO_DB_URL")
engine = create_engine(backoffice_db_url)
Base = declarative_base()

class House(Base):
    __tablename__ = 'homes'
    short_alias = Column(String, nullable=False)
    address = Column(JSON, nullable=False)
    primary_contact = Column(JSON, nullable=False)
    secondary_contact = Column(JSON, nullable=True)
    hardware_layout = Column(JSON, nullable=True)
    unique_id = Column(Integer, primary_key=True)
    g_node_alias = Column(String, nullable=False)
    alert_status = Column(JSON, nullable=False)
    representation_status = Column(JSON, nullable=True)
    scada_ip_address = Column(String, nullable=True)
    scada_git_commit = Column(String, nullable=True)

Session = sessionmaker(bind=engine)
session = Session()

GNODE_ALIAS = "hw1.isone.me.versant.keene.elm"

house_to_edit = session.query(House).filter_by(g_node_alias=GNODE_ALIAS).first()

if house_to_edit:
    house_to_edit.primary_contact['last_name'] = "DiBona"
    house_to_edit.secondary_contact['last_name'] = "DiBona"
    session.commit()
    print(f"Done editing {GNODE_ALIAS}.")
else:
    print(f"House with g_node_alias {GNODE_ALIAS} not found.")
