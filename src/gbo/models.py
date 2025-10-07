from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Boolean, Text
from sqlalchemy.sql import func
from database import Base

class House(Base):
    """Model for house/household data."""
    __tablename__ = 'homes'
    
    unique_id = Column(Integer, primary_key=True, autoincrement=True)
    short_alias = Column(String(50), nullable=False, unique=True)
    g_node_alias = Column(String(100), nullable=False, unique=True)
    address = Column(JSON, nullable=False)
    primary_contact = Column(JSON, nullable=False)
    secondary_contact = Column(JSON, nullable=True)
    hardware_layout = Column(JSON, nullable=True)
    alert_status = Column(JSON, nullable=False)
    representation_status = Column(JSON, nullable=True)
    scada_ip_address = Column(String(45), nullable=True)  # IPv4/IPv6
    scada_git_commit = Column(String(40), nullable=True)  # Git commit hash
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class User(Base):
    """Model for user authentication and management."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class HourlyElectricity(Base):
    """Model for hourly electricity consumption data."""
    __tablename__ = 'hourly_electricity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    g_node_alias = Column(String(100), nullable=False)
    short_alias = Column(String(50), nullable=False)
    hour_start_s = Column(Integer, nullable=False)
    kwh = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Add unique constraint
    __table_args__ = (
        {'extend_existing': True}
    )

class EnergyData(Base):
    """Model for general energy data"""
    __tablename__ = 'energy_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    g_node_alias = Column(String(100), nullable=False)
    short_alias = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    energy_type = Column(String(50), nullable=False)  # e.g., 'electricity', 'heat_pump'
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)  # e.g., 'kwh', 'watts'
    extra_data = Column(JSON, nullable=True)  # Additional data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SystemLog(Base):
    """Model for system logging and audit trails."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True)
    function = Column(String(100), nullable=True)
    user_id = Column(Integer, nullable=True)
    house_id = Column(Integer, nullable=True)
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
