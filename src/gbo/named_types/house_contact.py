from pydantic import BaseModel
from typing import Optional

class HouseContact(BaseModel ):
    first_name: str
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    
    def to_dict(self):
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone": self.phone,
            "email": self.email
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)