from pydantic import BaseModel

class HouseContact(BaseModel ):
    name: str
    phone: str
    email: str
    
    def to_dict(self):
        return {
            "name": self.name,
            "phone": self.phone,
            "email": self.email
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)