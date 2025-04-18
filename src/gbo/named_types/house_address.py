from pydantic import BaseModel

class HouseAddress(BaseModel):
    street: str
    city: str
    state: str
    zip: str
    country: str
    latitude: float
    longitude: float

    def to_dict(self):
        return {
            "street": self.street,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)