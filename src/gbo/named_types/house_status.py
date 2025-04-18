from pydantic import BaseModel

class HouseStatus(BaseModel):
    status: str
    message: str
    acked: str
    acked_by: str
    acked_at: str

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "acked": self.acked,
            "acked_by": self.acked_by,
            "acked_at": self.acked_at
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    