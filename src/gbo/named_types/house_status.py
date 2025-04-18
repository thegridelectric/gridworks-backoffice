from pydantic import BaseModel
from typing import Optional

class HouseStatus(BaseModel):
    status: str
    message: Optional[str] = None
    acked: Optional[bool] = None
    acked_by: Optional[str] = None
    acked_at: Optional[str] = None

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
