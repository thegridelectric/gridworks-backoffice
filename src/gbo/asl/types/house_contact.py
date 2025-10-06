from typing import Literal, Optional
from gbo.asl.codec import AslType

class HouseContact(AslType):
    first_name: str
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    type_name: Literal["gw0.house.contact"] = "gw0.house.contact"