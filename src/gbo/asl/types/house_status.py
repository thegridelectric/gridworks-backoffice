from typing import Literal, Optional
from gbo.asl.codec import AslType
from gbo.asl.enums import RepresentationStatus

class HouseStatus(AslType):
    status: RepresentationStatus
    message: Optional[str] = None
    acked: Optional[bool] = None
    acked_by: Optional[str] = None
    acked_at: Optional[str] = None
    type_name: Literal["gw0.house.status"] = "gw0.house.status"
    version: Literal["000"] = "000"