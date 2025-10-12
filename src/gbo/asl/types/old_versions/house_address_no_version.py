from typing import Literal
from gbo.asl.codec import AslType
from gbo.asl.types import HouseAddress

class HouseAddressNoVersion(AslType):
    street: str
    city: str
    state: str
    zip: str
    country: str
    latitude: float
    longitude: float
    type_name: Literal["gw0.house.address"] = "gw0.house.address"

    def to_latest(self) -> "HouseAddress":
        """Convert to the latest version."""
        return HouseAddress(
            street=self.street,
            city=self.city,
            state=self.state,
            zip=self.zip,
            country=self.country,
            latitude_micro_deg=int(self.latitude * 1_000_000),
            longitude_micro_deg=int(self.longitude * 1_000_000),
        )

