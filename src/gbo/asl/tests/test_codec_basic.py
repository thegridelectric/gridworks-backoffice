"""Basic tests to demonstrate ASL functionality"""

import pytest
from gbo.asl import AslCodec
from gbo.asl.types import HouseAddress, HouseStatus
from gbo.asl.types.old_versions import HouseStatusNoVersion
from gbo.asl.enums import RepresentationStatus


def test_codec_initialization():
    """Test that codec initializes with discovered types."""
    codec = AslCodec()
    assert "gw0.house.address" in codec.registry
    assert "gw0.house.contact" in codec.registry
    assert "gw0.house.status" in codec.registry


def test_round_trip():
    """Test encoding and decoding works"""
    codec = AslCodec()

    # Create a message
    original = HouseAddress(
        street="43 Avon St",
        city="New Haven",
        state="CT",
        zip="06511",
        country="USA",
        latitude_micro_deg=41320535,
        longitude_micro_deg=-72911723
    )

    wire_dict = original.to_dict()
    # test wire format uses Pascal case
    assert "Street" in wire_dict

    encoded = codec.to_bytes(original)
    decoded = codec.from_bytes(encoded)

    assert decoded == original

    decoded2 = codec.from_dict(wire_dict)

    assert decoded2 == decoded

    # "no version" original
    data = {
        "Street": "43 Avon St",
        "City": "New Haven",
        "State": "CT",
        "Zip": "06511",
        "Country": "USA",
        "Latitude": 41.320535,
        "Longitude": -72.911723,
        "TypeName": "gw0.house.address",
    }

    assert codec.from_dict(data) == decoded

    # "no version" original, statu
    status = HouseStatusNoVersion(
        status="StrangeStatus"
    )
    
    updated = codec.from_dict(status.to_dict())
    # translates to a RepresentationStatus enum
    assert updated.status == RepresentationStatus.Unknown
