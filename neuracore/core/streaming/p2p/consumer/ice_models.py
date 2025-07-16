"""Models for ICE (Interactive Connectivity Establishment) configuration.

This module defines Pydantic models for representing ICE server configurations,
which are used in WebRTC to establish peer-to-peer connections.
"""

from typing import List, Optional, Union

from pydantic import AnyUrl, BaseModel, Field


class IceServer(BaseModel):
    """Represents an ICE server configuration for WebRTC connection establishment.

    Attributes:
        urls: A URL or list of URLs for the ICE server(s).
        username: Optional username for the ICE server, if authentication is required.
        credential: Optional credential or password for the ICE server.
    """

    urls: Union[AnyUrl, List[AnyUrl]]
    username: Optional[str] = Field(default=None)
    credential: Optional[str] = Field(default=None)


class IceConfig(BaseModel):
    """ICE configuration schema containing a list of ICE servers.

    Attributes:
        iceServers: A list of ICE server configurations.
    """

    iceServers: List[IceServer]
