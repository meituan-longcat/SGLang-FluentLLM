from __future__ import annotations

import ipaddress
import logging
import os
import socket
from dataclasses import dataclass

import psutil
import zmq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZMQ helpers
# ---------------------------------------------------------------------------


def config_socket(socket, socket_type: zmq.SocketType):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")


def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: str | None = None,
    bind: bool = True,
) -> zmq.Socket | tuple[int, zmq.Socket]:
    """Create and configure a ZeroMQ socket.

    Args:
        context: ZeroMQ context to create the socket from.
        socket_type: Type of ZeroMQ socket to create.
        endpoint: Optional endpoint to bind/connect to. If None, binds to a random TCP port.
        bind: Whether to bind (True) or connect (False) to the endpoint. Ignored if endpoint is None.

    Returns:
        If endpoint is None: Tuple of (port, socket) where port is the randomly assigned TCP port.
        If endpoint is provided: The configured ZeroMQ socket.
    """
    sock = context.socket(socket_type)

    if endpoint is None:
        # Bind to random TCP port
        config_socket(sock, socket_type)
        port = sock.bind_to_random_port("tcp://*")
        return port, sock
    else:
        # Handle IPv6 if endpoint contains brackets
        if endpoint.find("[") != -1:
            sock.setsockopt(zmq.IPV6, 1)

        config_socket(sock, socket_type)

        if bind:
            sock.bind(endpoint)
        else:
            sock.connect(endpoint)

        return sock


# ---------------------------------------------------------------------------
# Local IP detection helpers
# ---------------------------------------------------------------------------


def get_local_ip_by_nic(interface: str = None) -> str | None:
    if not (interface := interface or os.environ.get("SGLANG_LOCAL_IP_NIC", None)):
        return None
    try:
        import netifaces
    except ImportError as e:
        raise ImportError(
            "Environment variable SGLANG_LOCAL_IP_NIC requires package netifaces, please install it through 'pip install netifaces'"
        ) from e

    try:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                ip = addr_info.get("addr")
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":  # noqa: S104
                    return ip
        if netifaces.AF_INET6 in addresses:
            for addr_info in addresses[netifaces.AF_INET6]:
                ip = addr_info.get("addr")
                if ip and not ip.startswith("fe80::") and ip != "::1":
                    return ip.split("%")[0]
    except (ValueError, OSError) as e:
        logger.warning(f"{e} Can not get local ip from NIC. Please verify whether SGLANG_LOCAL_IP_NIC is set correctly.")
    return None


def get_local_ip_by_remote() -> str | None:
    # Google's public DNS servers, used to discover the local IP.
    # UDP connect doesn't send packets; it just selects the right source address.
    # https://developers.google.com/speed/public-dns/docs/using#addresses
    # Try IPv4 first, then IPv6. getaddrinfo on a literal IP returns exactly
    # one result, so we unpack directly instead of looping.
    for dns_host, dns_port in [("8.8.8.8", 80), ("2001:4860:4860::8888", 80)]:
        try:
            family, socktype, proto, _, sockaddr = socket.getaddrinfo(
                dns_host,
                dns_port,
                socket.AF_UNSPEC,
                socket.SOCK_DGRAM,
                0,
                socket.AI_ADDRCONFIG,
            )[0]
            with socket.socket(family, socktype, proto) as s:
                s.connect(sockaddr)
                return s.getsockname()[0]
        except (socket.gaierror, OSError):
            continue

    # Fallback: resolve the local hostname to an IP address via /etc/hosts or DNS.
    # Unreliable — many machines resolve hostname to 127.0.0.1, so we skip loopback.
    try:
        hostname = socket.gethostname()
        ip = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)[0][4][0]
        if ip and ip not in ("127.0.0.1", "0.0.0.0", "::1"):  # noqa: S104
            return ip
    except Exception as e:
        logger.debug(f"Failed to get local ip by resolving hostname: {e}")

    logger.warning("Can not get local ip by remote")
    return None


def get_local_ip_auto(fallback: str = None) -> str:
    """
    Automatically detect the local IP address using multiple fallback strategies.

    This function attempts to obtain the local IP address through several methods.
    If all methods fail, it returns the specified fallback value or raises an exception.

    Args:
        fallback (str, optional): Fallback IP address to return if all detection
            methods fail. For server applications, explicitly set this to
            "0.0.0.0" (IPv4) or "::" (IPv6) to bind to all available interfaces.
            Defaults to None.

    Returns:
        str: The detected local IP address, or the fallback value if detection fails.

    Raises:
        ValueError: If IP detection fails and no fallback value is provided.

    Note:
        The function tries detection methods in the following order:
        1. Environment variable SGLANG_HOST_IP / HOST_IP
        2. Network interface enumeration via get_local_ip_by_nic()
        3. Remote connection method via get_local_ip_by_remote()
    """
    # Try environment variable
    host_ip = os.getenv("SGLANG_HOST_IP", "") or os.getenv("HOST_IP", "")
    if host_ip:
        return host_ip
    logger.debug("get_ip failed")
    # Fallback
    if ip := get_local_ip_by_nic():
        return ip
    logger.debug("get_local_ip_by_nic failed")
    # Fallback
    if ip := get_local_ip_by_remote():
        return ip
    logger.debug("get_local_ip_by_remote failed")
    if fallback:
        return fallback
    raise ValueError("Can not get local ip")


# ---------------------------------------------------------------------------
# NetworkAddress — immutable (host, port) value type with helpers
# ---------------------------------------------------------------------------


def _is_ipv6(host: str) -> bool:
    """Check whether *host* is a valid IPv6 address (without brackets)."""
    try:
        ipaddress.IPv6Address(host)
        return True
    except ValueError:
        return False


def _wrap(host: str) -> str:
    """Wrap an IPv6 address in brackets; pass IPv4/hostname through."""
    return f"[{host}]" if _is_ipv6(host) else host


def _parse_port(s: str) -> int:
    try:
        port = int(s)
    except ValueError as e:
        raise ValueError(f"Invalid port number: {s!r}") from e
    if not (0 <= port <= 65535):
        raise ValueError(f"Port out of range (0-65535): {port}")
    return port


@dataclass(frozen=True)
class NetworkAddress:
    host: str
    port: int

    def __post_init__(self):
        # Auto-strip IPv6 brackets so callers can pass "[::1]" or "::1"
        if self.host.startswith("[") and self.host.endswith("]"):
            object.__setattr__(self, "host", self.host[1:-1])

    @property
    def is_ipv6(self) -> bool:
        return _is_ipv6(self.host)

    @property
    def family(self) -> socket.AddressFamily:
        return socket.AF_INET6 if self.is_ipv6 else socket.AF_INET

    def to_url(self, scheme: str = "http") -> str:
        """``http://127.0.0.1:30000`` or ``http://[::1]:30000``."""
        return f"{scheme}://{_wrap(self.host)}:{self.port}"

    def to_tcp(self) -> str:
        """``tcp://`` endpoint for ZMQ / torch distributed."""
        return self.to_url("tcp")

    def to_host_port_str(self) -> str:
        """``host:port`` string for gRPC listen address, session IDs, logs."""
        return f"{_wrap(self.host)}:{self.port}"

    @staticmethod
    def resolve_host(host: str) -> str:
        """Return *host* as-is if it's an IP, otherwise DNS-resolve to one."""
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        try:
            return socket.getaddrinfo(host, None, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)[0][4][0]
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve host {host!r}: {e}") from e

    def resolved(self) -> NetworkAddress:
        """DNS-resolve hostname to IP; return self if already an IP."""
        ip = self.resolve_host(self.host)
        return self if ip == self.host else NetworkAddress(ip, self.port)

    def to_bind_tuple(self) -> tuple[str, int]:
        """Raw ``(host, port)`` tuple for ``socket.bind()`` / ``socket.connect()``.

        Returns the *unwrapped* host — sockets need the raw address, not
        the bracketed form.
        """
        return (self.host, self.port)

    @staticmethod
    def parse(addr: str) -> NetworkAddress:
        """Parse a ``host:port`` string into a ``NetworkAddress``.

        Accepted formats::

            [::1]:8000          → NetworkAddress("::1", 8000)
            127.0.0.1:8000      → NetworkAddress("127.0.0.1", 8000)
            my-hostname:8000    → NetworkAddress("my-hostname", 8000)

        IPv6 addresses **must** be bracketed.  Bare ``::1:8000`` is
        ambiguous and will raise ``ValueError``.

        Raises:
            ValueError: If the string cannot be unambiguously parsed.
        """
        if not addr:
            raise ValueError("Empty address string")

        # --- Bracketed IPv6: [addr]:port ---
        if addr.startswith("["):
            close = addr.find("]")
            if close == -1:
                raise ValueError(f"Missing closing bracket in IPv6 address: {addr!r}")
            host = addr[1:close]
            if not _is_ipv6(host):
                raise ValueError(f"Invalid IPv6 address inside brackets: {host!r}")
            rest = addr[close + 1 :]
            if not rest.startswith(":") or len(rest) < 2:
                raise ValueError(f"Expected ':port' after closing bracket, got: {rest!r}")
            return NetworkAddress(host, _parse_port(rest[1:]))

        # --- Plain host:port (IPv4 / hostname) ---
        if ":" not in addr:
            raise ValueError(f"Missing port in address (expected host:port): {addr!r}")
        host, port_str = addr.rsplit(":", 1)
        if not host:
            raise ValueError(f"Empty host in address: {addr!r}")
        # Guard against bare IPv6 slipping through
        if ":" in host and _is_ipv6(host):
            raise ValueError(f"Bare IPv6 address without brackets is ambiguous: {addr!r}. Use [{host}]:{port_str} instead.")
        return NetworkAddress(host, _parse_port(port_str))

    def __str__(self) -> str:
        return self.to_host_port_str()

    def __repr__(self) -> str:
        return f"NetworkAddress({self.host!r}, {self.port})"
