"""Resource URI handling and in-place transformation logic"""

import os
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ResourceReference:
    """Reference to a resource stored in Orbit Station"""

    def __init__(self, tool_call_id: str, uri: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize resource reference

        Args:
            tool_call_id: Unique identifier in Station
            uri: Resource URI (file path, S3 URL, etc.)
            metadata: Optional metadata about the resource
        """
        self.tool_call_id = tool_call_id
        self.uri = uri
        self.metadata = metadata or {}
        logger.debug("Created ResourceReference for tool_call_id: %s, uri: %s", tool_call_id, uri)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "tool_call_id": self.tool_call_id,
            "uri": self.uri,
            "metadata": self.metadata,
        }


class ResourceManager:
    """Manages resource URIs and in-place transformation logic"""

    def __init__(self) -> None:
        """Initialize resource manager"""
        self._uri_cache: Dict[str, str] = {}
        logger.info("ResourceManager initialized")

    def resolve_uri(self, uri: str) -> str:
        """
        Resolve a resource URI to a loadable path

        Supports:
        - Local file paths: /path/to/file.csv
        - S3 URIs: s3://bucket/key
        - HTTP URLs: https://example.com/file.csv

        Args:
            uri: Resource URI

        Returns:
            Resolved path or URL

        Raises:
            ValueError: If URI scheme is unsupported
        """
        parsed = urlparse(uri)

        if parsed.scheme in ("", "file"):
            # Local file
            path = parsed.path if parsed.scheme == "file" else uri
            if not os.path.exists(path):
                logger.warning("Local file does not exist: %s", path)
            return path

        if parsed.scheme == "s3":
            # S3 URL (would need boto3 in real implementation)
            logger.debug("S3 URI resolved: %s", uri)
            return uri

        if parsed.scheme in ("http", "https"):
            # HTTP(S) URL
            logger.debug("HTTP(S) URI resolved: %s", uri)
            return uri

        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")

    def generate_output_uri(
        self, original_uri: str, transform_name: str, in_place: bool = False
    ) -> str:
        """
        Generate output URI for a transformation result

        Args:
            original_uri: Original resource URI
            transform_name: Name of the transformation
            in_place: If True, return original URI. If False, generate new URI.

        Returns:
            Output URI
        """
        if in_place:
            logger.debug("In-place transformation, reusing URI: %s", original_uri)
            return original_uri

        parsed = urlparse(original_uri)

        if parsed.scheme in ("", "file"):
            # Local file: append transform name
            path = parsed.path if parsed.scheme == "file" else original_uri
            base, ext = os.path.splitext(path)
            new_path = f"{base}.{transform_name}{ext}"
            logger.debug("Generated output URI: %s", new_path)
            return new_path

        if parsed.scheme == "s3":
            # S3: append to key
            key = parsed.path.lstrip("/")
            base, ext = os.path.splitext(key)
            new_key = f"{base}.{transform_name}{ext}"
            new_uri = f"s3://{parsed.netloc}/{new_key}"
            logger.debug("Generated S3 output URI: %s", new_uri)
            return new_uri

        # Fallback
        logger.debug("Appending transform name to URI: %s", original_uri)
        return f"{original_uri}.{transform_name}"

    def get_file_size(self, uri: str) -> int:
        """
        Get file size in bytes

        Args:
            uri: Resource URI

        Returns:
            File size in bytes, or -1 if unavailable

        Raises:
            FileNotFoundError: If local file does not exist
        """
        parsed = urlparse(uri)

        if parsed.scheme in ("", "file"):
            path = parsed.path if parsed.scheme == "file" else uri
            if os.path.exists(path):
                size = os.path.getsize(path)
                logger.debug("File size for %s: %d bytes", path, size)
                return size
            raise FileNotFoundError(f"File not found: {path}")

        if parsed.scheme == "s3":
            # Would use boto3 in real implementation
            logger.debug("S3 file size unavailable (would require boto3)")
            return -1

        if parsed.scheme in ("http", "https"):
            # Would use requests.head() in real implementation
            logger.debug("HTTP(S) file size unavailable (would require requests.head())")
            return -1

        return -1


class TransformContext:
    """Context for a transformation execution"""

    def __init__(
        self,
        tool_call_id: str,
        original_uri: str,
        transform_name: str,
        in_place: bool = False,
    ) -> None:
        """
        Initialize transformation context

        Args:
            tool_call_id: Station reference ID
            original_uri: Original resource URI
            transform_name: Name of the transformation
            in_place: Whether transformation is in-place
        """
        self.tool_call_id = tool_call_id
        self.original_uri = original_uri
        self.transform_name = transform_name
        self.in_place = in_place
        self.execution_id = str(uuid.uuid4())
        self.manager = ResourceManager()

        output_uri = self.manager.generate_output_uri(
            original_uri, transform_name, in_place
        )
        self.output_uri = output_uri
        self.new_tool_call_id = tool_call_id if in_place else f"{tool_call_id}_{transform_name}"

        logger.debug(
            "Created TransformContext: execution_id=%s, input=%s, output=%s, in_place=%s",
            self.execution_id,
            self.original_uri,
            self.output_uri,
            self.in_place,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "tool_call_id": self.tool_call_id,
            "original_uri": self.original_uri,
            "output_uri": self.output_uri,
            "new_tool_call_id": self.new_tool_call_id,
            "transform_name": self.transform_name,
            "in_place": self.in_place,
            "execution_id": self.execution_id,
        }
