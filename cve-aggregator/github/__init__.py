"""GitHub API integration module."""

from .github_advisory_api import (
    GitHubAdvisoryClient,
    SecurityAdvisory,
)

__all__ = [
    "GitHubAdvisoryClient",
    "SecurityAdvisory",
]
