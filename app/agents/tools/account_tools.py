"""
Account tools — used by AccountAgent.

These tools return mock data for the take-home.
The integration (ADK tool wiring + agent invocation) is what's graded.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: str
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_BUILDS = [
    BuildSummary(
        build_id="b_001",
        pipeline="ci/main",
        status="passed",
        branch="main",
        started_at=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
        duration_seconds=142,
    ),
    BuildSummary(
        build_id="b_002",
        pipeline="ci/main",
        status="failed",
        branch="feature/auth",
        started_at=(datetime.utcnow() - timedelta(hours=3)).isoformat(),
        duration_seconds=87,
    ),
    BuildSummary(
        build_id="b_003",
        pipeline="ci/deploy",
        status="passed",
        branch="main",
        started_at=(datetime.utcnow() - timedelta(hours=6)).isoformat(),
        duration_seconds=210,
    ),
    BuildSummary(
        build_id="b_004",
        pipeline="ci/main",
        status="failed",
        branch="fix/logging",
        started_at=(datetime.utcnow() - timedelta(hours=12)).isoformat(),
        duration_seconds=55,
    ),
    BuildSummary(
        build_id="b_005",
        pipeline="ci/staging",
        status="cancelled",
        branch="experiment/new-ui",
        started_at=(datetime.utcnow() - timedelta(days=1)).isoformat(),
        duration_seconds=30,
    ),
]


async def get_recent_builds(user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the most recent builds for a user, newest first.

    Args:
        user_id: the user whose builds to fetch
        limit: maximum number of builds to return (default 5)

    Returns:
        List of build dicts with build_id, pipeline, status, branch,
        started_at, and duration_seconds.
    """
    builds = _MOCK_BUILDS[:limit]
    return [asdict(b) for b in builds]


async def get_account_status(user_id: str) -> dict:
    """
    Return current account status including plan tier and usage limits.

    Args:
        user_id: the user whose account status to fetch

    Returns:
        Dict with user_id, plan_tier, concurrent_builds_used/limit,
        and storage_used_gb/limit_gb.
    """
    status = AccountStatus(
        user_id=user_id,
        plan_tier="pro",
        concurrent_builds_used=2,
        concurrent_builds_limit=10,
        storage_used_gb=4.2,
        storage_limit_gb=50.0,
    )
    return asdict(status)
