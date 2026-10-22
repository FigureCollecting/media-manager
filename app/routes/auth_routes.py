from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import auth
from ..config import Settings, get_settings
from ..db import get_db
from ..schemas import DevTokenRequest, DevTokenResponse, RefreshTokenRequest, RefreshTokenResponse

# Two routers so a deployment can mount one without the other (app.boot_guard):
# dev_router issues tokens on request and is mounted in a dev environment only;
# router (token refresh) is mounted there or where AUTH_ROUTES_GRANTED says so.
dev_router = APIRouter(prefix="/auth", tags=["auth"])
router = APIRouter(prefix="/auth", tags=["auth"])


@dev_router.post("/dev-token", response_model=DevTokenResponse)
def issue_dev_token(
    payload: DevTokenRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> DevTokenResponse:
    if not settings.allow_dev_tokens:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="dev token issuance disabled"
        )

    try:
        token = auth.issue_dev_token(
            db, settings, user_id=payload.user_id, tenant_id=payload.tenant_id, aud=payload.aud
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return DevTokenResponse(token=token)


@router.post("/refresh", response_model=RefreshTokenResponse)
def refresh_token(
    payload: RefreshTokenRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> RefreshTokenResponse:
    claims = auth.decode_token(settings, payload.refresh_token)
    if not claims or claims.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired refresh token"
        )

    subject = claims.get("sub", "")
    tenant_id = claims.get("ten")

    access = auth.create_token(settings, subject=subject, tenant_id=tenant_id)
    refresh = auth.create_refresh_token(settings, subject=subject, tenant_id=tenant_id)
    return RefreshTokenResponse(access_token=access, refresh_token=refresh)
