"""
5bit Auth Server — JWT sessions + OAuth
=========================================
Bridges @fivebit/client to MultiModeGrid.
Signup/login/logout via JWT. Zero-mode users: server never sees keys.

POST /api/auth/signup   { email, password, name? }  → { userId }
POST /api/auth/login    { email, password }          → { session }
POST /api/auth/logout   (Bearer token)               → 200
"""
import os, sys, json, time, hashlib, hmac, base64, secrets
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from binary_grid_db import Encoder, Token
from fivebit.auth.multimode import MultiModeGrid

JWT_SECRET = os.environ.get('FIVEBIT_JWT_SECRET', 'fivebit-dev-secret-change-me').encode()
SESSION_DURATION = 86400  # 24 hours

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

def _sign(payload: dict) -> str:
    header = _b64url(json.dumps({"alg":"HS256","typ":"JWT"}).encode())
    body = _b64url(json.dumps(payload).encode())
    sig = _b64url(hmac.new(JWT_SECRET, f"{header}.{body}".encode(), 'sha256').digest())
    return f"{header}.{body}.{sig}"

def _verify(token: str) -> Optional[dict]:
    try:
        parts = token.split('.')
        if len(parts) != 3: return None
        header_b64, body_b64, sig_b64 = parts
        expected = _b64url(hmac.new(JWT_SECRET, f"{header_b64}.{body_b64}".encode(), 'sha256').digest())
        if not hmac.compare_digest(sig_b64.encode(), expected.encode()):
            return None
        body = json.loads(base64.urlsafe_b64decode(body_b64 + '=='))
        if body.get('exp', 0) < time.time(): return None
        return body
    except: return None

def _bearer_token(headers: dict) -> Optional[str]:
    auth = headers.get('Authorization', '')
    if auth.startswith('Bearer '): return auth[7:]
    return None


class AuthHandler:
    """Mix-in for APIHandler — adds auth routes."""

    def handle_auth(self, method: str, path: str, headers: dict, body: bytes):
        grid = self.grid  # Must be MultiModeGrid
        parsed = urlparse(path)

        if method == 'POST' and parsed.path == '/api/auth/signup':
            return self._signup(body)
        if method == 'POST' and parsed.path == '/api/auth/login':
            return self._login(body)
        if method == 'POST' and parsed.path == '/api/auth/logout':
            return self._logout(headers)
        if method == 'POST' and parsed.path == '/api/auth/oauth':
            return self._oauth(body)

        return None  # Not an auth route

    def _signup(self, body: bytes):
        try:
            data = json.loads(body)
            email = data.get('email', '')
            password = data.get('password', '')
            name = data.get('name', '')
            mode = data.get('mode', 'zero')
            if not email or len(password) < 6:
                return 400, {'error': 'Email + password (6+ chars) required'}

            # Check duplicate
            if self._find_user(email) is not None:
                return 409, {'error': 'Email already registered'}

            uid = self._next_user_id()
            self.grid.signup(uid, password, mode)
            # Store email in grid so login can find it
            self.grid.base.write(80_100_000 + uid, [
                *Encoder.encode_integer(uid),
                *Encoder.encode_word(email),
                Token.RECORD,
            ])
            return 201, {'userId': uid, 'mode': mode}
        except Exception as e:
            return 500, {'error': str(e)}

    def _login(self, body: bytes):
        try:
            data = json.loads(body)
            email = data.get('email', '')
            password = data.get('password', '')
            # Find user by scanning mode records
            uid = self._find_user(email)
            if uid is None:
                return 401, {'error': 'Invalid credentials'}
            self.grid.login(uid, password)
            payload = {'sub': uid, 'iat': int(time.time()), 'exp': int(time.time() + SESSION_DURATION)}
            token = _sign(payload)
            # Invalidate old sessions
            self.grid.lock()
            return 200, {'session': {'token': token, 'userId': uid, 'expiresAt': payload['exp']}}
        except Exception as e:
            return 401, {'error': 'Invalid credentials'}

    def _oauth(self, body: bytes):
        from fivebit.api.oauth import verify_google_token, exchange_github_code, find_or_create_oauth_user
        try:
            data = json.loads(body)
            provider = data.get('provider', '')
            mode = data.get('mode', 'managed')

            if provider == 'google':
                id_token = data.get('idToken', '')
                if not id_token: return 400, {'error': 'idToken required'}
                profile = verify_google_token(id_token)
                if not profile: return 401, {'error': 'Invalid Google token'}
            elif provider == 'github':
                code = data.get('code', '')
                client_id = os.environ.get('GITHUB_CLIENT_ID', '')
                client_secret = os.environ.get('GITHUB_CLIENT_SECRET', '')
                if not code: return 400, {'error': 'code required'}
                profile = exchange_github_code(code, client_id, client_secret)
                if not profile: return 401, {'error': 'Invalid GitHub code'}
            else:
                return 400, {'error': f'Unknown provider: {provider}'}

            uid = find_or_create_oauth_user(self.grid, provider, profile['sub'],
                                             profile.get('email', ''), profile.get('name', ''), mode)
            payload = {'sub': uid, 'iat': int(time.time()), 'exp': int(time.time() + SESSION_DURATION)}
            token = _sign(payload)
            return 200, {'session': {'token': token, 'userId': uid, 'expiresAt': payload['exp']}}
        except Exception as e:
            return 401, {'error': str(e)}

    def _logout(self, headers: dict):
        token = _bearer_token(headers)
        if not token: return 401, {'error': 'No token'}
        # JWT is stateless — logout is client-side (discard token)
        return 200, {'ok': True}

    def _next_user_id(self) -> int:
        uid = 1
        while True:
            rec = self.grid.base.read(80_000_000 + uid)  # MODE_RECORD
            if not rec or rec.is_tombstone: return uid
            uid += 1

    def _find_user(self, email: str) -> Optional[int]:
        for uid in range(1, 10000):
            rec = self.grid.base.read(80_100_000 + uid)
            if not rec or rec.is_tombstone: continue
            words = [p.text for p in rec.parsed if hasattr(p, 'text')]
            stored_email = ''.join(words)
            if stored_email == email:
                return uid
        return None
