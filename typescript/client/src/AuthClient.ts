export interface AuthSession { userId: number; token: string; expiresAt: number; }

export class AuthClient {
  private baseUrl: string;
  private _token: string | null = null;

  constructor(baseUrl: string) { this.baseUrl = baseUrl; }

  async signUp(email: string, password: string, name?: string): Promise<{ userId: number } | { error: string }> {
    const r = await fetch(`${this.baseUrl}/api/auth/signup`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name }),
    });
    return r.json();
  }

  async signIn(email: string, password: string): Promise<{ session: AuthSession } | { error: string }> {
    const r = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await r.json();
    if ('session' in data) this._token = data.session.token;
    return data;
  }

  async signOut(): Promise<void> {
    if (this._token) {
      await fetch(`${this.baseUrl}/api/auth/logout`, {
        method: 'POST', headers: { 'Authorization': `Bearer ${this._token}` },
      });
      this._token = null;
    }
  }

  get token(): string | null { return this._token; }
}
