import { AuthClient } from './AuthClient';
import { QueryClient } from './QueryClient';
import { RealtimeClient } from './RealtimeClient';

export interface FivebitConfig { url: string; apiKey?: string; }
export interface AuthSession { userId: number; token: string; expiresAt: number; }
export interface QueryResult<T> { data: T | null; error: string | null; etag: string; }

export class FivebitClient {
  config: FivebitConfig;
  auth: AuthClient;
  private _query: QueryClient;
  private _rt: RealtimeClient;
  private _session: AuthSession | null = null;

  constructor(config: FivebitConfig) {
    this.config = config;
    this.auth = new AuthClient(config.url);
    this._query = new QueryClient(config.url, () => this._session?.token || '');
    this._rt = new RealtimeClient(config.url);
  }

  get session(): AuthSession | null { return this._session; }

  setSession(s: AuthSession | null) { this._session = s; }

  from<T = any>(table: string) {
    return this._query.table<T>(table);
  }

  onChanges(table: string, fn: (record: any) => void) {
    this._rt.subscribe(table, fn);
  }

  close() { this._rt.close(); }
}

export function createClient(config: FivebitConfig): FivebitClient {
  return new FivebitClient(config);
}
