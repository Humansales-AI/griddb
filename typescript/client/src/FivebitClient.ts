import { AuthClient } from './AuthClient';
import { QueryClient } from './QueryClient';
import { RealtimeClient } from './RealtimeClient';
import { StorageClient } from './StorageClient';

export interface FivebitConfig { url: string; apiKey?: string; }
export interface AuthSession { userId: number; token: string; expiresAt: number; }
export interface QueryResult<T> { data: T | null; error: string | null; etag: string; }

export class FivebitClient {
  config: FivebitConfig;
  auth: AuthClient;
  storage: StorageClient;
  realtime: RealtimeClient;
  private _query: QueryClient;
  private _session: AuthSession | null = null;

  constructor(config: FivebitConfig) {
    this.config = config;
    this.auth = new AuthClient(config.url);
    const getToken = () => this._session?.token || '';
    this._query = new QueryClient(config.url, getToken);
    this.storage = new StorageClient(config.url, getToken);
    this.realtime = new RealtimeClient(config.url);
  }

  get session(): AuthSession | null { return this._session; }
  setSession(s: AuthSession | null) { this._session = s; }

  from<T = any>(table: string) { return this._query.table<T>(table); }
  onChanges(table: string, fn: (record: any) => void) { return this.realtime.subscribe(table, fn); }
  channel(name: string) { return this.realtime.channel(name); }
  close() { this.realtime.close(); }
}

export function createClient(config: FivebitConfig): FivebitClient {
  return new FivebitClient(config);
}
