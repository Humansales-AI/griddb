import type { QueryResult } from './FivebitClient';

export class QueryClient {
  private baseUrl: string;
  private getToken: () => string;
  private etagCache: Map<number, string> = new Map();

  constructor(baseUrl: string, getToken: () => string) {
    this.baseUrl = baseUrl; this.getToken = getToken;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    const t = this.getToken();
    if (t) h['Authorization'] = `Bearer ${t}`;
    return h;
  }

  table<T = any>(name: string) {
    const self = this;
    return {
      async getById(id: number): Promise<QueryResult<T>> {
        const h = self.headers();
        const etag = self.etagCache.get(id);
        if (etag) h['If-None-Match'] = `"${etag}"`;
        const r = await fetch(`${self.baseUrl}/records/${id}`, { headers: h });
        if (r.status === 304) return { data: null, error: null, etag: etag || '' };
        const data = await r.json();
        const newEtag = r.headers.get('ETag')?.replace(/"/g, '') || '';
        if (newEtag) self.etagCache.set(id, newEtag);
        return { data, error: null, etag: newEtag };
      },

      async insert(record: Omit<T, '_id' | '_hash'>): Promise<QueryResult<T>> {
        const r = await fetch(`${self.baseUrl}/records`, {
          method: 'POST', headers: self.headers(), body: JSON.stringify(record),
        });
        const data = await r.json();
        return { data, error: r.ok ? null : data.error, etag: '' };
      },

      async update(id: number, record: Partial<T>): Promise<QueryResult<T>> {
        const h = self.headers();
        const etag = self.etagCache.get(id);
        if (etag) h['If-Match'] = `"${etag}"`;
        const r = await fetch(`${self.baseUrl}/records/${id}`, {
          method: 'PUT', headers: h, body: JSON.stringify(record),
        });
        if (r.status === 412) return { data: null, error: 'Conflict — record modified', etag: '' };
        const data = await r.json();
        return { data, error: null, etag: '' };
      },

      async delete(id: number): Promise<boolean> {
        const r = await fetch(`${self.baseUrl}/records/${id}`, {
          method: 'DELETE', headers: self.headers(),
        });
        self.etagCache.delete(id);
        return r.ok;
      },

      async query(field: string, value: string, limit = 20): Promise<T[]> {
        const r = await fetch(`${self.baseUrl}/records?field=${field}&value=${value}&limit=${limit}`, {
          headers: self.headers(),
        });
        const data = await r.json();
        return data.results || [];
      },
    };
  }
}
