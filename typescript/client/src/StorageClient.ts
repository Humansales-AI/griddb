export class StorageClient {
  private baseUrl: string; private getToken: () => string;

  constructor(baseUrl: string, getToken: () => string) {
    this.baseUrl = baseUrl; this.getToken = getToken;
  }

  bucket(name: string) {
    const self = this;
    return {
      async upload(path: string, data: Blob | Buffer | Uint8Array): Promise<{ path: string; hash: string; size: number }> {
        const form = new FormData();
        const blob = data instanceof Blob ? data : new Blob([data as any]);
        form.append('file', blob, path);
        const r = await fetch(`${self.baseUrl}/storage/${name}/${path}`, {
          method: 'POST', headers: self._auth(), body: form,
        });
        return r.json();
      },
      async download(path: string): Promise<Blob | null> {
        const r = await fetch(`${self.baseUrl}/storage/${name}/${path}`, { headers: self._auth() });
        return r.ok ? r.blob() : null;
      },
      async list(opts?: { prefix?: string; limit?: number }): Promise<{ path: string; size: number; hash: string }[]> {
        const prefix = opts?.prefix || ''; const limit = opts?.limit || 100;
        const r = await fetch(`${self.baseUrl}/storage/${name}?prefix=${prefix}&limit=${limit}`, { headers: self._auth() });
        const d = await r.json(); return d.files || [];
      },
      async remove(path: string): Promise<boolean> {
        const r = await fetch(`${self.baseUrl}/storage/${name}/${path}`, { method: 'DELETE', headers: self._auth() });
        return r.ok;
      },
    };
  }

  private _auth(): Record<string, string> {
    const t = this.getToken(); return t ? { Authorization: `Bearer ${t}` } : {};
  }
}
