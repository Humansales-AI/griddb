type ChangeHandler = (record: any) => void;

export class RealtimeClient {
  private baseUrl: string;
  private sources: Map<string, EventSource> = new Map();
  private handlers: Map<string, Set<ChangeHandler>> = new Map();
  private reconnectMs = 3000;

  constructor(baseUrl: string) { this.baseUrl = baseUrl; }

  subscribe(table: string, fn: ChangeHandler): () => void {
    if (!this.handlers.has(table)) {
      this.handlers.set(table, new Set());
      this._connect(table);
    }
    this.handlers.get(table)!.add(fn);
    return () => this.handlers.get(table)?.delete(fn);
  }

  private _connect(table: string): void {
    const url = `${this.baseUrl}/stream/${table}`;
    const es = new EventSource(url);
    es.onmessage = (e) => {
      try {
        const record = JSON.parse(e.data);
        this.handlers.get(table)?.forEach(fn => fn(record));
      } catch {}
    };
    es.onerror = () => {
      es.close();
      setTimeout(() => this._connect(table), this.reconnectMs);
    };
    this.sources.set(table, es);
  }

  close(): void {
    for (const es of this.sources.values()) es.close();
    this.sources.clear();
    this.handlers.clear();
  }
}
