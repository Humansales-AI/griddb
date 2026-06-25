type ChangeHandler = (record: any) => void;
type MessageHandler = (payload: any) => void;

export class RealtimeClient {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private tableHandlers: Map<string, Set<ChangeHandler>> = new Map();
  private channelHandlers: Map<string, Set<MessageHandler>> = new Map();
  private presenceCache: any[] = [];
  private reconnectMs = 3000;
  private wsUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    this.wsUrl = baseUrl.replace('http', 'ws') + '/ws';
    this._connect();
  }

  private _connect(): void {
    if (typeof WebSocket === 'undefined') return;
    this.ws = new WebSocket(this.wsUrl);
    this.ws.onopen = () => {
      // Re-subscribe to tables
      for (const table of this.tableHandlers.keys()) {
        this.ws!.send(JSON.stringify({ type: 'subscribe', table }));
      }
    };
    this.ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'change') {
          this.tableHandlers.get(msg.table)?.forEach(fn => fn(msg.event));
        } else if (msg.type === 'broadcast') {
          this.channelHandlers.get(msg.channel)?.forEach(fn => fn(msg.payload));
        } else if (msg.type === 'presence') {
          this.presenceCache = msg.users || [];
          this.channelHandlers.get(`__presence:${msg.channel || ''}`)?.forEach(fn => fn(msg));
        }
      } catch {}
    };
    this.ws.onclose = () => setTimeout(() => this._connect(), this.reconnectMs);
  }

  // Table changes (backward compat + new)
  subscribe(table: string, fn: ChangeHandler): () => void {
    if (!this.tableHandlers.has(table)) this.tableHandlers.set(table, new Set());
    this.tableHandlers.get(table)!.add(fn);
    if (this.ws?.readyState === 1) {
      this.ws.send(JSON.stringify({ type: 'subscribe', table }));
    }
    return () => this.tableHandlers.get(table)?.delete(fn);
  }

  // Channels (new)
  channel(name: string) {
    const self = this;
    return {
      on(fn: MessageHandler) {
        if (!self.channelHandlers.has(name)) self.channelHandlers.set(name, new Set());
        self.channelHandlers.get(name)!.add(fn);
        if (self.ws?.readyState === 1) {
          self.ws.send(JSON.stringify({ type: 'channel', channel: name }));
        }
      },
      broadcast(payload: any) {
        if (self.ws?.readyState === 1) {
          self.ws.send(JSON.stringify({ type: 'broadcast', channel: name, payload }));
        }
      },
      presence(userId: number, userName: string) {
        if (self.ws?.readyState === 1) {
          self.ws.send(JSON.stringify({ type: 'presence', userId, name: userName, channel: name }));
        }
      },
      get users(): any[] { return self.presenceCache; },
    };
  }

  close(): void { this.ws?.close(); }
}
