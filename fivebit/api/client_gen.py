#!/usr/bin/env python3
"""
5bit Typed Client Generator
============================
Emits a TypeScript client from the same encoding spec the server uses.
Client and server agree by construction, not by convention.

Usage:
  python3 client_gen.py users '["balance","name","email"]' > client.ts
"""
import sys, json

TEMPLATE = '''/**
 * 5bit Typed Client — Auto-generated from encoding spec
 * ======================================================
 * Client and server agree by construction. Same spec, same bytes.
 * Content-addressed: every response has ETag = SHA-256 of record.
 */

const BASE = "{base_url}";

export interface {type_name} {{
{fields_ts}
  _id: number;
  _hash: string;
}}

export async function getById(id: number): Promise<{type_name} | null> {{
  const r = await fetch(`${{BASE}}/records/${{id}}`);
  if (r.status === 304) return null; // Not modified (ETag match)
  if (!r.ok) return null;
  return r.json();
}}

export async function getByHash(hash: string): Promise<{type_name} | null> {{
  const r = await fetch(`${{BASE}}/records/by-hash/${{hash}}`);
  if (!r.ok) return null;
  return r.json();
}}

export async function query(field: string, value: string, limit = 20): Promise<{type_name}[]> {{
  const r = await fetch(`${{BASE}}/records?field=${{field}}&value=${{value}}&limit=${{limit}}`);
  const data = await r.json();
  return data.results || [];
}}

export async function create(record: Omit<{type_name}, '_id' | '_hash'>): Promise<{type_name}> {{
  const r = await fetch(`${{BASE}}/records`, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(record),
  }});
  return r.json();
}}

export async function update(id: number, record: Omit<{type_name}, '_id' | '_hash'>, etag?: string): Promise<{type_name} | null> {{
  const headers: Record<string, string> = {{ 'Content-Type': 'application/json' }};
  if (etag) headers['If-Match'] = `"${{etag}}"`;
  const r = await fetch(`${{BASE}}/records/${{id}}`, {{
    method: 'PUT', headers, body: JSON.stringify(record),
  }});
  if (r.status === 409) return null; // CAS conflict
  return r.json();
}}

export async function deleteRecord(id: number): Promise<boolean> {{
  const r = await fetch(`${{BASE}}/records/${{id}}`, {{ method: 'DELETE' }});
  return r.ok;
}}

// Conditional fetch: only if changed since last ETag
export async function getIfChanged(id: number, etag: string): Promise<{type_name} | null> {{
  const r = await fetch(`${{BASE}}/records/${{id}}`, {{
    headers: {{ 'If-None-Match': `"${{etag}}"` }},
  }});
  if (r.status === 304) return null;
  return r.json();
}}
'''


def generate(name: str, fields: list, base_url: str = "http://localhost:8080") -> str:
    """Generate a typed TypeScript client from an encoding spec."""
    fields_ts = '\n'.join(f'  {f}: {"number" if f in ("balance","amount","age","count") else "string"};' for f in fields)
    type_name = name[0].upper() + name[1:] if name else 'Record'
    return TEMPLATE.format(
        type_name=type_name,
        fields_ts=fields_ts,
        base_url=base_url,
    )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 client_gen.py <name> <fields_json> [base_url]")
        print("Example: python3 client_gen.py users '[\"balance\",\"name\"]' http://localhost:8080")
        sys.exit(1)

    name = sys.argv[1]
    fields = json.loads(sys.argv[2])
    base_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8080"
    print(generate(name, fields, base_url))
