/**
 * GridDB Secondary Indexes — Hash + B-tree
 * ========================================
 * HashIndex: O(1) equality. hash(key) → bucket → chain → recordId.
 * BTreeIndex: O(log n) range. B-tree nodes stored in AllocGrid.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { Token, ParsedNumber, ParsedWord } from './types';
import { Encoder } from './encoder';
import { Parser } from './parser';
import { AllocGrid, AllocRecord } from './alloc';

// ═══════════════════════════════════════════════════════════════════════════════
// HashIndex
// ═══════════════════════════════════════════════════════════════════════════════

export class HashIndex {
  private grid: AllocGrid;
  private buckets: number;
  private name: string;
  private _stats = { puts: 0, gets: 0, deletes: 0 };

  constructor(name: string, dataDir: string, buckets: number = 100000) {
    this.name = name;
    this.buckets = buckets;
    this.grid = new AllocGrid(path.join(dataDir, `idx_hash_${name}`));
  }

  private _hash(key: string): number {
    const h = crypto.createHash('sha256').update(key).digest();
    return Number(h.readBigUInt64BE(0)) % this.buckets;
  }

  put(key: string, recordId: number): void {
    const bucket = this._hash(key);
    const existing = this.grid.read(bucket);
    const chainTokens: Token[] = existing && !existing.isTombstone ? [...existing.tokens] : [];
    chainTokens.push(...Encoder.encodeWord(key), ...Encoder.encodeInteger(recordId), Token.RECORD);
    this.grid.write(bucket, chainTokens);
    this._stats.puts++;
  }

  get(key: string): number | null {
    const bucket = this._hash(key);
    const existing = this.grid.read(bucket);
    this._stats.gets++;
    if (!existing || existing.isTombstone) return null;
    const words = existing.parsed.filter(p => p.type === 'word').map(p => (p as any).text);
    const nums = existing.parsed.filter(p => p.type === 'number').map(p => (p as ParsedNumber).value);
    // Chain: WORD(key1) NUM(id1) WORD(key2) NUM(id2) ... — consecutive words joined
    let wordIdx = 0, numIdx = 0;
    while (wordIdx < words.length && numIdx < nums.length) {
      let fullKey = '';
      while (wordIdx < words.length && (numIdx === wordIdx || words[wordIdx] !== '')) {
        fullKey += words[wordIdx]; wordIdx++;
      }
      if (fullKey === key) return nums[numIdx];
      wordIdx++; numIdx++;
    }
    return null;
  }

  delete(key: string): boolean {
    this._stats.deletes++;
    const bucket = this._hash(key);
    const existing = this.grid.read(bucket);
    if (!existing || existing.isTombstone) return false;
    const words = existing.parsed.filter(p => p.type === 'word').map(p => (p as any).text);
    const nums = existing.parsed.filter(p => p.type === 'number').map(p => (p as ParsedNumber).value);
    const newChain: Token[] = [];
    let wi = 0;
    for (let ni = 0; ni < nums.length; ni++) {
      let k = '';
      while (wi < words.length && words[wi] !== '') { k += words[wi]; wi++; }
      wi++;
      if (k !== key) newChain.push(...Encoder.encodeWord(k), ...Encoder.encodeInteger(nums[ni]), Token.RECORD);
    }
    if (newChain.length > 0) this.grid.write(bucket, newChain);
    else this.grid.delete(bucket);
    return true;
  }

  stats(): any { return { ...this.grid, buckets: this.buckets, ...this._stats }; }
  close(): void { this.grid.close(); }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BTreeIndex
// ═══════════════════════════════════════════════════════════════════════════════

interface BTreeNode {
  nodeId: number; isLeaf: boolean; keys: number[]; children: number[]; nextLeaf: number;
}

const MIN_DEGREE = 2;

export class BTreeIndex {
  private grid: AllocGrid;
  private name: string;
  private rootId = -1;
  private nextNodeId = 0;

  constructor(name: string, dataDir: string) {
    this.name = name;
    this.grid = new AllocGrid(path.join(dataDir, `idx_btree_${name}`));
    this._bootstrap();
  }

  private _bootstrap(): void {
    const root = this.grid.read(0);
    if (root && !root.isTombstone) {
      const node = this._parseNode(root);
      if (node) { this.rootId = 0; this.nextNodeId = this.grid.totalEntries; }
    }
  }

  private _encodeNode(n: BTreeNode): Token[] {
    const t: Token[] = [
      ...Encoder.encodeInteger(n.nodeId), ...Encoder.encodeInteger(n.isLeaf ? 1 : 0),
      ...Encoder.encodeInteger(n.keys.length), ...Encoder.encodeInteger(n.nextLeaf),
    ];
    for (let i = 0; i < n.keys.length; i++) {
      t.push(...Encoder.encodeInteger(n.keys[i]));
      t.push(...Encoder.encodeInteger(n.children[i] ?? -1));
    }
    if (n.children.length > n.keys.length) t.push(...Encoder.encodeInteger(n.children[n.children.length - 1]));
    t.push(Token.RECORD);
    return t;
  }

  private _parseNode(rec: AllocRecord): BTreeNode | null {
    const nums = rec.parsed.filter(p => p.type === 'number').map(p => (p as ParsedNumber).value);
    if (nums.length < 4) return null;
    const data = nums.slice(4);
    return {
      nodeId: nums[0], isLeaf: nums[1] === 1,
      keys: data.filter((_, i) => i % 2 === 0).slice(0, nums[2]),
      children: data.filter((_, i) => i % 2 === 1).concat(data.length % 2 === 1 ? [data[data.length - 1]] : []).slice(0, nums[2] + 1),
      nextLeaf: nums[3],
    };
  }

  private _writeNode(n: BTreeNode): void { this.grid.write(n.nodeId, this._encodeNode(n)); }
  private _readNode(id: number): BTreeNode | null { const r = this.grid.read(id); return r && !r.isTombstone ? this._parseNode(r) : null; }
  private _allocId(): number { return this.nextNodeId++; }

  put(key: number, recordId: number): void {
    if (this.rootId < 0) {
      const root: BTreeNode = { nodeId: this._allocId(), isLeaf: true, keys: [key], children: [recordId], nextLeaf: -1 };
      this._writeNode(root); this.rootId = root.nodeId; return;
    }
    const [leaf, ancestors] = this._findLeaf(key);
    if (!leaf) return;
    let idx = 0;
    for (let i = 0; i < leaf.keys.length; i++) { if (key >= leaf.keys[i]) idx = i + 1; }
    leaf.keys.splice(idx, 0, key); leaf.children.splice(idx, 0, recordId);
    this._writeNode(leaf);
    if (leaf.keys.length > MIN_DEGREE * 2) this._splitLeaf(leaf, ancestors);
  }

  get(key: number): number | null {
    if (this.rootId < 0) return null;
    let node = this._readNode(this.rootId);
    while (node) {
      if (node.isLeaf) {
        for (let i = 0; i < node.keys.length; i++) { if (node.keys[i] === key) return node.children[i]; }
        return null;
      }
      let ci = 0;
      for (let i = 0; i < node.keys.length; i++) { if (key >= node.keys[i]) ci = i + 1; }
      node = this._readNode(node.children[ci]);
    }
    return null;
  }

  rangeScan(start: number, end: number): number[] {
    const r: number[] = [];
    if (this.rootId < 0) return r;
    let node = this._readNode(this.rootId);
    while (node && !node.isLeaf) {
      let ci = 0;
      for (let i = 0; i < node.keys.length; i++) { if (start >= node.keys[i]) ci = i + 1; }
      node = this._readNode(node.children[ci]);
    }
    while (node) {
      for (let i = 0; i < node.keys.length; i++) {
        if (node.keys[i] >= start && node.keys[i] < end && node.children[i] >= 0) r.push(node.children[i]);
        if (node.keys[i] >= end) return r;
      }
      node = node.nextLeaf >= 0 ? this._readNode(node.nextLeaf) : null;
    }
    return r;
  }

  delete(key: number): boolean {
    const node = this._readNode(this.rootId);
    if (!node) return false;
    // Soft delete: mark record_id as -1
    let cur = node;
    while (cur) {
      if (cur.isLeaf) {
        for (let i = 0; i < cur.keys.length; i++) {
          if (cur.keys[i] === key) { cur.children[i] = -1; this._writeNode(cur); return true; }
        }
        return false;
      }
      let ci = 0;
      for (let i = 0; i < cur.keys.length; i++) { if (key >= cur.keys[i]) ci = i + 1; }
      const next = this._readNode(cur.children[ci]);
      if (!next) break;
      cur = next;
    }
    return false;
  }

  private _findLeaf(key: number): [BTreeNode | null, BTreeNode[]] {
    const ancestors: BTreeNode[] = [];
    let node = this._readNode(this.rootId);
    while (node && !node.isLeaf) {
      ancestors.push(node); let ci = 0;
      for (let i = 0; i < node.keys.length; i++) { if (key >= node.keys[i]) ci = i + 1; }
      node = this._readNode(node.children[ci]);
    }
    return [node, ancestors];
  }

  private _splitLeaf(leaf: BTreeNode, ancestors: BTreeNode[]): void {
    const mid = Math.floor(leaf.keys.length / 2);
    const newLeaf: BTreeNode = { nodeId: this._allocId(), isLeaf: true, keys: leaf.keys.slice(mid), children: leaf.children.slice(mid), nextLeaf: leaf.nextLeaf };
    this._writeNode(newLeaf);
    leaf.keys = leaf.keys.slice(0, mid); leaf.children = leaf.children.slice(0, mid); leaf.nextLeaf = newLeaf.nodeId;
    this._writeNode(leaf);
    const promote = newLeaf.keys[0];
    if (ancestors.length === 0) {
      const root: BTreeNode = { nodeId: this._allocId(), isLeaf: false, keys: [promote], children: [leaf.nodeId, newLeaf.nodeId], nextLeaf: -1 };
      this._writeNode(root); this.rootId = root.nodeId;
    } else {
      this._insertInternal(ancestors[ancestors.length - 1], promote, leaf.nodeId, newLeaf.nodeId, ancestors.slice(0, -1));
    }
  }

  private _insertInternal(parent: BTreeNode, key: number, left: number, right: number, ancestors: BTreeNode[]): void {
    let pos = 0;
    for (let i = 0; i < parent.children.length; i++) { if (parent.children[i] === left) { pos = i; break; } }
    parent.keys.splice(pos, 0, key); parent.children.splice(pos + 1, 0, right);
    this._writeNode(parent);
    if (parent.keys.length > MIN_DEGREE * 2) this._splitInternal(parent, ancestors);
  }

  private _splitInternal(node: BTreeNode, ancestors: BTreeNode[]): void {
    const mid = Math.floor(node.keys.length / 2);
    const pk = node.keys[mid];
    const nn: BTreeNode = { nodeId: this._allocId(), isLeaf: false, keys: node.keys.slice(mid + 1), children: node.children.slice(mid + 1), nextLeaf: -1 };
    this._writeNode(nn);
    node.keys = node.keys.slice(0, mid); node.children = node.children.slice(0, mid + 1);
    this._writeNode(node);
    if (ancestors.length === 0) {
      const root: BTreeNode = { nodeId: this._allocId(), isLeaf: false, keys: [pk], children: [node.nodeId, nn.nodeId], nextLeaf: -1 };
      this._writeNode(root); this.rootId = root.nodeId;
    } else {
      this._insertInternal(ancestors[ancestors.length - 1], pk, node.nodeId, nn.nodeId, ancestors.slice(0, -1));
    }
  }

  stats(): any { return { rootId: this.rootId, nextNodeId: this.nextNodeId }; }
  close(): void { this.grid.close(); }
}
