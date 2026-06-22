#!/usr/bin/env python3
"""
GridDB Crypto Wallet Store — Multi-Chain HD Wallet on GridDB
===============================================================
Storage layer for BIP44 HD wallets across BTC, ETH, SOL.

Record layout (AllocGrid, O(1) reads):
  recordId = user_base + offset
    user_base = userId * 1000
    offset 0-9:   chain configs (BTC=0, ETH=1, SOL=2)
    offset 10-99: derived addresses (BIP44 index 0-89)
    offset 100+:  transactions

Usage:
  store = CryptoWalletStore("./wallet_data")
  store.init_wallet(userId=1, mnemonic="abandon ...", passphrase="")
  addr = store.get_address(userId=1, coin='ETH', index=5)
  store.record_deposit(userId=1, coin='ETH', tx_hash='0x...', amount_wei=10**18)
  balance = store.get_balance(userId=1, coin='ETH')
"""
import os, sys, hashlib, struct
from typing import Optional, List, Tuple, Dict

# Add parent griddb/python to path
_GRIDDB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'python')
sys.path.insert(0, _GRIDDB_PATH)
from binary_grid_db import Token, Encoder, Parser, ParsedNumber, ParsedWord
from griddb_alloc import AllocGrid

# Try importing HD wallet; fall back gracefully
# Also add crypto/python to path for hd_wallet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hd_wallet import HDKey, mnemonic_to_seed, derive_address, COIN_TYPES
except ImportError:
    HDKey = None

# ── Store ───────────────────────────────────────────────────────────────

class CryptoWalletStore:
    """Multi-chain HD wallet backed by GridDB AllocGrid.

    One master seed → infinite addresses per user per chain.
    All lookups are O(1) via positioned record IDs.
    """

    def __init__(self, data_dir: str = "./crypto_data"):
        self.grid = AllocGrid(data_dir=data_dir)
        # In-memory index: userId → master HD key
        self._keys: Dict[int, object] = {}

    # ── Wallet Initialization ──────────────────────────────────────────

    def init_wallet(self, userId: int, mnemonic: str, passphrase: str = '') -> bool:
        """Initialize a user's HD wallet from a BIP39 mnemonic."""
        if HDKey is None:
            raise ImportError("hd_wallet module not available")

        seed = mnemonic_to_seed(mnemonic, passphrase)
        master = HDKey.from_seed(seed)
        self._keys[userId] = master

        # Store master fingerprint as a record
        base = userId * 1000
        fp_tokens = [*Encoder.encode_word('MASTER'),
                     *Encoder.encode_integer(userId),
                     Token.RECORD]
        self.grid.write(base, fp_tokens)
        return True

    def init_wallet_from_seed(self, userId: int, seed_hex: str) -> bool:
        """Initialize from a raw hex seed."""
        if HDKey is None:
            raise ImportError("hd_wallet module not available")
        seed = bytes.fromhex(seed_hex)
        master = HDKey.from_seed(seed)
        self._keys[userId] = master
        return True

    # ── Address Derivation ─────────────────────────────────────────────

    def get_address(self, userId: int, coin: str, index: int = 0,
                     change: int = 0, account: int = 0) -> str:
        """Derive and cache a BIP44 address. O(1) after first derivation."""
        base = userId * 1000
        coin_offset = {'BTC': 0, 'ETH': 1, 'SOL': 2}.get(coin.upper(), 0)
        record_id = base + 10 + coin_offset * 100 + index

        # Check cache
        existing = self.grid.read(record_id)
        if existing and not existing.is_tombstone:
            words = [p.text for p in existing.parsed if isinstance(p, ParsedWord)]
            return ''.join(words)

        # Derive and cache
        if userId not in self._keys:
            raise ValueError(f"Wallet not initialized for user {userId}")

        master = self._keys[userId]
        key, addr = derive_address(master, coin, account, change, index)

        # Cache: WORD(address) NUM(index) WORD(coin) RECORD
        tokens = [*Encoder.encode_word(addr),
                  *Encoder.encode_integer(index),
                  *Encoder.encode_word(coin),
                  Token.RECORD]
        self.grid.write(record_id, tokens)

        return addr

    def get_addresses(self, userId: int, coin: str, count: int = 10) -> List[str]:
        """Get multiple derived addresses."""
        return [self.get_address(userId, coin, i) for i in range(count)]

    # ── Transactions ───────────────────────────────────────────────────

    def record_deposit(self, userId: int, coin: str, tx_hash: str,
                        amount_satoshis: int, address_index: int = 0):
        """Record a deposit transaction."""
        base = userId * 1000
        tx_id = self._next_tx_id(userId)
        record_id = base + 100 + tx_id

        tokens = [*Encoder.encode_word('DEPOSIT'),
                  *Encoder.encode_word(coin),
                  *Encoder.encode_word(tx_hash),
                  *Encoder.encode_integer(amount_satoshis),
                  *Encoder.encode_integer(address_index),
                  *Encoder.encode_integer(userId),
                  Token.RECORD]
        self.grid.write(record_id, tokens)

    def record_withdrawal(self, userId: int, coin: str, tx_hash: str,
                           amount_satoshis: int, address_index: int = 0):
        """Record a withdrawal transaction."""
        base = userId * 1000
        tx_id = self._next_tx_id(userId)
        record_id = base + 100 + tx_id

        tokens = [*Encoder.encode_word('WITHDRAW'),
                  *Encoder.encode_word(coin),
                  *Encoder.encode_word(tx_hash),
                  *Encoder.encode_integer(amount_satoshis),
                  *Encoder.encode_integer(address_index),
                  *Encoder.encode_integer(userId),
                  Token.RECORD]
        self.grid.write(record_id, tokens)

    def get_balance(self, userId: int, coin: str) -> int:
        """Compute balance by replaying all transactions (event sourcing)."""
        base = userId * 1000
        balance = 0
        tx_id = 0
        while True:
            rec = self.grid.read(base + 100 + tx_id)
            if not rec or rec.is_tombstone:
                break
            words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
            nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
            if len(words) >= 3 and len(nums) >= 3:
                tx_type, tx_coin, tx_hash = words[0], words[1], words[2]
                amount = nums[0]
                if tx_coin.upper() == coin.upper():
                    if tx_type == 'DEPOSIT':
                        balance += amount
                    elif tx_type == 'WITHDRAW':
                        balance -= amount
            tx_id += 1
        return balance

    def get_transactions(self, userId: int, limit: int = 50) -> List[dict]:
        """Get recent transactions for a user."""
        base = userId * 1000
        txs = []
        tx_id = 0
        while len(txs) < limit:
            rec = self.grid.read(base + 100 + tx_id)
            if not rec:
                break
            if not rec.is_tombstone:
                words = [p.text for p in rec.parsed if isinstance(p, ParsedWord)]
                nums = [p.value for p in rec.parsed if isinstance(p, ParsedNumber)]
                if len(words) >= 3 and len(nums) >= 3:
                    txs.append({
                        'type': words[0],
                        'coin': words[1],
                        'tx_hash': words[2],
                        'amount': nums[0],
                        'address_index': nums[1] if len(nums) > 1 else 0,
                    })
            tx_id += 1
        return txs

    def _next_tx_id(self, userId: int) -> int:
        """Find the next available transaction ID for a user."""
        base = userId * 1000
        tx_id = 0
        while self.grid.read(base + 100 + tx_id):
            tx_id += 1
        return tx_id

    def close(self):
        self.grid.close()


# ── Demo ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, shutil

    d = tempfile.mkdtemp(prefix='griddb_crypto_')
    print(f"Demo dir: {d}")
    print("═" * 50)
    print("  GridDB Crypto Wallet Store Demo")
    print("═" * 50)

    try:
        store = CryptoWalletStore(data_dir=d)

        # Since we may not have the HD wallet module, demo the storage layer
        if HDKey is None:
            print("\n  HD wallet module not available — demoing storage layer only.")
            print("  Install pycryptodome + hdwallet for full BIP32/BIP44 support.")

        # Simulate wallet operations with raw record writes
        print("\n── Simulated wallet operations ──")

        # Init wallet for user 1
        store.grid.write(1000, [*Encoder.encode_word('MASTER'),
                                 *Encoder.encode_integer(1),
                                 Token.RECORD])
        print("  Wallet initialized for user #1")

        # Derive and cache ETH address
        addr_eth = '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1'
        store.grid.write(1011, [*Encoder.encode_word(addr_eth),
                                 *Encoder.encode_integer(0),
                                 *Encoder.encode_word('ETH'),
                                 Token.RECORD])
        print(f"  ETH address #0: {addr_eth}")

        # Derive BTC address
        addr_btc = '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'
        store.grid.write(1010, [*Encoder.encode_word(addr_btc),
                                 *Encoder.encode_integer(0),
                                 *Encoder.encode_word('BTC'),
                                 Token.RECORD])
        print(f"  BTC address #0: {addr_btc}")

        # Record a deposit
        store.record_deposit(1, 'ETH', '0xabc123...', 10**18, 0)
        store.record_deposit(1, 'BTC', 'btctx001...', 50000000, 0)
        print("  Deposits recorded: 1 ETH + 0.5 BTC")

        # Check balances
        eth_bal = store.get_balance(1, 'ETH')
        btc_bal = store.get_balance(1, 'BTC')
        print(f"  ETH balance: {eth_bal} wei ({eth_bal / 1e18} ETH)")
        print(f"  BTC balance: {btc_bal} satoshis ({btc_bal / 1e8} BTC)")

        # List transactions
        txs = store.get_transactions(1)
        print(f"  Transactions: {len(txs)}")
        for tx in txs:
            print(f"    {tx['type']} {tx['amount']} {tx['coin']} ({tx['tx_hash'][:20]}...)")

        store.close()
        print(f"\n  GridDB crypto store: {d}")
        print("═" * 50)

    finally:
        shutil.rmtree(d, ignore_errors=True)
