"""SQLite 本地持仓；不保存任何券商、身份或密钥信息。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any


class PortfolioStore:
    """管理本地组合资料、持仓与模拟调仓历史。"""

    def __init__(self, path: str | Path = "data/user/kronos.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def get_portfolio(self, profile_id: str = "default") -> dict[str, Any]:
        """读取组合；首次访问时创建默认均衡档案。"""
        with closing(self._connect()) as conn:
            self._ensure_profile(conn, profile_id)
            profile = conn.execute("SELECT profile_id, name, risk_profile, cash, version FROM portfolio_profiles WHERE profile_id = ?", (profile_id,)).fetchone()
            holdings = conn.execute("SELECT stock_code, shares, cost_basis, updated_at FROM holdings WHERE profile_id = ? ORDER BY stock_code", (profile_id,)).fetchall()
        return {"id": profile[0], "name": profile[1], "risk_profile": profile[2], "cash": profile[3], "version": profile[4], "holdings": [{"stock_code": row[0], "shares": row[1], "cost_basis": row[2], "updated_at": row[3]} for row in holdings]}

    def replace_portfolio(self, payload: dict[str, Any], profile_id: str = "default") -> dict[str, Any]:
        """原子替换现金、风险档案和持仓，并校验股票数量规则。"""
        self._validate_payload(payload)
        with closing(self._connect()) as conn:
            self._ensure_profile(conn, profile_id)
            conn.execute("UPDATE portfolio_profiles SET name=?, risk_profile=?, cash=?, version=version+1 WHERE profile_id=?", (payload.get("name", "默认组合"), payload["risk_profile"], payload["cash"], profile_id))
            conn.execute("DELETE FROM holdings WHERE profile_id = ?", (profile_id,))
            conn.executemany("INSERT INTO holdings(profile_id, stock_code, shares, cost_basis, updated_at) VALUES (?, ?, ?, ?, datetime('now'))", [(profile_id, item["stock_code"], item["shares"], item["cost_basis"]) for item in payload.get("holdings", [])])
            conn.commit()
        return self.get_portfolio(profile_id)

    def save_rebalance(self, profile_id: str, target: dict[str, Any], orders: list[dict[str, Any]], fees: float) -> int:
        """保存模拟调仓，不会向任何外部交易系统发送请求。"""
        current = self.get_portfolio(profile_id)
        with closing(self._connect()) as conn:
            self._ensure_profile(conn, profile_id)
            cursor = conn.execute("INSERT INTO rebalance_runs(profile_id, input_json, target_json, orders_json, fees, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))", (profile_id, json.dumps(current, ensure_ascii=False), json.dumps(target, ensure_ascii=False), json.dumps(orders, ensure_ascii=False), fees))
            conn.commit()
            return int(cursor.lastrowid)

    def save_recommendation(self, profile_id: str, stock_code: str, action: str, horizons: dict[str, Any], evidence_version: str, generated_at: str | None = None, data_as_of: str | None = None, evidence_gate_passed: bool = False) -> int:
        """保存一条建议历史，供前瞻模拟与审计追溯。"""
        with closing(self._connect()) as conn:
            self._ensure_profile(conn, profile_id)
            cursor = conn.execute("INSERT INTO recommendation_history(profile_id, stock_code, action, horizons_json, evidence_version, generated_at, data_as_of, evidence_gate_passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (profile_id, stock_code, action, json.dumps(horizons, ensure_ascii=False), evidence_version, generated_at or datetime.now().isoformat(), data_as_of, int(evidence_gate_passed)))
            conn.commit()
            return int(cursor.lastrowid)

    def list_recommendations(self, profile_id: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        """按时间倒序返回建议历史。"""
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id, stock_code, action, horizons_json, evidence_version, generated_at, data_as_of, evidence_gate_passed FROM recommendation_history WHERE profile_id = ? ORDER BY id DESC LIMIT ?", (profile_id, limit)).fetchall()
        return [{"id": row[0], "stock_code": row[1], "action": row[2], "horizons": json.loads(row[3]), "evidence_version": row[4], "generated_at": row[5], "data_as_of": row[6], "evidence_gate_passed": bool(row[7])} for row in rows]

    def _migrate(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_profiles (profile_id TEXT PRIMARY KEY, name TEXT NOT NULL, risk_profile TEXT NOT NULL, cash REAL NOT NULL, version INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS holdings (profile_id TEXT NOT NULL, stock_code TEXT NOT NULL, shares INTEGER NOT NULL, cost_basis REAL NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(profile_id, stock_code));
                CREATE TABLE IF NOT EXISTS recommendation_history (id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, stock_code TEXT NOT NULL, action TEXT NOT NULL, horizons_json TEXT NOT NULL, evidence_version TEXT NOT NULL, generated_at TEXT NOT NULL, data_as_of TEXT, evidence_gate_passed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS rebalance_runs (id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, input_json TEXT NOT NULL, target_json TEXT NOT NULL, orders_json TEXT NOT NULL, fees REAL NOT NULL, created_at TEXT NOT NULL);
            """)
            self._ensure_column(conn, "recommendation_history", "data_as_of", "TEXT")
            self._ensure_column(conn, "recommendation_history", "evidence_gate_passed", "INTEGER NOT NULL DEFAULT 0")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        """为旧本地数据库补齐可向后兼容的列。"""
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _ensure_profile(conn: sqlite3.Connection, profile_id: str) -> None:
        conn.execute("INSERT OR IGNORE INTO portfolio_profiles(profile_id, name, risk_profile, cash) VALUES (?, ?, ?, ?)", (profile_id, "默认组合", "balanced", 0.0))

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        if payload.get("risk_profile") not in {"conservative", "balanced", "aggressive"}:
            raise ValueError("风险档案必须是 conservative、balanced 或 aggressive。")
        if float(payload.get("cash", -1)) < 0:
            raise ValueError("现金不得为负数。")
        for item in payload.get("holdings", []):
            if len(str(item.get("stock_code", ""))) != 6 or int(item.get("shares", -1)) < 0 or float(item.get("cost_basis", -1)) < 0:
                raise ValueError("持仓数据不合法。")
