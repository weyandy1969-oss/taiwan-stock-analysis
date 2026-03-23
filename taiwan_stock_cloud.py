"""
taiwan_stock_cloud.py  ─  Streamlit Cloud 雲端部署版本
==========================================================
功能：從 Streamlit Cloud Secrets 自動載入 API Keys，
      使用者不需每次手動輸入，其餘功能與 taiwan_stock.py 完全相同。

部署步驟（iPhone 可用）：
  1. 將整個 台股分析/ 資料夾推送至 GitHub
  2. 前往 https://share.streamlit.io → New app
  3. 選擇 Repository / Branch
  4. Main file path 填：taiwan_stock_cloud.py
  5. 點 Advanced settings → Secrets，貼入：
       FINMIND_TOKEN = "你的Token"
       OPENAI_API_KEY = "sk-..."
       GOOGLE_API_KEY = "AIza..."
  6. Deploy！取得公開網址後，iPhone Safari 開啟
     → 點右下方「分享」→「加入主畫面」→ 即成 App 捷徑
"""
import os
import streamlit as st

# ── 從 Streamlit Secrets 預填 API Keys ──────────────────
# 對應 taiwan_stock.py 中 st.text_input 的 key 參數
_SECRET_MAP = {
    "fm_token":        "FINMIND_TOKEN",
    "openai_key_input": "OPENAI_API_KEY",
    "google_key_input": "GOOGLE_API_KEY",
}
for _session_key, _secret_key in _SECRET_MAP.items():
    if _session_key not in st.session_state:
        try:
            st.session_state[_session_key] = st.secrets[_secret_key]
        except Exception:
            pass  # Secrets 未設定時靜默忽略，使用者仍可在側欄手動輸入

# ── 執行主程式（不修改 taiwan_stock.py）──────────────────
_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taiwan_stock.py")
with open(_main, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _main, "exec"), {"__file__": _main, "__name__": "__main__"})
