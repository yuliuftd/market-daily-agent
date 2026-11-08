import os
import sys
import json
import datetime
import traceback
import urllib.parse
import requests
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import markdown

HISTORY_FILE = "market_history.json"

# ==================== 1. 加载自选股配置 ====================
def load_config(filepath="watchlist.json") -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

MACRO_TICKERS = {
    "^NDX": "纳斯达克100",
    "^GSPC": "标普500",
    "^VIX": "VIX恐慌指数",
    "^TNX": "10年期美债收益率",
    "BZ=F": "布伦特原油",
    "000001.SS": "上证指数",
    "^HSI": "恒生指数",
    "^AXJO": "澳洲ASX200指数",
    "510300.SS": "沪深300ETF(华泰柏瑞)",
    "588000.SS": "科创50ETF(华夏)"
}

# ==================== 2. 增量时序数据库 (market_history.json) ====================
DEFAULT_HUIJIN_SERIES = [
    {"date": "2021-12", "value": 680},
    {"date": "2022-12", "value": 690},
    {"date": "2023-06", "value": 710},
    {"date": "2023-12", "value": 1177},
    {"date": "2024-06", "value": 5839},
    {"date": "2024-12", "value": 10439},
    {"date": "2025-06", "value": 12800},
    {"date": "2025-12", "value": 15263},
    {"date": "2026-02", "value": 6288},
    {"date": "2026-05", "value": 1500},
    {"date": "2026-09", "value": 4000},
    {"date": "2026-10", "value": 5700}
]

FALLBACK_3Y_MARGIN_SERIES = [
    {"date": "2023-06", "value": 1.59}, {"date": "2023-09", "value": 1.59}, {"date": "2023-12", "value": 1.65},
    {"date": "2024-02", "value": 1.42}, {"date": "2024-06", "value": 1.48}, {"date": "2024-09", "value": 1.44},
    {"date": "2024-10", "value": 1.71}, {"date": "2024-12", "value": 1.86}, {"date": "2025-03", "value": 1.92},
    {"date": "2025-06", "value": 1.95}, {"date": "2025-09", "value": 2.15}, {"date": "2025-12", "value": 2.35},
    {"date": "2026-01", "value": 2.88}, {"date": "2026-03", "value": 2.31}, {"date": "2026-05", "value": 2.12},
    {"date": "2026-07", "value": 2.18}, {"date": "2026-09", "value": 2.28}, {"date": "2026-10", "value": 2.36}
]

def fetch_eastmoney_margin_api(page_size: int = 1) -> list:
    """调用东方财富两融接口：冷启动时拉取 750 条(约3年)，日常增量仅拉取最新 1 条(pageSize=1)"""
    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get?"
        f"reportName=RPTA_RZRQ_LSHJ&columns=DIM_DATE,RZRQYE&"
        f"pageNumber=1&pageSize={page_size}&sortColumns=DIM_DATE&sortTypes=-1"
    )
    resp = requests.get(url, timeout=10)
    data = resp.json()
    records = data["result"]["data"]
    records.reverse()  # 按时间从早到晚正序排列
    result = []
    for r in records:
        if r.get("RZRQYE") is not None:
            dt_str = str(r["DIM_DATE"])[:10]
            val_trillion = round(float(r["RZRQYE"]) / 1e12, 4)
            result.append({"date": dt_str, "value": val_trillion})
    return result

def update_and_load_history(config: dict) -> dict:
    """
    核心增量存储引擎：
    - 若 market_history.json 不存在，执行冷启动回填过去 3 年两融数据；
    - 若已存在，每天只拉取最新 1 个坐标点 (pageSize=1) 追加到末尾。
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {}
    else:
        store = {}

    # 1. 初始化或同步汇金持仓序列
    if "huijin_series" not in store or not store["huijin_series"]:
        cfg_hj = config.get("huijin_history_billion_rmb", [])
        if cfg_hj:
            store["huijin_series"] = [{"date": x["period"], "value": x["value"]} for x in cfg_hj]
        else:
            store["huijin_series"] = DEFAULT_HUIJIN_SERIES

    # 2. 两融余额：区分【首次冷启动(拉取3年)】与【日常单点增量(pageSize=1)】
    margin_series = store.get("margin_series", [])
    if len(margin_series) < 15:
        print("首次初始化：正在回填过去 3 年(约750个交易日)两融余额长周期序列...")
        try:
            raw_3y = fetch_eastmoney_margin_api(page_size=750)
            # 抽样策略：历史部分每5个交易日(周度)保留1个点，最近30个交易日每天保留1个点，兼顾3年跨度与图表清晰度
            if len(raw_3y) > 35:
                historical_weekly = raw_3y[:-30][::5]
                recent_daily = raw_3y[-30:]
                margin_series = historical_weekly + recent_daily
            else:
                margin_series = raw_3y
        except Exception as e:
            print(f"东财3年历史初始化受限({e})，载入内置3年基准序列，后续按日增量追加...")
            margin_series = list(FALLBACK_3Y_MARGIN_SERIES)
    else:
        print("日常增量更新：正在拉取今日最新 1 个两融坐标点 (pageSize=1)...")
        try:
            latest_pts = fetch_eastmoney_margin_api(page_size=1)
            if latest_pts:
                new_pt = latest_pts[-1]
                if margin_series[-1]["date"] == new_pt["date"]:
                    margin_series[-1]["value"] = new_pt["value"]  # 同日更新
                elif new_pt["date"] > margin_series[-1]["date"]:
                    margin_series.append(new_pt)  # 跨日仅新增1个坐标点
                    print(f"✅ 已新增两融坐标点: {new_pt}")
        except Exception as e:
            print(f"今日两融单点增量拉取超时({e})，保留已有历史序列。")

    store["margin_series"] = margin_series

    # 3. 每日增量记录宽基ETF成交额与上证20年均线轨迹
    etf_daily = store.get("etf_daily_series", [])
    try:
        hs300 = yf.Ticker("510300.SS").history(period="5d")
        kc50 = yf.Ticker("588000.SS").history(period="5d")
        if not hs300.empty and not kc50.empty:
            latest_dt = hs300.index[-1].strftime("%Y-%m-%d")
            hs_amt = round(float(hs300["Close"].iloc[-1] * hs300["Volume"].iloc[-1]) / 1e8, 2)
            kc_amt = round(float(kc50["Close"].iloc[-1] * kc50["Volume"].iloc[-1]) / 1e8, 2)
            
            # 若为空则初始化最近5天，否则仅追加最新1天
            if not etf_daily:
                for idx in range(len(hs300)):
                    d_str = hs300.index[idx].strftime("%Y-%m-%d")
                    h_v = round(float(hs300["Close"].iloc[idx] * hs300["Volume"].iloc[idx]) / 1e8, 2)
                    k_v = round(float(kc50["Close"].iloc[idx] * kc50["Volume"].iloc[idx]) / 1e8, 2) if idx < len(kc50) else 0
                    etf_daily.append({"date": d_str, "hs300_yi": h_v, "kc50_yi": k_v})
            else:
                if etf_daily[-1]["date"] == latest_dt:
                    etf_daily[-1] = {"date": latest_dt, "hs300_yi": hs_amt, "kc50_yi": kc_amt}
                elif latest_dt > etf_daily[-1]["date"]:
                    etf_daily.append({"date": latest_dt, "hs300_yi": hs_amt, "kc50_yi": kc_amt})
            store["etf_daily_series"] = etf_daily[-60:]  # 滚动保留最近60个交易日
    except Exception:
        pass

    # 将更新后的单点增量写回本地 market_history.json（由 GitHub Action 自动 commit 保存）
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    return store

# ==================== 3. 上证指数 20年均线铁底监测器 ====================
def analyze_shanghai_20y_ma(store: dict) -> str:
    """
    利用增量缓存加速计算上证20年均线：
    支持保存前一日均线基准，结合最新收盘价快速算出偏离度。
    """
    try:
        tk = yf.Ticker("000001.SS")
        # 若本地尚未缓存20年均线基数，则拉取一次全量历史并存入 market_history.json
        cached_20y = store.get("sh_20y_cache", {})
        today_month = datetime.date.today().strftime("%Y-%m")

        if cached_20y.get("month") != today_month or "ma_20y" not in cached_20y:
            hist_max = tk.history(period="max")
            cutoff_date = hist_max.index[-1] - pd.DateOffset(years=20)
            hist_20y = hist_max[hist_max.index >= cutoff_date]
            ma_20y = round(float(hist_20y["Close"].mean()), 2)
            store["sh_20y_cache"] = {"month": today_month, "ma_20y": ma_20y}
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False, indent=2)
        else:
            ma_20y = float(cached_20y["ma_20y"])

        # 日常仅需读取现成的 fast_info 或近期数据即可获取最新点位与200日均线
        current_close = float(tk.fast_info["last_price"])
        ma_200d = float(tk.fast_info["two_hundred_day_average"])
        bias_20y = (current_close - ma_20y) / ma_20y * 100
        bias_200d = (current_close - ma_200d) / ma_200d * 100

        if bias_20y <= 5.0:
            signal = "🚨【史诗级铁底信号：极度接近/跌破20年均线，战略级大胆买入区间！】"
        elif bias_20y <= 12.0:
            signal = "🟢【底部低估区域：距20年均线偏离度较低，具备极强长线安全垫】"
        elif bias_20y <= 25.0:
            signal = "🟡【估值中枢温和期：处于20年均线上方合理区间，重在结构与个股赔率】"
        else:
            signal = "🔴【显著偏离20年均线：长周期均值回归引力增强，警惕高位题材风险】"

        return (
            f"- **上证指数最新收盘**: `{current_close:.2f}` 点\n"
            f"- **上证20年均线 (长牛铁底线)**: `{ma_20y:.2f}` 点 | **20年均线偏离度**: `{bias_20y:+.2f}%`\n"
            f"- **上证200日均线 (牛熊分界线)**: `{ma_200d:.2f}` 点 | **200日均线偏离度**: `{bias_200d:+.2f}%`\n"
            f"- **量化系统判定**: {signal}"
        )
    except Exception as e:
        return f"上证20年均线计算异常: {e}"

# ==================== 4. 个股 MA50 / MA200 支撑采集 (前复权 + 现成属性双擎) ====================
def evaluate_ma_support(close: float, ma50: float, ma200: float, chg_1d: float) -> str:
    bias_50 = (close - ma50) / ma50 * 100 if ma50 > 0 else 0
    bias_200 = (close - ma200) / ma200 * 100 if ma200 > 0 else 0

    tags = []
    if abs(chg_1d) >= 3.5:
        tags.append("⚠️单日异动")

    if -3.5 <= bias_200 <= 4.0:
        tags.append("⚓考验MA200强支撑")
    elif -3.0 <= bias_50 <= 3.5:
        tags.append("🛡️回踩MA50支撑")
    elif close < ma200 and bias_200 < -3.5:
        tags.append("📉MA200下方(左侧深水区)")
    elif close > ma50 > ma200:
        tags.append("🚀多头排列")
    else:
        tags.append("〰️均线间震荡")

    return " ".join(tags)

def fetch_market_group_table(group_title: str, tickers_dict: dict) -> str:
    """
    结合前复权K线与 Yahoo 现成的 fast_info 均线属性：
    - 优先使用前复权K线计算 MA50/MA200（消除高股息股票除权跳空造成的假破位）；
    - 当K线不足或超时时，直接秒读 tk.fast_info['fifty_day_average'] 和 'two_hundred_day_average' 现成值。
    """
    rows = []
    for symbol, info in tickers_dict.items():
        name = info["name"] if isinstance(info, dict) else info
        focus = info.get("focus", "-") if isinstance(info, dict) else "-"
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y")  # 默认 auto_adjust=True 前复权

            if not hist.empty and len(hist) >= 50:
                close = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg_1d = (close - prev) / prev * 100
                ma50 = float(hist["Close"].tail(50).mean())
                ma200 = float(hist["Close"].tail(200).mean())
                
                vol_ratio = 1.0
                if "Volume" in hist.columns and float(hist["Volume"].iloc[-1]) > 0:
                    vol_ma20 = float(hist["Volume"].iloc[:-1].tail(20).mean())
                    if vol_ma20 > 0:
                        vol_ratio = float(hist["Volume"].iloc[-1]) / vol_ma20
            else:
                # 直接读取 Yahoo Finance 预计算好的现成均线值
                close = float(tk.fast_info["last_price"])
                prev = float(tk.fast_info["previous_close"])
                chg_1d = (close - prev) / prev * 100
                ma50 = float(tk.fast_info["fifty_day_average"])
                ma200 = float(tk.fast_info["two_hundred_day_average"])
                vol_ratio = 1.0

            bias_50 = (close - ma50) / ma50 * 100
            bias_200 = (close - ma200) / ma200 * 100
            status_tag = evaluate_ma_support(close, ma50, ma200, chg_1d)

            rows.append(
                f"| {symbol} | {name} | {close:.2f} | {chg_1d:+.2f}% | "
                f"{ma50:.2f} ({bias_50:+.1f}%) | {ma200:.2f} ({bias_200:+.1f}%) | "
                f"{vol_ratio:.2f}x | {status_tag} | {focus} |"
            )
        except Exception as e:
            rows.append(f"| {symbol} | {name} | 停牌/无数据 | - | - | - | - | - | {focus} |")

    header = (
        f"#### 📌 {group_title}\n"
        "| 代码 | 名称 | 最新价 | 单日涨跌 | MA50 (偏离度) | MA200 (偏离度) | 20日量比 | 均线支撑状态 | 基本面跟踪核心 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"

# ==================== 5. 长跨度图表渲染引擎 (支持百点以上长序列 POST 渲染) ====================
def build_quickchart_url(chart_config: dict) -> str:
    """优先使用 POST 接口生成短链接图片，突破长周期(3年+)数百个坐标点的 URL 长度限制"""
    try:
        resp = requests.post(
            "https://quickchart.io/chart/create",
            json={"width": 760, "height": 360, "chart": chart_config},
            timeout=10
        )
        if resp.status_code == 200 and resp.json().get("success"):
            return resp.json()["url"]
    except Exception:
        pass
    # 降级使用 GET 编码
    json_str = json.dumps(chart_config, ensure_ascii=False)
    return f"https://quickchart.io/chart?w=760&h=360&c={urllib.parse.quote(json_str)}"

def generate_visual_charts_md(store: dict) -> str:
    # 图表 1：中央汇金系股票ETF持仓市值变化 (2021–2026)
    hj_series = store.get("huijin_series", DEFAULT_HUIJIN_SERIES)
    hj_chart_cfg = {
        "type": "bar",
        "data": {
            "labels": [x["date"] for x in hj_series],
            "datasets": [{
                "label": "中央汇金系宽基ETF持仓市值 (亿元)",
                "data": [x["value"] for x in hj_series],
                "backgroundColor": "rgba(54, 162, 235, 0.7)",
                "borderColor": "rgb(54, 162, 235)",
                "borderWidth": 1
            }]
        },
        "options": {"title": {"display": True, "text": "国家队(中央汇金系)股票ETF持仓市值周期演变 (亿元)"}}
    }
    hj_img_url = build_quickchart_url(hj_chart_cfg)

    # 图表 2：A股两融余额近3年长周期趋势 (每日增量追加1个点)
    m_series = store.get("margin_series", FALLBACK_3Y_MARGIN_SERIES)
    margin_chart_cfg = {
        "type": "line",
        "data": {
            "labels": [x["date"][2:] for x in m_series],  # 显示 YY-MM-DD
            "datasets": [{
                "label": "A股两融余额 (万亿元)",
                "data": [x["value"] for x in m_series],
                "fill": True,
                "pointRadius": 2,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.15)"
            }]
        },
        "options": {
            "title": {
                "display": True,
                "text": f"A股融资融券余额长周期趋势 (万亿元 | 已累计存储 {len(m_series)} 个坐标点)"
            }
        }
    }
    margin_img_url = build_quickchart_url(margin_chart_cfg)

    # 图表 3：增量记录的核心宽基ETF日成交额监控
    etf_series = store.get("etf_daily_series", [])
    etf_md = ""
    if etf_series:
        etf_chart_cfg = {
            "type": "bar",
            "data": {
                "labels": [x["date"][5:] for x in etf_series],
                "datasets": [
                    {"label": "沪深300ETF(510300)成交额(亿)", "data": [x["hs300_yi"] for x in etf_series], "backgroundColor": "rgba(75, 192, 192, 0.7)"},
                    {"label": "科创50ETF(588000)成交额(亿)", "data": [x["kc50_yi"] for x in etf_series], "backgroundColor": "rgba(153, 102, 255, 0.7)"}
                ]
            },
            "options": {"title": {"display": True, "text": "核心宽基ETF每日成交额追踪 (监控国家队脉冲托底)"}}
        }
        etf_md = f"\n![核心宽基ETF成交额追踪]({build_quickchart_url(etf_chart_cfg)})\n"

    return (
        "## 📈 国家队持仓与两融长周期资金看板 (增量持久化存储)\n\n"
        f"![国家队ETF持仓市值变化]({hj_img_url})\n\n"
        f"![A股两融余额长周期趋势]({margin_img_url})\n"
        f"{etf_md}\n"
    )

# ==================== 6. System Prompt 与 AI 投研推理 ====================
SYSTEM_PROMPT = """
# Role
你是一位掌管跨市场（A股、港股、澳股、美股）组合的买方首席投资官（CIO）。请基于我提供的【量化行情与均线支撑快照】并调用 Google Search 检索最新宏观与个股基本面动态，生成一份高信息密度的《每日全球市场温度与持仓复盘报告》。

# Core Analytical Rules
1. **市场温度计 + 上证20年均线战略择时**：
   - 美股部分：检索最新 Nasdaq 100 / S&P 500 的 Forward PE 与 TTM PE，结合 ^TNX（10年期美债收益率）计算隐含 ERP = (1/Forward PE) - 10Y美债收益率。
   - 中国市场部分：重点解读输入数据中的【上证指数20年均线偏离度】。当偏离度处于低位（<12%）甚至接近20年均线时，必须在报告醒目位置发出“长周期铁底区域、战略级大胆买入”的强提醒；同时结合两融余额长周期位置与宽基ETF量能，判断国家队动向。
2. **个股 MA50 / MA200 支撑力度 × 基本面交叉验证（核心任务）**：
   - 扫描【A股、港股、澳股、美股】四张表中所有状态带有 `🛡️回踩MA50支撑`、`⚓考验MA200强支撑` 或 `📉MA200下方(左侧深水区)` 的个股。
   - 结合其最新基本面（财报现金流、大宗商品价格、股息率、回购力度或管线催化剂）进行体检。**只要基本面过关（逻辑未破），必须将其单列出来，发出明确的“均线支撑有效/左侧击球区买入提醒”！**
   - 若基本面存疑或面临重大二元事件（如 SMMT 临近 PDUFA 审批、微创机器人停牌核数师审计），则提示规避盲目接飞刀或改用期权策略。
3. **四大市场全覆盖与跨市场联动**：
   - 联动分析铝业（云铝、中国宏桥）、煤炭高息（陕西煤业、中煤能源H + 澳股 YAL/WHC）、油运与集运（中远海能H、中远海控H + 美股 TRMD）、电信与电力红利（国电电力、中国电信、中国联通）、澳股核心资产（WDS、TWE、YAL、WHC）以及美股科技/高息烟草/Biotech（COIN、QQQ、GOOGL、NVDA、KO、PM、BTI、SMMT、VOR）。
4. **量化操作清单**：
   - 给出具体的【左侧挂单价格区间（锚定MA50/MA200点位）】或【Cash-Secured Put 期权行权价建议】。

# Output Structure
## 一、 跨市场宏观温度计与上证20年均线择时雷达
## 二、 均线支撑共振池：回踩 MA50 / MA200 且基本面过关的重点击球标的
## 三、 持仓股重大敏感事件与催化剂排雷 (Biotech / 财报 / 回购 / 停复牌)
## 四、 跨市场产业景气度联动 (铝/煤/油运/烟草红利/AI科技)
## 五、 今日实战行动清单 (按 A股 / 港股 / 澳股 / 美股 分类给出具体点位与期权策略)
"""

def discover_candidate_models(client: genai.Client) -> list:
    priority = [
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-preview",
        "gemini-3.0-pro",
        "gemini-3.0-flash"
    ]
    skip_keywords = ["1.5", "2.0", "2.5", "embedding", "imagen", "veo", "aqa", "audio", "tts", "vision", "learnlm"]
    discovered = []
    try:
        for m in client.models.list():
            short_name = getattr(m, "name", "").replace("models/", "")
            if short_name.startswith("gemini") and not any(k in short_name.lower() for k in skip_keywords):
                discovered.append(short_name)
    except Exception:
        pass
    final_list = []
    for m in priority + discovered:
        if m not in final_list:
            final_list.append(m)
    return final_list

def generate_daily_report(api_key: str, sh_20y_info: str, macro_table: str, stock_tables: str, store: dict) -> str:
    client = genai.Client(api_key=api_key)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    latest_margin = store.get("margin_series", [{}])[-1]

    user_prompt = f"""
今天是 {today_str}。以下是量化脚本刚刚计算的跨市场最新收盘数据与 MA50 / MA200 / 20年均线偏离度：

### 1. 上证指数 20年均线（长牛铁底线）与两融最新坐标
{sh_20y_info}
- **A股最新两融余额坐标**: `{latest_margin.get('date', '-')}` 余额为 **`{latest_margin.get('value', '-')} 万亿元`**（相比2026年1月历史极值 2.88 万亿元与2024年低点 1.42 万亿元所处水位）。

### 2. 全球核心宏观指数快照
{macro_table}

### 3. 四大市场（A股 / 港股 / 澳股 / 美股）自选股 MA50 & MA200 支撑全景表
{stock_tables}

### 4. 请调用 Google Search 补齐以下实时信息并完成深度归因
1. 检索最新美股 Nasdaq 100 与 S&P 500 的 Forward PE，结合 ^TNX 算出隐含 ERP。
2. 针对上方表格中触发 `🛡️回踩MA50支撑`、`⚓考验MA200强支撑` 或处于 `📉MA200下方` 的标的，逐一核查其基本面是否过关，挑出最值得左侧买入或卖 Put 接货的标的并给出精确到价格的提醒。
3. 检索 SMMT、VOR、科伦博泰(6990.HK)、微创机器人(2252.HK)、长城汽车(2333.HK)、澳股 TWE/WDS/YAL/WHC 的最新公告或催化剂进展。
"""

    models_to_try = discover_candidate_models(client)
    error_records = []

    for model_name in models_to_try:
        try:
            print(f"正在调用 {model_name} (开启 Google Search)...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.2
                )
            )
            if response and response.text:
                return f"> 💡 *本报告由 `{model_name}` (已启用 Google 实时联网检索) 生成*\n\n" + response.text
        except Exception as e_search:
            err_msg = str(e_search)
            error_records.append(f"- `{model_name} [含搜索]`: {err_msg}")
            if "404" not in err_msg and "NOT_FOUND" not in err_msg:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.2
                        )
                    )
                    if response and response.text:
                        return f"> 💡 *本报告由 `{model_name}` 基于实盘量化快照生成*\n\n" + response.text
                except Exception as e_plain:
                    error_records.append(f"- `{model_name} [纯量化]`: {str(e_plain)}")

    raise RuntimeError("所有候选模型均未成功返回结果:\n" + "\n".join(error_records))

# ==================== 7. 输出与推送 ====================
def write_summary(content_md: str):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(content_md + "\n")

def send_email_report(report_md: str, today_str: str):
    """将 Markdown 报告转换为带表格样式的 HTML 邮件并发送"""
    sender = os.environ.get("EMAIL_SENDER", "").strip()
    # 自动去除复制 Gmail 应用专用密码时可能带入的空格
    password = os.environ.get("EMAIL_PASSWORD", "").replace(" ", "").strip()
    receiver_raw = os.environ.get("EMAIL_RECEIVER", "").strip()

    if not (sender and password and receiver_raw):
        print("未配置完整的邮箱环境变量 (EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECEIVER)，跳过邮件发送。")
        return

    receivers = [r.strip() for r in receiver_raw.split(",") if r.strip()]

    # 根据发件箱后缀自动匹配 SMTP 服务器
    domain = sender.split("@")[-1].lower()
    smtp_map = {
        "gmail.com": ("smtp.gmail.com", 465),
        "qq.com": ("smtp.qq.com", 465),
        "foxmail.com": ("smtp.qq.com", 465),
        "163.com": ("smtp.163.com", 465),
        "126.com": ("smtp.126.com", 465),
    }
    smtp_host, smtp_port = smtp_map.get(domain, ("smtp.gmail.com", 465))

    # 将 Markdown 转换为 HTML（开启 tables 与 fenced_code 扩展）
    html_body = markdown.markdown(report_md, extensions=["tables", "fenced_code"])

    # 注入干净的投研研报级 CSS 样式，确保手机/电脑邮件客户端表格与图片完美自适应
    styled_html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1f2937; max-width: 920px; margin: 0 auto; padding: 16px; }}
        h1, h2, h3, h4 {{ color: #111827; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; margin-top: 24px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }}
        th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
        th {{ background-color: #f3f4f6; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f9fafb; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; margin: 10px 0; border: 1px solid #e5e7eb; }}
        blockquote {{ border-left: 4px solid #3b82f6; background: #eff6ff; margin: 12px 0; padding: 8px 14px; color: #1e40af; }}
        code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; font-size: 90%; }}
      </style>
    </head>
    <body>
      {html_body}
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(f"📊 全球市场温度与均线支撑复盘 ({today_str})", "utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)

    # 同时附带纯文本与 HTML 版本，兼容所有邮件客户端
    msg.attach(MIMEText(report_md, "plain", "utf-8"))
    msg.attach(MIMEText(styled_html, "html", "utf-8"))

    try:
        print(f"正在通过 {smtp_host}:{smtp_port} 发送邮件至 {receivers} ...")
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())
        print("✅ 投研日报邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

def push_notifications(report_md: str):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # 1. 发送 HTML 富文本邮件
    send_email_report(report_md, today_str)

    # 2. Telegram 推送（若已配置）
    tg_token = os.environ.get("TG_BOT_TOKEN", "").strip()
    tg_chat_id = os.environ.get("TG_CHAT_ID", "").strip()
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        for i in range(0, len(report_md), 3800):
            requests.post(url, json={"chat_id": tg_chat_id, "text": report_md[i:i+3800]})

    # 3. PushPlus 推送（若已配置）
    pushplus_token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if pushplus_token:
        requests.post(
            "http://www.pushplus.plus/send",
            json={
                "token": pushplus_token,
                "title": f"📊 全球持仓与均线支撑日报 ({today_str})",
                "content": report_md,
                "template": "markdown"
            }
        )

if __name__ == "__main__":
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        write_summary("## ❌ 未检测到 `GEMINI_API_KEY`")
        sys.exit(1)

    try:
        print("1. 正在加载 watchlist.json 配置与 market_history.json 增量数据库...")
        config = load_config("watchlist.json")
        history_store = update_and_load_history(config)

        print("2. 正在计算上证指数 20年均线偏离度...")
        sh_20y_md = analyze_shanghai_20y_ma(history_store)

        print("3. 正在生成长跨度国家队持仓与两融余额可视化图表...")
        charts_md = generate_visual_charts_md(history_store)

        print("4. 正在计算四大市场个股 MA50 / MA200 支撑...")
        macro_md = fetch_market_group_table("全球核心指数与宏观锚点", MACRO_TICKERS)
        a_md = fetch_market_group_table("A股核心持仓与自选", config.get("a_shares", {}))
        hk_md = fetch_market_group_table("港股核心持仓与自选", config.get("hk_shares", {}))
        au_md = fetch_market_group_table("澳股(ASX)核心持仓与自选", config.get("au_shares", {}))
        us_md = fetch_market_group_table("美股核心持仓与自选", config.get("us_shares", {}))
        all_stocks_md = "\n".join([a_md, hk_md, au_md, us_md])

        print("5. 正在调用 Gemini 进行均线支撑 × 基本面交叉验证...")
        ai_report = generate_daily_report(api_key, sh_20y_md, macro_md, all_stocks_md, history_store)

        final_output = (
            f"# 📊 全球市场温度与均线支撑复盘 ({today_str})\n\n"
            f"### 🧭 上证指数 20年均线铁底仪表盘\n{sh_20y_md}\n\n"
            f"{charts_md}\n---\n\n"
            f"{ai_report}\n\n---\n\n"
            f"## 📎 附录：四大市场个股 MA50 & MA200 量化支撑底表\n\n"
            f"{all_stocks_md}"
        )

        write_summary(final_output)
        push_notifications(final_output)
        print("✅ 全部执行完毕！")

    except Exception as e:
        tb_str = traceback.format_exc()
        write_summary(f"## ❌ 运行异常 ({today_str})\n\n`{e}`\n```python\n{tb_str}\n```")
        sys.exit(1)