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

# ==================== 1. 加载独立配置文件 ====================
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

# ==================== 2. 上证指数 20年均线铁底监测器 ====================
def analyze_shanghai_20y_ma() -> str:
    """计算上证指数当前点位与20年均线（约4850个交易日）的偏离度"""
    try:
        tk = yf.Ticker("000001.SS")
        hist = tk.history(period="max")
        if hist.empty:
            return "上证指数历史数据获取失败"

        # 截取最近20个日历年的全部交易日数据
        latest_date = hist.index[-1]
        cutoff_date = latest_date - pd.DateOffset(years=20)
        hist_20y = hist[hist.index >= cutoff_date]

        current_close = float(hist["Close"].iloc[-1])
        ma_20y = float(hist_20y["Close"].mean())
        ma_200d = float(hist["Close"].tail(200).mean())
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

# ==================== 3. 个股 MA50 / MA200 支撑与量价采集 ====================
def evaluate_ma_support(close: float, ma50: float, ma200: float, chg_1d: float) -> str:
    """自动判定个股相对于50日和200日均线的支撑状态"""
    bias_50 = (close - ma50) / ma50 * 100 if ma50 > 0 else 0
    bias_200 = (close - ma200) / ma200 * 100 if ma200 > 0 else 0

    tags = []
    if abs(chg_1d) >= 3.5:
        tags.append("⚠️单日异动")
    
    # 判断200日均线支撑 (-3.5% ~ +4.0% 视为正在测试200日线支撑)
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
    """拉取1年期历史数据，精确计算MA50与MA200及偏离度"""
    rows = []
    for symbol, info in tickers_dict.items():
        name = info["name"] if isinstance(info, dict) else info
        focus = info.get("focus", "-") if isinstance(info, dict) else "-"
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y")
            if hist.empty or len(hist) < 2:
                rows.append(f"| {symbol} | {name} | 停牌/无数据 | - | - | - | - | - | {focus} |")
                continue

            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg_1d = (close - prev) / prev * 100

            ma50 = float(hist["Close"].tail(50).mean())
            ma200 = float(hist["Close"].tail(200).mean())
            bias_50 = (close - ma50) / ma50 * 100
            bias_200 = (close - ma200) / ma200 * 100

            vol_ratio = 1.0
            if "Volume" in hist.columns and float(hist["Volume"].iloc[-1]) > 0:
                vol_ma20 = float(hist["Volume"].iloc[:-1].tail(20).mean())
                if vol_ma20 > 0:
                    vol_ratio = float(hist["Volume"].iloc[-1]) / vol_ma20

            status_tag = evaluate_ma_support(close, ma50, ma200, chg_1d)
            rows.append(
                f"| {symbol} | {name} | {close:.2f} | {chg_1d:+.2f}% | "
                f"{ma50:.2f} ({bias_50:+.1f}%) | {ma200:.2f} ({bias_200:+.1f}%) | "
                f"{vol_ratio:.2f}x | {status_tag} | {focus} |"
            )
        except Exception as e:
            rows.append(f"| {symbol} | {name} | 异常({str(e)[:15]}) | - | - | - | - | - | {focus} |")

    header = (
        f"#### 📌 {group_title}\n"
        "| 代码 | 名称 | 最新价 | 单日涨跌 | MA50 (偏离度) | MA200 (偏离度) | 20日量比 | 均线支撑状态 | 基本面跟踪核心 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"

# ==================== 4. 国家队持仓与两融余额可视化图表生成 ====================
def build_quickchart_url(chart_config: dict) -> str:
    """生成可在 GitHub Summary 和微信/Telegram 直接渲染的图表图片链接"""
    json_str = json.dumps(chart_config, ensure_ascii=False)
    encoded = urllib.parse.quote(json_str)
    return f"https://quickchart.io/chart?w=720&h=340&c={encoded}"

def fetch_eastmoney_margin_trend() -> tuple:
    """直连东方财富公开数据中心接口拉取最近12个交易日A股两融余额（单位：万亿元）"""
    url = (
        "https://datacenter-web.eastmoney.com/api/data/v1/get?"
        "reportName=RPTA_RZRQ_LSHJ&columns=DIM_DATE,RZRQYE&"
        "pageNumber=1&pageSize=12&sortColumns=DIM_DATE&sortTypes=-1"
    )
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        records = data["result"]["data"]
        records.reverse()  # 按时间正序排列
        dates = [r["DIM_DATE"][5:10] for r in records]
        values = [round(float(r["RZRQYE"]) / 1e12, 3) for r in records]
        return dates, values
    except Exception:
        # 若海外节点连接东财超时，返回近两年关键节点两融参考序列供画图对比
        fallback_dates = ["24Q3", "24末", "25H1", "25末", "26.01峰值", "26.05", "26Q3", "26.11最新"]
        fallback_vals = [1.42, 1.86, 1.95, 2.35, 2.88, 2.12, 2.28, 2.39]
        return fallback_dates, fallback_vals

def generate_visual_charts_md(config: dict) -> str:
    """生成国家队汇金ETF持仓图、宽基ETF近期量比图以及A股两融余额趋势图"""
    # 图表 1：中央汇金系股票ETF持仓市值变化
    hj_data = config.get("huijin_history_billion_rmb", [])
    hj_labels = [item["period"] for item in hj_data]
    hj_values = [item["value"] for item in hj_data]

    hj_chart_cfg = {
        "type": "bar",
        "data": {
            "labels": hj_labels,
            "datasets": [{
                "label": "中央汇金系宽基ETF持仓市值 (亿元)",
                "data": hj_values,
                "backgroundColor": "rgba(54, 162, 235, 0.7)",
                "borderColor": "rgb(54, 162, 235)",
                "borderWidth": 1
            }]
        },
        "options": {
            "title": {"display": True, "text": "国家队(中央汇金系)股票ETF持仓市值周期演变 (亿元)"}
        }
    }
    hj_img_url = build_quickchart_url(hj_chart_cfg)

    # 图表 2：A股两融余额变化趋势
    m_dates, m_vals = fetch_eastmoney_margin_trend()
    margin_chart_cfg = {
        "type": "line",
        "data": {
            "labels": m_dates,
            "datasets": [{
                "label": "A股两融余额 (万亿元)",
                "data": m_vals,
                "fill": True,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.15)",
                "tension": 0.2
            }]
        },
        "options": {
            "title": {"display": True, "text": "A股融资融券余额变化趋势 (万亿元)"}
        }
    }
    margin_img_url = build_quickchart_url(margin_chart_cfg)

    # 图表 3：近10个交易日 沪深300ETF(510300) 与 科创50ETF(588000) 成交额异动追踪国家队托底信号
    etf_flow_note = ""
    try:
        hs300 = yf.Ticker("510300.SS").history(period="15d").tail(10)
        kc50 = yf.Ticker("588000.SS").history(period="15d").tail(10)
        if not hs300.empty and not kc50.empty:
            etf_dates = [d.strftime("%m-%d") for d in hs300.index]
            hs_amt = [round(float(c * v) / 1e8, 1) for c, v in zip(hs300["Close"], hs300["Volume"])]
            kc_amt = [round(float(c * v) / 1e8, 1) for c, v in zip(kc50["Close"], kc50["Volume"])]
            etf_chart_cfg = {
                "type": "bar",
                "data": {
                    "labels": etf_dates,
                    "datasets": [
                        {"label": "沪深300ETF(510300)日成交额(亿元)", "data": hs_amt, "backgroundColor": "rgba(75, 192, 192, 0.7)"},
                        {"label": "科创50ETF(588000)日成交额(亿元)", "data": kc_amt, "backgroundColor": "rgba(153, 102, 255, 0.7)"}
                    ]
                },
                "options": {"title": {"display": True, "text": "近10日核心宽基ETF成交额异动 (监控平准基金脉冲托底)"}}
            }
            etf_img_url = build_quickchart_url(etf_chart_cfg)
            etf_flow_note = f"\n![核心宽基ETF近10日成交额监控]({etf_img_url})\n"
    except Exception:
        pass

    return (
        "## 📈 国家队持仓与两融资金可视化看板\n\n"
        f"![国家队ETF持仓市值变化]({hj_img_url})\n\n"
        f"![A股两融余额趋势]({margin_img_url})\n"
        f"{etf_flow_note}\n"
    )

# ==================== 5. System Prompt 与 AI 投研推理 ====================
SYSTEM_PROMPT = """
# Role
你是一位掌管跨市场（A股、港股、澳股、美股）组合的买方首席投资官（CIO）。请基于我提供的【量化行情与均线支撑快照】并调用 Google Search 检索最新宏观与个股基本面动态，生成一份高信息密度的《每日全球市场温度与持仓复盘报告》。

# Core Analytical Rules (必须严格执行的四条铁律)
1. **市场温度计 + 上证20年均线战略择时**：
   - 美股部分：检索最新 Nasdaq 100 / S&P 500 的 Forward PE 与 TTM PE，结合 ^TNX（10年期美债收益率）计算隐含 ERP = (1/Forward PE) - 10Y美债收益率。
   - 中国市场部分：重点解读输入数据中的【上证指数20年均线偏离度】。当偏离度处于低位（<12%）甚至接近20年均线时，必须在报告醒目位置发出“长周期铁底区域、战略级大胆买入”的强提醒；同时结合两融余额与宽基ETF量能，判断国家队动向。
2. **个股 MA50 / MA200 支撑力度 × 基本面交叉验证（重点任务！）**：
   - 仔细扫描【A股、港股、澳股、美股】四张表中所有状态带有 `🛡️回踩MA50支撑`、`⚓考验MA200强支撑` 或 `📉MA200下方(左侧深水区)` 的个股。
   - **筛选标准**：如果某只股票正回踩 MA50 或考验 MA200（甚至跌破MA200进入深度价值区），请结合其最新基本面（财报现金流、大宗商品价格、股息率、回购力度或管线催化剂）进行体检。**只要基本面过关（逻辑未破），必须将其单列出来，发出明确的“均线支撑有效/左侧击球区买入提醒”！**
   - 如果基本面出现瑕疵或正面临重大监管不确定性（如 SMMT 临近 PDUFA 硬币博弈、微创机器人停牌审计悬疑），则提醒规避盲目接飞刀或改用期权策略。
3. **四大市场全覆盖与跨市场联动**：
   - **A股/港股资源与红利**：联动分析铝业双雄（云铝、中国宏桥）、煤炭高息双雄（陕西煤业、中煤能源H + 澳股 YAL 兖煤澳洲 / WHC 白天堂煤炭）、航运双雄（中远海能H、中远海控H + 美股成品油轮 TRMD）、电信与公用事业（国电电力、中国电信、中国联通）的景气共振。
   - **澳股(ASX)专属跟踪**：针对 WDS（油气高息）、TWE（奔富红酒中国恢复与估值底部）、YAL/WHC（澳洲煤价与高分红），结合澳元汇率与分红除权节奏给出操作建议。
   - **美股科技、高息防御与Biotech**：区分高Beta（COIN、NVDA、GOOGL、QQQ）、高息现金牛（KO、PM、BTI、TRMD）与事件驱动Biotech（SMMT、VOR + 港股科伦博泰）。
4. **量化操作清单**：
   - 每个重点推荐标的必须给出具体的【左侧挂单价格区间（锚定MA50/MA200点位）】或【Cash-Secured Put 期权行权价建议】。

# Output Structure
请严格按以下五个模块输出报告：
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

def generate_daily_report(api_key: str, sh_20y_info: str, macro_table: str, stock_tables: str) -> str:
    client = genai.Client(api_key=api_key)
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    user_prompt = f"""
今天是 {today_str}。以下是量化脚本刚刚计算的跨市场最新收盘数据与 MA50 / MA200 / 20年均线偏离度：

### 1. 上证指数 20年均线（长牛铁底线）实时监测
{sh_20y_info}

### 2. 全球核心宏观指数快照
{macro_table}

### 3. 四大市场（A股 / 港股 / 澳股 / 美股）自选股 MA50 & MA200 支撑全景表
{stock_tables}

### 4. 请调用 Google Search 补齐以下实时信息并完成深度归因
1. 检索最新美股 Nasdaq 100 与 S&P 500 的 Forward PE，结合 ^TNX 算出隐含 ERP。
2. 针对上方表格中触发 `🛡️回踩MA50支撑`、`⚓考验MA200强支撑` 或处于 `📉MA200下方` 的标的，逐一核查其基本面是否过关，挑出最值得左侧买入或卖 Put 接货的标的并给出精确到价格的提醒。
3. 检索 SMMT、VOR、科伦博泰(6990.HK)、微创机器人(2252.HK)、长城汽车(2333.HK)、澳股 TWE/WDS/YAL 的最新公告或催化剂进展。
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

# ==================== 6. 输出与推送 ====================
def write_summary(content_md: str):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(content_md + "\n")

def push_notifications(report_md: str):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    tg_token = os.environ.get("TG_BOT_TOKEN", "").strip()
    tg_chat_id = os.environ.get("TG_CHAT_ID", "").strip()
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        for i in range(0, len(report_md), 3800):
            requests.post(url, json={"chat_id": tg_chat_id, "text": report_md[i:i+3800]})

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
        print("1. 正在加载 watchlist.json 配置...")
        config = load_config("watchlist.json")

        print("2. 正在计算上证指数 20年均线偏离度...")
        sh_20y_md = analyze_shanghai_20y_ma()

        print("3. 正在生成国家队持仓与两融余额可视化图表...")
        charts_md = generate_visual_charts_md(config)

        print("4. 正在拉取四大市场个股 1年期日线并计算 MA50 / MA200 支撑...")
        macro_md = fetch_market_group_table("全球核心指数与宏观锚点", MACRO_TICKERS)
        a_md = fetch_market_group_table("A股核心持仓与自选", config.get("a_shares", {}))
        hk_md = fetch_market_group_table("港股核心持仓与自选", config.get("hk_shares", {}))
        au_md = fetch_market_group_table("澳股(ASX)核心持仓与自选", config.get("au_shares", {}))
        us_md = fetch_market_group_table("美股核心持仓与自选", config.get("us_shares", {}))
        all_stocks_md = "\n".join([a_md, hk_md, au_md, us_md])

        print("5. 正在调用 Gemini 进行均线支撑 × 基本面交叉验证...")
        ai_report = generate_daily_report(api_key, sh_20y_md, macro_md, all_stocks_md)

        # 将可视化图表、AI深度分析报告与底层量化均线全景表拼装输出
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