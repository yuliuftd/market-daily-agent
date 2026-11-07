import os
import datetime
import requests
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

# ==================== 1. 配置自选股与监控池 ====================
MACRO_TICKERS = {
    "^NDX": "纳斯达克100",
    "^GSPC": "标普500",
    "^VIX": "VIX恐慌指数",
    "^TNX": "10年期美债收益率",
    "BZ=F": "布伦特原油",
    "000001.SS": "上证指数",
    "^HSI": "恒生指数",
    "510300.SS": "沪深300ETF(华泰柏瑞)",
    "588000.SS": "科创50ETF(华夏)"
}

WATCHLIST_TICKERS = {
    "QQQ": {"name": "纳指100ETF", "focus": "科技巨头盈利预期与期权负Gamma波动"},
    "SMMT": {"name": "Summit Therapeutics", "focus": "依沃西单抗FDA审评(PDUFA)、HARMONi-3数据及SEC文件"},
    "VOR": {"name": "Vor Biopharma", "focus": "泰它西普全球III期(UPSTREAM MG/SjD)进展、权证与ATM增发动态"},
    "2333.HK": {"name": "长城汽车(H股)", "focus": "每日H股回购披露、月度产销快报、海外报废税补贴回款"},
    "1138.HK": {"name": "中远海能(H股)", "focus": "VLCC TD3C/中东及西非航线TCE运价、霍尔木兹海峡动态"},
    "600026.SS": {"name": "中远海能(A股)", "focus": "AH溢价率、外贸油轮周转与主力资金流向"},
    "601872.SS": {"name": "招商轮船(A股)", "focus": "VLCC现货运价与散货/集运景气度"}
}

# ==================== 2. 确定性量价数据采集 ====================
def fetch_yf_metrics(tickers_dict: dict) -> str:
    """拉取最新收盘价、单日涨跌、5日涨跌、20日量比及20日均线偏离度"""
    rows = []
    for symbol, info in tickers_dict.items():
        name = info["name"] if isinstance(info, dict) else info
        focus = info.get("focus", "-") if isinstance(info, dict) else "-"
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1mo")
            if hist.empty or len(hist) < 2:
                continue
            
            close = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            chg_1d = (close - prev) / prev * 100
            
            chg_5d = 0.0
            if len(hist) >= 6:
                close_5d = hist["Close"].iloc[-6]
                chg_5d = (close - close_5d) / close_5d * 100
                
            ma20 = hist["Close"].tail(20).mean()
            bias_ma20 = (close - ma20) / ma20 * 100
            
            vol_ratio = 1.0
            if "Volume" in hist.columns and hist["Volume"].iloc[-1] > 0:
                vol_ma20 = hist["Volume"].iloc[:-1].tail(20).mean()
                if vol_ma20 > 0:
                    vol_ratio = hist["Volume"].iloc[-1] / vol_ma20

            flag = "⚠️异动" if (abs(chg_1d) >= 3.0 or vol_ratio >= 1.5) else "正常"
            rows.append(
                f"| {symbol} | {name} | {close:.2f} | {chg_1d:+.2f}% | {chg_5d:+.2f}% | "
                f"{vol_ratio:.2f}x | {bias_ma20:+.2f}% | {flag} | {focus} |"
            )
        except Exception as e:
            rows.append(f"| {symbol} | {name} | 获取失败({e}) | - | - | - | - | - | {focus} |")

    header = "| 代码 | 名称 | 最新价 | 单日涨跌 | 5日涨跌 | 20日量比 | MA20偏离 | 状态 | 重点跟踪方向 |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    return header + sep + "\n".join(rows)

def fetch_cn_liquidity() -> str:
    """尝试通过 AkShare 获取A股成交额与两融数据（海外节点超时则自动降级交给Google搜索）"""
    try:
        import akshare as ak
        # 获取A股两市成交额概览
        df_sh = ak.stock_zh_index_daily_em(symbol="sh000001").tail(2)
        df_sz = ak.stock_zh_index_daily_em(symbol="sz399001").tail(2)
        total_amt = (df_sh["amount"].iloc[-1] + df_sz["amount"].iloc[-1]) / 1e12
        prev_amt = (df_sh["amount"].iloc[-2] + df_sz["amount"].iloc[-2]) / 1e12
        return f"A股最新两市合计成交额: {total_amt:.2f} 万亿元 (前一日: {prev_amt:.2f} 万亿元)"
    except Exception:
        return "AkShare海外节点连接受限，请直接通过 Google Search 检索昨日A股两市成交额与两融余额。"

# ==================== 3. System Prompt 与 AI 投研推理 ====================
SYSTEM_PROMPT = """
# Role
你是一位兼具华尔街宏观视野与产业基本面深度的买方首席投资官（CIO）兼量化风控专家。请基于我提供的【量化行情快照】并通过你的 Google Search 工具实时检索最新宏观估值与个股公告，生成一份直接指导实战的《每日市场温度与持仓复盘报告》。

# Core Analytical Framework (分析铁律)
1. **穿透指数分子与分母**：
   - 必须通过搜索获取最新 **纳指100 / 标普500 的 Forward PE（如 MacroMicro 数据）与 TTM PE（Multpl / GuruFocus 数据）**。
   - 结合输入中的 10年期美债收益率（^TNX），实时计算并展示 **隐含股权风险溢价 ERP = (1 / Forward PE) - 10Y美债收益率**，据此判断美股处于“杀估值”还是“杀业绩预期”。
2. **A股/港股紧盯筹码与平准基金动向**：
   - 结合 510300(沪深300ETF) 和 588000(科创50ETF) 的放量情况及最新搜索到的 ETF 份额净申赎、两融余额变化，判断“国家队（汇金/国新）”是在逆向抽水还是托底回补。
3. **持仓股敏感事件定性（区分噪音与逻辑质变）**：
   - 必须联网检索自选股过去 24-48 小时内的最新 SEC 文件（8-K/10-Q/424B5）、港交所/上交所公告（如长城汽车最新一日回购股数与价格、微创系停复牌进展）、以及行业高频指标（如 VLCC TD3C 最新日租金 TCE）。
   - 对单日涨跌超 3% 或量比超 1.5x 的标的，明确判定属于：① 短期情绪/流动性错杀；② 核心基本面受损；③ 底部安全垫加固。
4. **建议必须具体到价格区间与工具**：
   - 严禁输出“建议关注”、“谨慎操作”等套话。
   - 必须给出量化操作预案：左侧分批挂单区间、右侧加仓信号、或期权策略（如高IV且下方有现金/回购铁底时卖出虚值 Cash-Secured Put 的具体行权价区间）。

# Output Structure
请严格按以下四个 Markdown 模块输出：
## 一、 跨市场宏观温度计 (美股 vs. A股/港股)
## 二、 持仓与自选股敏感事件雷达 (逐一拆解异动股与催化剂进展)
## 三、 行业高频景气度追踪 (VLCC油运运价 / 创新药审评 / 车企回购与销量)
## 四、 今日赔率测算与实战行动清单 (编号列出具体操作、价格区间与期权建议)
"""

def generate_daily_report(macro_table: str, watchlist_table: str, cn_liquidity: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    user_prompt = f"""
今天是 {today_str}。以下是刚刚通过量化脚本抓取的跨市场收盘快照：

### 1. 宏观与核心指数快照
{macro_table}
- {cn_liquidity}

### 2. 持仓与自选股监控池
{watchlist_table}

### 3. 强制联网检索任务 (请调用 Google Search 补齐以下最新实时数据)
1. 检索 MacroMicro / Multpl 最新 Nasdaq 100 与 S&P 500 的 Forward PE、TTM PE，并结合上方 ^TNX 计算当前隐含 ERP。
2. 检索昨日 A股全市场成交额、两融余额变动，以及宽基 ETF（沪深300、科创50）是否有大额净申赎。
3. 检索 SMMT (Summit) 与 VOR (Vor Biopharma) 最近 48 小时的最新新闻、分析师评级或 SEC 公告。
4. 检索 长城汽车(02333.HK) 最新披露的 H股回购记录，以及 中远海能 / 招商轮船 相关的最新 VLCC TD3C 日租金报价。
5. 顺带检索 微创机器人-B (02252.HK) 是否有最新复牌或核数师审计进展公告。

请严格按照 System Prompt 的分析框架输出今日深度复盘报告。
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[{"google_search": {}}],
            temperature=0.2
        )
    )
    return response.text

# ==================== 4. 多渠道消息推送 ====================
def push_report(report_md: str):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    # 渠道 1：写入 GitHub Actions Summary（无需任何配置，点开 Action 页面直接阅读排版精美的报告）
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"# 📊 每日市场温度与持仓复盘 ({today_str})\n\n{report_md}\n")

    # 渠道 2：Telegram Bot 推送（可选）
    tg_token = os.environ.get("TG_BOT_TOKEN")
    tg_chat_id = os.environ.get("TG_CHAT_ID")
    if tg_token and tg_chat_id:
        # Telegram 单条消息上限 4096 字符，超出则分段发送
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        chunks = [report_md[i:i+3800] for i in range(0, len(report_md), 3800)]
        for chunk in chunks:
            requests.post(url, json={"chat_id": tg_chat_id, "text": chunk})

    # 渠道 3：微信推送 PushPlus（可选，国内微信直接弹窗接收 Markdown）
    pushplus_token = os.environ.get("PUSHPLUS_TOKEN")
    if pushplus_token:
        requests.post(
            "http://www.pushplus.plus/send",
            json={
                "token": pushplus_token,
                "title": f"📊 投研日报 ({today_str})",
                "content": report_md,
                "template": "markdown"
            }
        )

if __name__ == "__main__":
    print("正在拉取跨市场行情数据...")
    macro_md = fetch_yf_metrics(MACRO_TICKERS)
    watchlist_md = fetch_yf_metrics(WATCHLIST_TICKERS)
    cn_liq = fetch_cn_liquidity()
    
    print("正在调用 Gemini 2.5 Pro 进行实时联网检索与深度归因...")
    report = generate_daily_report(macro_md, watchlist_md, cn_liq)
    
    print("正在推送报告...")
    push_report(report)
    print("执行完毕！")
