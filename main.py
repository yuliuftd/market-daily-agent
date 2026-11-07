import os
import sys
import datetime
import traceback
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
    rows = []
    for symbol, info in tickers_dict.items():
        name = info["name"] if isinstance(info, dict) else info
        focus = info.get("focus", "-") if isinstance(info, dict) else "-"
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1mo")
            if hist.empty or len(hist) < 2:
                rows.append(f"| {symbol} | {name} | 暂无数据 | - | - | - | - | - | {focus} |")
                continue
            
            close = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg_1d = (close - prev) / prev * 100
            
            chg_5d = 0.0
            if len(hist) >= 6:
                close_5d = float(hist["Close"].iloc[-6])
                chg_5d = (close - close_5d) / close_5d * 100
                
            ma20 = float(hist["Close"].tail(20).mean())
            bias_ma20 = (close - ma20) / ma20 * 100
            
            vol_ratio = 1.0
            if "Volume" in hist.columns and float(hist["Volume"].iloc[-1]) > 0:
                vol_ma20 = float(hist["Volume"].iloc[:-1].tail(20).mean())
                if vol_ma20 > 0:
                    vol_ratio = float(hist["Volume"].iloc[-1]) / vol_ma20

            flag = "⚠️异动" if (abs(chg_1d) >= 3.0 or vol_ratio >= 1.5) else "正常"
            rows.append(
                f"| {symbol} | {name} | {close:.2f} | {chg_1d:+.2f}% | {chg_5d:+.2f}% | "
                f"{vol_ratio:.2f}x | {bias_ma20:+.2f}% | {flag} | {focus} |"
            )
        except Exception as e:
            rows.append(f"| {symbol} | {name} | 获取异常({str(e)[:20]}) | - | - | - | - | - | {focus} |")

    header = "| 代码 | 名称 | 最新价 | 单日涨跌 | 5日涨跌 | 20日量比 | MA20偏离 | 状态 | 重点跟踪方向 |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    return header + sep + "\n".join(rows)

# ==================== 3. System Prompt 与 AI 投研推理 ====================
SYSTEM_PROMPT = """
# Role
你是一位兼具华尔街宏观视野与产业基本面深度的买方首席投资官（CIO）兼量化风控专家。请基于我提供的【量化行情快照】并通过你的 Google Search 工具实时检索最新宏观估值与个股公告，生成一份直接指导实战的《每日市场温度与持仓复盘报告》。

# Core Analytical Framework
1. **穿透指数分子与分母**：
   - 通过搜索获取最新 **纳指100 / 标普500 的 Forward PE（如 MacroMicro 数据）与 TTM PE（Multpl / GuruFocus 数据）**。
   - 结合输入中的 10年期美债收益率（^TNX），计算并展示 **隐含股权风险溢价 ERP = (1 / Forward PE) - 10Y美债收益率**，判断美股处于“杀估值”还是“杀业绩预期”。
2. **A股/港股紧盯筹码与平准基金动向**：
   - 结合 510300(沪深300ETF) 和 588000(科创50ETF) 的放量情况及最新搜索到的A股两市成交额、ETF份额净申赎、两融余额变化，判断“国家队（汇金/国新）”是在逆向抽水还是托底回补。
3. **持仓股敏感事件定性**：
   - 联网检索自选股过去 24-48 小时内的最新 SEC 文件、港交所/上交所公告（如长城汽车最新H股回购记录、微创系停复牌进展）、以及行业高频指标（如 VLCC TD3C 最新日租金 TCE）。
   - 对单日涨跌超 3% 或量比超 1.5x 的标的，明确判定属于：① 短期情绪/流动性错杀；② 核心基本面受损；③ 底部安全垫加固。
4. **建议具体到价格区间与工具**：
   - 给出量化操作预案：左侧分批挂单区间、右侧加仓信号、或期权策略（如高IV且下方有铁底时卖出虚值 Cash-Secured Put 的具体行权价区间）。

# Output Structure
请严格按以下四个 Markdown 模块输出：
## 一、 跨市场宏观温度计 (美股 vs. A股/港股)
## 二、 持仓与自选股敏感事件雷达 (逐一拆解异动股与催化剂进展)
## 三、 行业高频景气度追踪 (VLCC油运运价 / 创新药审评 / 车企回购与销量)
## 四、 今日赔率测算与实战行动清单 (编号列出具体操作、价格区间与期权建议)
"""

def generate_daily_report(api_key: str, macro_table: str, watchlist_table: str) -> str:
    client = genai.Client(api_key=api_key)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    user_prompt = f"""
今天是 {today_str}。以下是刚刚通过量化脚本抓取的跨市场收盘快照：

### 1. 宏观与核心指数快照
{macro_table}

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

        # 按照官方指引更新为 Gemini 3 系列模型：
    # 优先使用官方推荐的 3.1 Pro Preview，备选使用轻量高速的 3.1 Flash / 2.5 Flash
    models_to_try = [
        "gemini-3.1-pro-preview", 
        "gemini-3.1-flash-preview", 
        "gemini-2.5-flash"
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            print(f"正在尝试调用模型: {model_name} ...")
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
                return f"> 💡 *本报告由 `{model_name}` 结合 Google 实时搜索生成*\n\n" + response.text
        except Exception as e:
            print(f"模型 {model_name} 调用失败: {e}")
            last_error = e

    raise RuntimeError(f"所有模型调用均失败，最后一次报错信息: {last_error}")

# ==================== 4. 报告与诊断输出 ====================
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
        chunks = [report_md[i:i+3800] for i in range(0, len(report_md), 3800)]
        for chunk in chunks:
            requests.post(url, json={"chat_id": tg_chat_id, "text": chunk})

    pushplus_token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
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
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        err_msg = (
            "## ❌ 运行失败：未检测到 `GEMINI_API_KEY`\n\n"
            "请检查你的 GitHub 仓库配置：\n"
            "1. 点击仓库顶部 **Settings** -> 左侧 **Secrets and variables** -> **Actions**。\n"
            "2. 确保在下方的 **Repository secrets**（而不是 Environment secrets）中添加了名为 `GEMINI_API_KEY` 的密钥。"
        )
        print(err_msg)
        write_summary(err_msg)
        sys.exit(1)

    try:
        print("1. 正在拉取跨市场行情数据...")
        macro_md = fetch_yf_metrics(MACRO_TICKERS)
        watchlist_md = fetch_yf_metrics(WATCHLIST_TICKERS)

        print("2. 正在调用 Gemini 进行实时联网检索与深度归因...")
        report = generate_daily_report(api_key, macro_md, watchlist_md)

        print("3. 正在写入报告并推送...")
        write_summary(f"# 📊 每日市场温度与持仓复盘 ({today_str})\n\n{report}")
        push_notifications(report)
        print("✅ 全部执行完毕！")

    except Exception as e:
        tb_str = traceback.format_exc()
        err_summary = (
            f"## ❌ 运行异常诊断 ({today_str})\n\n"
            f"**核心错误原因**: `{str(e)}`\n\n"
            f"### 详细堆栈日志\n```python\n{tb_str}\n```\n\n"
            f"### 已成功抓取的行情快照备份\n{macro_md if 'macro_md' in locals() else '行情抓取阶段报错'}"
        )
        print(tb_str)
        write_summary(err_summary)
        sys.exit(1)
