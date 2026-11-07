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
你是一位兼具华尔街宏观视野与产业基本面深度的买方首席投资官（CIO）兼量化风控专家。请基于我提供的【量化行情快照】并结合最新宏观估值与个股公告，生成一份直接指导实战的《每日市场温度与持仓复盘报告》。

# Core Analytical Framework
1. **穿透指数分子与分母**：
   - 获取最新 **纳指100 / 标普500 的 Forward PE（如 MacroMicro 数据）与 TTM PE（Multpl / GuruFocus 数据）**。
   - 结合输入中的 10年期美债收益率（^TNX），计算并展示 **隐含股权风险溢价 ERP = (1 / Forward PE) - 10Y美债收益率**，判断美股处于“杀估值”还是“杀业绩预期”。
2. **A股/港股紧盯筹码与平准基金动向**：
   - 结合 510300(沪深300ETF) 和 588000(科创50ETF) 的放量情况及A股两市成交额、ETF份额净申赎、两融余额变化，判断“国家队（汇金/国新）”是在逆向抽水还是托底回补。
3. **持仓股敏感事件定性**：
   - 梳理自选股最新 SEC 文件、港交所/上交所公告（如长城汽车最新H股回购记录、微创系停复牌进展）、以及行业高频指标（如 VLCC TD3C 最新日租金 TCE）。
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

def discover_candidate_models(client: genai.Client) -> list:
    """动态探测当前 API Key 可用的最新 Gemini 文本模型，并彻底过滤已下线的 1.5/2.0/2.5 旧模型"""
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
            raw_name = getattr(m, "name", "")
            short_name = raw_name.replace("models/", "")
            if not short_name.startswith("gemini"):
                continue
            if any(k in short_name.lower() for k in skip_keywords):
                continue
            discovered.append(short_name)
        print(f"API Key 在线探测到的新一代模型: {discovered}")
    except Exception as e:
        print(f"在线探测模型列表异常: {e}")

    # 组合优先级列表：优先把官方点名的 3.8-flash 和 3.1-pro-preview 放最前，再追加探测到的其他 3.x 模型
    final_list = []
    for m in priority + discovered:
        if m not in final_list:
            final_list.append(m)
    return final_list

def generate_daily_report(api_key: str, macro_table: str, watchlist_table: str) -> str:
    client = genai.Client(api_key=api_key)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    user_prompt = f"""
今天是 {today_str}。以下是刚刚通过量化脚本抓取的跨市场收盘快照：

### 1. 宏观与核心指数快照
{macro_table}

### 2. 持仓与自选股监控池
{watchlist_table}

### 3. 检索与分析任务
1. 结合 MacroMicro / Multpl 最新 Nasdaq 100 与 S&P 500 的 Forward PE、TTM PE，以及上方 ^TNX 计算当前隐含 ERP。
2. 分析昨日 A股全市场成交额、两融余额变动，以及宽基 ETF（510300、588000）的量能与资金动向。
3. 梳理 SMMT (Summit) 与 VOR (Vor Biopharma) 最新进展与催化剂窗口。
4. 梳理 长城汽车(02333.HK) 最新H股回购支撑位，以及 中远海能 / 招商轮船 相关的 VLCC 油运景气度与赔率。
5. 跟踪 微创机器人-B (02252.HK) 停复牌及审计进展。

请严格按照 System Prompt 的分析框架输出今日深度复盘报告。
"""

    models_to_try = discover_candidate_models(client)
    error_records = []

    for model_name in models_to_try:
        # 阶段 A：优先尝试开启 Google Search 实时联网增强
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
            print(f"{model_name} (含搜索) 调用未通过: {err_msg}")
            error_records.append(f"- `{model_name} [含搜索]`: {err_msg}")

            # 若错误并非 404 (说明模型存在，只是免费层不支持挂载搜索工具或搜索配额受限)，立即尝试不挂工具调用
            if "404" not in err_msg and "NOT_FOUND" not in err_msg:
                try:
                    print(f"正在降级调用 {model_name} (纯量化数据模式)...")
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

    all_errors_str = "\n".join(error_records)
    raise RuntimeError(f"所有候选模型均未成功返回结果，逐项诊断如下:\n{all_errors_str}")

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
            "请检查 GitHub 仓库 **Settings -> Secrets and variables -> Actions -> Repository secrets**。"
        )
        print(err_msg)
        write_summary(err_msg)
        sys.exit(1)

    try:
        print("1. 正在拉取跨市场行情数据...")
        macro_md = fetch_yf_metrics(MACRO_TICKERS)
        watchlist_md = fetch_yf_metrics(WATCHLIST_TICKERS)

        print("2. 正在调用 Gemini 生成投研日报...")
        report = generate_daily_report(api_key, macro_md, watchlist_md)

        print("3. 正在写入报告并推送...")
        write_summary(f"# 📊 每日市场温度与持仓复盘 ({today_str})\n\n{report}")
        push_notifications(report)
        print("✅ 全部执行完毕！")

    except Exception as e:
        tb_str = traceback.format_exc()
        err_summary = (
            f"## ❌ 运行异常诊断 ({today_str})\n\n"
            f"{str(e)}\n\n"
            f"### 详细堆栈日志\n```python\n{tb_str}\n```\n\n"
            f"### 已成功抓取的行情快照备份\n{macro_md if 'macro_md' in locals() else '行情抓取阶段报错'}"
        )
        print(tb_str)
        write_summary(err_summary)
        sys.exit(1)
