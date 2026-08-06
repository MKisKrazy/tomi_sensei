from google.adk.agents import LlmAgent , SequentialAgent
from .tools.stock_tools import fetch_stock_price, get_company_info,convert_usd_to_inr
from .callbacks.guardrails import validate_ticker_before_tool,add_disclaimer_after_model,audit_log_before_agent


# Stock Analyst
stock_analyst = LlmAgent(
    name="StockAnalyst",
    model="gemini-3.5-flash-lite",
    description="Analyzes individual stocks by fetching live market data.",
    instruction="""You are a Stock Analyst at TomiSensei.

Your job is to analyze individual stocks using your tools.
When asked about a stock:
1. Use fetch_stock_price to get current price and key metrics.
2. Use convert_usd_to_inr to convert USD to INR 
3. Use get_company_info to get company background.
4. Provide a clear, concise analysis:
   - Current price and valuation (P/E ratio)
   - 52-week range context (is it near highs or lows?)
   - Company sector and business summary
   - A brief outlook (bullish/bearish/neutral)

Keep your analysis factual and data-driven.
Do NOT make specific buy/sell recommendations.""",
tools=[fetch_stock_price,get_company_info,convert_usd_to_inr] ,
before_tool_callback=[validate_ticker_before_tool]
)

# ── Portfolio Advisor ─────────────────────────────────
# note: model is the V1 default — V2 overrides via model_switcher_callback
portfolio_advisor = LlmAgent(
    name="PortfolioAdvisor",
    model="gemini-3.5-flash-lite",
    description="Advises on portfolio allocation based on user preferences.",
    instruction="""You are a Portfolio Advisor at TomiSensei.
Your job is to recommend portfolio allocations based on:
- The stock analyses provided by StockAnalyst
- The user's preferences from session state (check for: risk_tolerance, investment_budget, investment_horizon)

Guidelines by risk tolerance:
- Conservative: max 40% stocks, 60%+ bonds/stable assets
- Moderate: 50-70% stocks, 30-50% bonds
- Aggressive: 70-90% stocks, 10-30% bonds

# Use your tools:
# - calculate_compound_returns: to project growth over the investment horizon
# - calculate_portfolio_allocation: to compute dollar amounts for each position

Always explain your reasoning.""",

)

#  Report Generator 

report_generator = LlmAgent(
    name="ReportGenerator",
    model="gemini-3.5-flash-lite",
    description="Generates formatted portfolio analysis reports.",
    instruction="""You are the Report Generator at TomiSensei.
Create a comprehensive portfolio report including:
1. Executive Summary
2. Stock Analysis
3. Recommended Allocation
4. Risk Assessment
5. Save the report as a versioned artifact using save_portfolio_report tool

Format your output in clean Markdown.""",

)


#Analysis Pipeline
analysis_pipeline = SequentialAgent (
    name="AnalysisPipeline",
    description="runs the full analysis: analyze stocks -> advise -> report.",
    sub_agents=[stock_analyst, portfolio_advisor,report_generator],
)

root_agent = LlmAgent(
    name="TomiSensei",
    model="gemini-3.5-flash-lite",
    description="AI Wealth Advisor - analyses stocks and builds portfolios.",
    instruction="""You are **TomiSensei**, an AI Wealth Advisor.

You help users analyze stocks and build investment portfolios.
You have a team of specialist agents to help you:
- **AnalysisPipeline**: Full sequential flow (analyze → advise → report)

## Conversation Flow
1. Greet the user warmly
2. Use LoadMemoryTool to check for any past preferences or conversations with this user
3. Ask about their investment goals, budget, risk tolerance
4. Save Preferences using save_user_preferences tool once you have the info
5. When they want analysis, transfer to AnalysisPipeline

Be conversational, warm, and professional.
Always include: "This is AI-generated analysis, not financial advice." """,
sub_agents=[analysis_pipeline],
before_agent_callback=[audit_log_before_agent],
after_model_callback=[add_disclaimer_after_model]
)