from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import anthropic
import json
import logging
import os
import traceback
from urllib.parse import unquote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personalize")

_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
if not _anthropic_key:
    logger.warning("ANTHROPIC_API_KEY is missing or empty.")
else:
    logger.info("ANTHROPIC_API_KEY loaded (length=%d).", len(_anthropic_key))

app = FastAPI()
client = anthropic.Anthropic(api_key=_anthropic_key)


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


EMAIL_TOOL = {
    "name": "write_email",
    "description": "Return a personalised abandoned cart recovery email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Subject line: personalised, punchy, under 10 words.",
            },
            "preheader": {
                "type": "string",
                "description": "Preheader: 60-90 characters, complements the subject without repeating it.",
            },
            "body": {
                "type": "string",
                "description": "Email body: max 100 words, conversational, ends with a clear call to action. Do NOT start with the customer's name or a greeting — jump straight into the message. No placeholder text.",
            },
        },
        "required": ["subject", "preheader", "body"],
    },
}


@app.post("/api/personalize")
async def personalize(request: Request):
    try:
        return await _personalize_impl(request)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Unhandled exception: %s: %s\n%s", type(e).__name__, e, tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "exception_type": type(e).__name__,
                "message": str(e),
            },
        )


async def _personalize_impl(request: Request):
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    raw_text = raw_body.decode("utf-8", errors="replace")

    logger.info("=== /api/personalize request ===")
    logger.info("Content-Type: %s", content_type)
    logger.info("Raw body (%d bytes): %s", len(raw_body), raw_text)

    if "application/x-www-form-urlencoded" in content_type.lower():
        decoded_text = unquote_plus(raw_text)
        if "=" in decoded_text and not decoded_text.lstrip().startswith("{"):
            decoded_text = decoded_text.split("=", 1)[1]
    else:
        decoded_text = raw_text

    try:
        data = json.loads(decoded_text)
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "message": e.msg, "raw_body": raw_text},
        )

    # Core fields
    first_name = data.get("first_name", "there")
    account_tier = data.get("account_tier", "Standard")
    clv_score = _safe_int(data.get("clv_score"), 50)
    cart_items = data.get("cart_items", "your items")

    # Enrichment signals
    churn_risk = data.get("churn_risk", "Medium")
    price_check_count = _safe_int(data.get("price_check_count"), 0)
    product_repeat_view_count = _safe_int(data.get("product_repeat_view_count"), 0)
    discount_sensitivity_score = _safe_float(data.get("discount_sensitivity_score"), 0.0)
    cart_value_at_abandonment = _safe_float(data.get("cart_value_at_abandonment"), 0.0)
    product_stock_level = _safe_int(data.get("product_stock_level"), 100)
    abandonment_device_type = data.get("abandonment_device_type", "Desktop")

    # Decision flags
    mention_scarcity = product_stock_level < 5
    offer_discount = discount_sensitivity_score > 0.6 and account_tier in ("Bronze", "Silver")
    acknowledge_mobile = abandonment_device_type == "Mobile"

    # Tone
    if account_tier == "Gold" or clv_score >= 70:
        tone = "warm and personal — acknowledge their loyalty and value to us"
    elif churn_risk == "High" or clv_score < 35:
        tone = "empathetic and reassuring — focus on value, no pressure"
    else:
        tone = "friendly and professional — moderate urgency"

    instructions = [f"Tone: {tone}."]
    if mention_scarcity:
        instructions.append(f"Scarcity: mention that only {product_stock_level} unit(s) remain — keep it factual.")
    if offer_discount:
        instructions.append("Discount: offer 10% off framed as exclusive. Do not invent a different discount amount.")
    else:
        instructions.append("Discount: do NOT mention any discount.")
    if acknowledge_mobile:
        instructions.append("Mobile: briefly acknowledge that checkout is easy on any device.")
    if product_repeat_view_count >= 5:
        instructions.append(f"Interest: the customer viewed this product {product_repeat_view_count} times — raise urgency slightly.")

    prompt = f"""Write an abandoned cart recovery email for this customer.

CUSTOMER:
- Name: {first_name}
- Account tier: {account_tier}
- CLV score: {clv_score}/100
- Cart: {cart_items}
- Cart value: ${cart_value_at_abandonment:.2f}
- Churn risk: {churn_risk}
- Price checks: {price_check_count}
- Product views: {product_repeat_view_count}
- Stock remaining: {product_stock_level}
- Device at abandonment: {abandonment_device_type}

INSTRUCTIONS:
{chr(10).join(f"- {i}" for i in instructions)}

Call write_email with subject, preheader, and body."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system="You are a B2C email copywriter for a fashion brand. Always write in German. Always call the write_email tool.",
        tools=[EMAIL_TOOL],
        tool_choice={"type": "tool", "name": "write_email"},
        messages=[{"role": "user", "content": prompt}],
    )

    tool_block = next(b for b in message.content if b.type == "tool_use")
    return JSONResponse(content=tool_block.input)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/debug/echo")
async def debug_echo(request: Request):
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    raw_text = raw_body.decode("utf-8", errors="replace")

    if "application/x-www-form-urlencoded" in content_type.lower():
        decoded_text = unquote_plus(raw_text)
        if "=" in decoded_text and not decoded_text.lstrip().startswith("{"):
            decoded_text = decoded_text.split("=", 1)[1]
    else:
        decoded_text = raw_text

    parsed = None
    parse_error = None
    try:
        parsed = json.loads(decoded_text)
    except json.JSONDecodeError as e:
        parse_error = {"message": e.msg, "line": e.lineno, "column": e.colno}

    return {
        "method": request.method,
        "content_type": content_type,
        "raw_body": raw_text,
        "decoded_body": decoded_text,
        "parsed_json": parsed,
        "parse_error": parse_error,
    }
