"""
TrustNet Telegram Verification Bot.

Production-ready, privacy-respecting Telegram verification bot built with python-telegram-bot.
Wraps the Stage 10 asynchronous verification API.

PRIVACY & SAFETY INVARIANTS:
1. NEVER logs user message text, claims, or image bytes.
2. Only logs user ID, request timestamps, and operational status codes.
3. Enforces sliding-window rate limiting per user.
4. Handles backend timeouts, 429 backpressure, and network drops cleanly.
5. Formats structured responses with:
   - Verdict icon and explanation
   - One-sentence summary
   - Share advice
   - Visual confidence bar
   - Optional link to Stage 11 web dashboard report
   - Standard automated AI disclaimer
"""

import asyncio
from collections import defaultdict
import datetime
import io
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Set up logging: strictly exclude message content
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [User:%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trustnet.telegram_bot")

# Configuration via environment variables
API_BASE_URL = os.environ.get("TRUSTNET_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("TRUSTNET_API_KEY", None)
WEB_DASHBOARD_URL = os.environ.get("TRUSTNET_DASHBOARD_URL", "http://localhost:5173").rstrip("/")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Sliding-window rate limiter: max 5 requests per 60 seconds per user
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60.0
user_request_timestamps: Dict[int, List[float]] = defaultdict(list)


def check_rate_limit(user_id: int) -> Tuple[bool, float]:
    """
    Checks if a user is within sliding-window rate limits.
    Returns: (is_allowed: bool, retry_after_seconds: float)
    """
    now = time.monotonic()
    timestamps = user_request_timestamps[user_id]

    # Filter out timestamps older than the window
    valid_timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    user_request_timestamps[user_id] = valid_timestamps

    if len(valid_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        oldest = valid_timestamps[0]
        retry_after = round(RATE_LIMIT_WINDOW_SECONDS - (now - oldest), 1)
        return False, max(1.0, retry_after)

    valid_timestamps.append(now)
    return True, 0.0


def render_confidence_bar(confidence: float) -> str:
    """Renders a 10-segment visual Unicode confidence bar."""
    conf = max(0.0, min(1.0, confidence))
    filled_blocks = int(round(conf * 10))
    empty_blocks = 10 - filled_blocks
    bar = "█" * filled_blocks + "░" * empty_blocks
    pct = int(round(conf * 100))
    return f"`[{bar}]` **{pct}%**"


def format_verification_response(
    report_envelope: Dict[str, Any],
    dashboard_base_url: Optional[str] = None,
) -> str:
    """
    Formats the verification result into clean Telegram MarkdownV2 or standard Markdown.
    """
    claims = report_envelope.get("claims", [])
    report_id = report_envelope.get("report_id", "")

    if not claims:
        return (
            "🔍 **TrustNet Verification Result**\n\n"
            "🟡 **Verdict**: *Insufficient Evidence*\n"
            "No extractable factual assertions could be verified in the submitted message.\n\n"
            "⚠️ *Disclaimer: TrustNet is an AI verification assistant. Consult primary journalistic sources.*"
        )

    # Primary claim report
    primary = claims[0]
    verdict = primary.get("verdict", "insufficient_evidence").lower()
    conf = float(primary.get("confidence", 0.5))
    share_advice = primary.get("share_advice", "share_with_caution")
    claim_text = primary.get("claim", {}).get("text", "")

    # Verdict iconography
    if verdict == "verified":
        v_title = "🟢 **VERIFIED**"
        v_desc = "The statement is supported by credible evidence."
    elif verdict == "contradicted":
        v_title = "🔴 **CONTRADICTED**"
        v_desc = "The statement is refuted by verified facts or evidence."
    else:
        v_title = "🟡 **INSUFFICIENT EVIDENCE**"
        v_desc = "Available evidence is inconclusive, absent, or conflicting."

    # Share advice mapping
    advice_map = {
        "safe_to_share": "✅ **Safe to Share** (Supported by primary sources)",
        "share_with_caution": "⚠️ **Share with Caution** (Context or evidence is incomplete)",
        "do_not_share_as_confirmed": "🛑 **Do Not Share as Confirmed** (High risk of misinformation)",
    }
    advice_str = advice_map.get(share_advice, "⚠️ **Share with Caution**")

    # Construct formatted message
    lines = [
        f"{v_title}",
        f"_{v_desc}_\n",
        f"📝 **Evaluated Claim**:\n\"{claim_text}\"\n",
        f"📊 **Confidence**: {render_confidence_bar(conf)}",
        f"🛡️ **Sharing Guidance**: {advice_str}\n",
    ]

    # Add evidence source mentions if present
    evidence_list = primary.get("evidence", [])
    if evidence_list:
        lines.append("📚 **Top Evidence Sources**:")
        for ev in evidence_list[:2]:
            title = ev.get("title") or ev.get("source") or "Indexed Document"
            snippet = ev.get("text", "")[:120].strip()
            lines.append(f"• *{title}*: \"{snippet}...\"")
        lines.append("")

    # Add dashboard link if configured
    if dashboard_base_url and report_id:
        lines.append(f"🔗 [View Detailed Web Audit Report]({dashboard_base_url}/reports/{report_id})\n")

    # Standard ethical disclaimer
    lines.append(
        "⚠️ _Disclaimer: TrustNet uses calibrated multimodal AI to detect misinformation. "
        "It is not an infallible authority. Always independently check critical news._"
    )

    return "\n".join(lines)


async def call_verification_api(
    text: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: str = "image.jpg",
    base_url: str = API_BASE_URL,
    api_key: Optional[str] = API_KEY,
    poll_timeout_s: float = 35.0,
) -> Dict[str, Any]:
    """
    Submits a claim to POST /v1/verify and polls /v1/jobs/{job_id} until completed.
    """
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Prepare multipart payload
        data = {}
        files = {}
        if text:
            data["text"] = text
        if image_bytes:
            files["image"] = (image_filename, io.BytesIO(image_bytes), "image/jpeg")

        if not text and not image_bytes:
            raise ValueError("No text or image provided.")

        # 1. Submit verification job
        submit_url = f"{base_url}/v1/verify"
        resp = await client.post(submit_url, data=data, files=files if files else None, headers=headers)

        if resp.status_code == 429:
            raise RuntimeError("API rate limit reached (HTTP 429). Please retry shortly.")
        if resp.status_code == 413:
            raise RuntimeError("Image file exceeds the permitted upload size limit.")
        if resp.status_code >= 400:
            raise RuntimeError(f"API rejected request with status HTTP {resp.status_code}: {resp.text}")

        job_data = resp.json()
        job_id = job_data.get("job_id")
        if not job_id:
            raise RuntimeError("No job_id returned by verification endpoint.")

        # 2. Poll job status
        poll_url = f"{base_url}/v1/jobs/{job_id}"
        start_time = time.monotonic()

        while time.monotonic() - start_time < poll_timeout_s:
            poll_resp = await client.get(poll_url, headers=headers)
            if poll_resp.status_code != 200:
                await asyncio.sleep(0.5)
                continue

            j_status = poll_resp.json()
            status_str = j_status.get("status")

            if status_str == "done":
                report = j_status.get("report")
                if not report:
                    raise RuntimeError("Job finished but no report was attached.")
                return report
            elif status_str == "failed":
                err = j_status.get("error", "Unknown pipeline execution failure.")
                raise RuntimeError(f"Verification pipeline failed: {err}")

            await asyncio.sleep(0.5)

        raise TimeoutError(f"Verification timed out after {poll_timeout_s} seconds.")


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start greeting."""
    user = update.effective_user
    uid = user.id if user else 0
    logger.info(f"User {uid} triggered /start")

    welcome_text = (
        "👋 **Welcome to TrustNet Fact-Checking Bot**\n\n"
        "Send me any forwarded claim, news statement, or social media image (in English, Hindi, or Hinglish), "
        "and I will analyze its credibility using grounded multimodal evidence retrieval.\n\n"
        "**How to use**:\n"
        "• Send a **text message** containing the claim.\n"
        "• Send a **photo** (with or without a caption).\n"
        "• Use /help for guidelines, rate limits, and privacy guarantees."
    )
    if update.message:
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /help command."""
    help_text = (
        "ℹ️ **TrustNet Bot Help & Documentation**\n\n"
        "**Supported Modalities**:\n"
        "1. Text claims: Statements asserted as factual.\n"
        "2. Image verification: Photos, screenshots, infographics.\n"
        "3. Multilingual: Standard English, native Hindi (Devanagari), and Hinglish (code-mixed).\n\n"
        "**Privacy & Security Invariant**:\n"
        "🔒 *TrustNet never retains or logs your submitted message text or images. "
        "All processing is ephemeral and privacy-preserving.*\n\n"
        "**Rate Limits**:\n"
        "Users are limited to 5 verification requests per minute."
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def handle_incoming_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes incoming text and/or photo messages for verification."""
    if not update.message:
        return

    user = update.effective_user
    user_id = user.id if user else 0

    # 1. Check Rate Limit
    allowed, retry_after = check_rate_limit(user_id)
    if not allowed:
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text(
            f"⏳ You have reached the request limit. Please wait {retry_after}s before sending another claim.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # 2. Extract inputs (Zero logging of content)
    text_content = update.message.text or update.message.caption or ""
    image_bytes = None

    if update.message.photo:
        # Download the highest resolution photo version
        photo_obj = update.message.photo[-1]
        try:
            photo_file = await photo_obj.get_file()
            buf = io.BytesIO()
            await photo_file.download_to_memory(buf)
            image_bytes = buf.getvalue()
        except Exception as e:
            logger.error(f"Failed to fetch photo from Telegram for user {user_id}: {e}")
            await update.message.reply_text("⚠️ Could not download the provided image. Please re-send.")
            return

    if not text_content and not image_bytes:
        await update.message.reply_text("Please provide a text statement or photo to verify.")
        return

    # Notify user that processing has started
    status_msg = await update.message.reply_text("🔎 _Analyzing claim against verified evidence sources..._", parse_mode=ParseMode.MARKDOWN)

    # 3. Call backend verification API
    try:
        report = await call_verification_api(
            text=text_content if text_content else None,
            image_bytes=image_bytes,
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
        formatted_reply = format_verification_response(report, dashboard_base_url=WEB_DASHBOARD_URL)
        await status_msg.edit_text(formatted_reply, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
        logger.info(f"Verification successfully completed for user {user_id}")
    except httpx.ConnectError:
        logger.error(f"API connection refused for user {user_id}")
        await status_msg.edit_text(
            "⚠️ **TrustNet verification backend is currently unreachable.**\n"
            "The service may be offline or undergoing maintenance. Please try again shortly.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except TimeoutError:
        logger.error(f"Verification request timed out for user {user_id}")
        await status_msg.edit_text(
            "⏱️ **Verification Timed Out**\n"
            "The pipeline took longer than expected to retrieve evidence. Please retry with a shorter statement.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Verification error for user {user_id}: {type(e).__name__}")
        await status_msg.edit_text(
            f"⚠️ **Verification Error**: {str(e)}",
            parse_mode=ParseMode.MARKDOWN,
        )


def build_telegram_application(token: str) -> Application:
    """Builds and registers handlers for the Telegram bot application."""
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, handle_incoming_message))

    return app


def main():
    token = BOT_TOKEN or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not token:
        sys.stderr.write("ERROR: TELEGRAM_BOT_TOKEN environment variable or command line argument is required.\n")
        sys.exit(1)

    print(f"[TrustNet Bot] Starting Telegram bot connected to API at {API_BASE_URL}...")
    app = build_telegram_application(token)
    app.run_polling()


if __name__ == "__main__":
    main()
