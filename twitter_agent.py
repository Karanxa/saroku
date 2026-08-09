#!/usr/bin/env python3
"""
Twitter engagement agent for saroku.
- Generates and posts threads on different strategies
- Monitors mentions and replies to relevant questions
- Hunts for safety/AI discussions to engage with
- Tracks history to avoid repetition and spam
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import anthropic

# Config
HISTORY_DIR = Path.home() / ".saroku"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
POSTS_FILE = HISTORY_DIR / "twitter_posts.json"
ENGAGEMENT_FILE = HISTORY_DIR / "twitter_engagement.json"

STRATEGIES = {
    "technical": "Deep technical explainers about classifier internals, policy DSL, async patterns",
    "launch": "New feature announcements, milestone celebrations, beta releases",
    "tips": "Developer tips: how to use saroku, best practices, common patterns",
    "case_study": "Real-world safety scenarios: agent failures, how saroku catches them",
    "research": "Safety research findings, benchmark interpretations, behavioral properties explained",
}

ENGAGEMENT_QUERIES = [
    "agent safety",
    "LLM guardrails",
    "prompt injection",
    "goal drift",
    "behavioral testing",
    "AI alignment",
    "safety benchmarks",
    "model jailbreak",
    "agentic AI",
]

SAROKU_CONTEXT = """
saroku is an LLM agent safety platform with:
- 8 behavioral properties: sycophancy, honesty, consistency, prompt_injection, trust_hierarchy, minimal_footprint, goal_drift, corrigibility
- Pluggable classifiers: LLM-based, rule-based, HuggingFace models, local 0.5B model, ensembles
- Policy DSL: declarative YAML policies defining execution layers, confidence thresholds, fallbacks
- ExecutionEngine: orchestrates classifiers with cascade/speculative strategies
- Observability: metrics on every classifier invocation (latency, confidence, outcome)
- Framework integration: Google ADK, AutoGen, LangChain
- bench-v1: 96 hand-authored probes for reproducible benchmarking

Key positioning:
- NOT a model, but a platform for composing safety stacks
- Behavior-focused, not accuracy-metric focused
- Designed for agent frameworks, not just LLM APIs
- Observable, so you can tune your policy based on production data

GitHub: https://github.com/Karanxa/saroku
PyPI: pip install saroku
"""


def load_posts_history() -> dict:
    """Load posted thread history."""
    if POSTS_FILE.exists():
        with open(POSTS_FILE) as f:
            return json.load(f)
    return {"posted": [], "strategies_used": {}}


def save_posts_history(history: dict):
    """Save thread history."""
    with open(POSTS_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_engagement_history() -> dict:
    """Load engagement history (replies, retweets)."""
    if ENGAGEMENT_FILE.exists():
        with open(ENGAGEMENT_FILE) as f:
            return json.load(f)
    return {"replied_to": [], "engaged_posts": []}


def save_engagement_history(history: dict):
    """Save engagement history."""
    with open(ENGAGEMENT_FILE, "w") as f:
        json.dump(history, f, indent=2)


def generate_thread(
    strategy: str, num_tweets: int = 7, existing_topics: Optional[list] = None
) -> list[str]:
    """Generate a tweet thread using Claude."""
    client = anthropic.Anthropic()
    existing = "\n".join(f"- {t}" for t in (existing_topics or [])) if existing_topics else "None yet"

    prompt = f"""You are a Twitter/X content strategist for saroku, an LLM agent safety platform.

Your task: Generate a {num_tweets}-tweet THREAD on the "{strategy}" strategy.

SAROKU CONTEXT:
{SAROKU_CONTEXT}

STRATEGY BRIEF: {STRATEGIES[strategy]}

EXISTING TOPICS (avoid repeating):
{existing}

REQUIREMENTS:
1. First tweet: Hook/headline that makes people stop scrolling
2. Tweets 2-{num_tweets - 1}: Build the narrative, provide depth
3. Last tweet: Call to action (GitHub star, try saroku, reply with questions)
4. Each tweet: max 280 chars, conversational tone, avoid self-promotion fluff
5. Thread should be EDUCATIONAL or INSIGHTFUL, not just promotional
6. Use specific examples from saroku when relevant (actual features, actual tradeoffs)
7. No fake metrics or claims — be honest about what saroku does

OUTPUT FORMAT:
Return ONLY the tweets, one per line, numbered 1-{num_tweets}. Example:
1. First tweet text here
2. Second tweet text here
...

Start now:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    tweets = []
    for line in message.content[0].text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            tweet = line.split(".", 1)[1].strip()
            if tweet:
                tweets.append(tweet)

    return tweets[:num_tweets]


def generate_reply(mention: dict, context: str = "") -> Optional[str]:
    """Generate a reply to a mention using Claude.

    Only reply if it's a genuine question or comment worth engaging with.
    Don't over-explain what saroku is unless they ask.
    """
    client = anthropic.Anthropic()

    prompt = f"""Someone mentioned @saroku on Twitter. Here's their post:
"{mention.get('text', '')}"

Generate a SHORT, NATURAL reply that:
1. Actually addresses what they said (show you read it)
2. Adds useful information OR answers their question
3. Max 280 chars
4. Conversational, not corporate/bot-like
5. Only mention saroku features if directly relevant to their question
6. If they just mentioned saroku in passing → don't force a response, return "[SKIP]"

If it's a genuine question: Answer it well.
If it's a comment about agent safety: Share insight, saroku mention optional.
If it's irrelevant spam: Return "[SKIP]"

Generate ONLY the reply text."""

    message = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    reply = message.content[0].text.strip()
    if reply == "[SKIP]" or not reply or len(reply) > 280:
        return None
    return reply


def generate_engagement_reply(query: str, post_text: str) -> Optional[str]:
    """Generate a reply to engage with relevant discussions.

    CRITICAL: Only reply if you have genuine insight. Never mention saroku unless it's
    naturally relevant. Act like a knowledgeable person, not a bot.
    """
    client = anthropic.Anthropic()

    prompt = f"""You are an AI safety researcher who knows agent safety deeply.
You're on Twitter and someone posted something relevant to your expertise.

Their post:
"{post_text}"

Your goal: Reply with genuine insight. Not to promote saroku, but because you have
something meaningful to contribute.

RULES (STRICT):
1. ONLY mention saroku if their problem would be directly solved by it AND it's natural
2. You are NOT a bot, you are a thoughtful researcher
3. Insight > promotion. Always.
4. If their point is good but irrelevant to what you know, skip it
5. Max 280 chars
6. If you can't add novel value, return "[SKIP]"
7. Never say "great question!" or other bot-speak

GUIDANCE:
- If they're discussing agent tool use safety → can mention saroku's runtime check
- If they're discussing model honesty → saroku's sycophancy/honesty properties relevant
- If they're asking basic LLM questions → just answer helpfully, skip saroku mention
- If they're discussing something saroku does poorly → don't mention saroku

Generate ONLY the reply text. If skipping, return "[SKIP]"."""

    message = client.messages.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    reply = message.content[0].text.strip()
    if reply == "[SKIP]" or len(reply) > 280 or not reply:
        return None
    return reply


def post_tweet(text: str, reply_to: Optional[str] = None) -> Optional[str]:
    """Post a single tweet. Returns tweet ID if successful."""
    try:
        if reply_to:
            result = subprocess.run(
                ["npx", "-y", "@xdevplatform/xurl", "--app", "saroku", "reply", reply_to, text],
                capture_output=True,
                text=True,
                timeout=30,
            )
        else:
            result = subprocess.run(
                ["npx", "-y", "@xdevplatform/xurl", "--app", "saroku", "post", text],
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode != 0:
            print(f"❌ Post failed: {result.stderr}")
            return None

        # Extract ID
        import re

        match = re.search(r'"id":"(\d+)"', result.stdout)
        if match:
            return match.group(1)
        return "posted"

    except Exception as e:
        print(f"❌ Error posting: {e}")
        return None


def post_thread(tweets: list[str]) -> Optional[str]:
    """Post tweets as a thread. Returns first tweet ID."""
    if not tweets:
        return None

    first_id = None
    for i, tweet_text in enumerate(tweets):
        tweet_id = post_tweet(tweet_text, reply_to=first_id if i > 0 else None)
        if not tweet_id:
            print(f"❌ Failed at tweet {i + 1}")
            return first_id if i > 0 else None

        if i == 0:
            first_id = tweet_id
        print(f"✅ Tweet {i + 1} posted")

    return first_id


def monitor_mentions() -> list[dict]:
    """Check mentions (placeholder — would need X API integration)."""
    # This would use X API to fetch mentions
    # For now, returns empty
    return []


def search_engagement_topics() -> list[dict]:
    """Search for relevant discussions to engage with."""
    # This would use X API search
    # For now, returns empty
    print("💡 Engagement monitoring requires X API search (TODO)")
    return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run saroku Twitter engagement")
    parser.add_argument(
        "action",
        choices=["post", "monitor", "engage", "full", "history"],
        help="Action to take",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        help="Content strategy (random if omitted)",
    )
    parser.add_argument(
        "--num-tweets",
        type=int,
        default=7,
        help="Number of tweets in thread",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually post",
    )
    args = parser.parse_args()

    # Show history
    if args.action == "history":
        posts = load_posts_history()
        engagement = load_engagement_history()
        print(f"\n📊 Twitter Stats")
        print(f"  Threads posted: {len(posts['posted'])}")
        print(f"  Replies sent: {len(engagement['replied_to'])}")
        print(f"  Engagement posts: {len(engagement['engaged_posts'])}")
        return

    # Post new thread
    if args.action in ["post", "full"]:
        import random

        strategy = args.strategy or random.choice(list(STRATEGIES.keys()))
        posts = load_posts_history()
        existing_topics = [p.get("topic", "") for p in posts["posted"]]

        print(f"\n🚀 Generating {strategy.upper()} thread...")
        try:
            tweets = generate_thread(strategy, args.num_tweets, existing_topics)
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return

        print(f"📝 Generated {len(tweets)} tweets:")
        for i, t in enumerate(tweets, 1):
            print(f"  {i}. {t}")

        if not args.dry_run:
            first_id = post_thread(tweets)
            if first_id:
                posts["posted"].append(
                    {
                        "date": datetime.now().isoformat(),
                        "strategy": strategy,
                        "topic": tweets[0][:50],
                        "first_tweet_id": first_id,
                    }
                )
                posts["strategies_used"][strategy] = posts["strategies_used"].get(strategy, 0) + 1
                save_posts_history(posts)
                print(f"\n✅ Posted! First tweet: {first_id}")

    # Monitor mentions
    if args.action in ["monitor", "full"]:
        print("\n📬 Checking mentions...")
        mentions = monitor_mentions()
        if mentions:
            print(f"Found {len(mentions)} mentions")
        else:
            print("No new mentions (requires X API setup)")

    # Engage with relevant discussions
    if args.action in ["engage", "full"]:
        print("\n💬 Hunting for engagement opportunities...")
        posts = search_engagement_topics()
        if posts:
            print(f"Found {len(posts)} relevant discussions")
        else:
            print("No posts found (requires X API setup)")


if __name__ == "__main__":
    main()
