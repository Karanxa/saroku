# saroku Twitter Agent

Autonomous agent for growing saroku's Twitter presence through educational content and genuine engagement.

## Quick Start

```bash
cd /home/karan/saroku

# Generate and post a single thread
python twitter_agent.py post

# Run full engagement cycle (post + monitor mentions + hunt engagement)
python twitter_agent.py full

# Dry-run (see tweets without posting)
python twitter_agent.py post --dry-run

# Check history
python twitter_agent.py history
```

## How It Works

**1. Thread Generation** (every post)
- Rotates through 5 strategies: technical, launch, tips, case_study, research
- Generates 7-tweet threads using Claude (Haiku)
- Tracks topics to avoid repetition
- Posts via xurl (native Twitter API)

**2. Engagement Monitoring** (future)
- Scans mentions and replies
- Detects genuine questions about saroku/agent safety
- Replies with insights (not promotions)
- Tracks engagement history

**3. Hunt Mode** (future)
- Searches for relevant AI/safety discussions
- Adds meaningful value to conversations
- Never spams or self-promotes
- Identifies topics worth engaging on

## Configuration

**Env vars needed:**
- `X_OAUTH2_CLIENT_ID` — from X Developer Portal
- `X_OAUTH2_CLIENT_SECRET` — from X Developer Portal
- `X_OAUTH2_ACCESS_TOKEN` — OAuth access token
- `X_OAUTH2_REFRESH_TOKEN` — OAuth refresh token
- `ANTHROPIC_API_KEY` — for Claude API

(Already set in `/home/karan/saroku/.env`)

## Content Strategies

| Strategy | Purpose | Frequency |
|----------|---------|-----------|
| **technical** | Deep dives into architecture, classifiers, policy DSL | 2x/week |
| **tips** | Developer best practices, how-to guides | 2x/week |
| **launch** | New features, releases, milestones | 1x/week |
| **case_study** | Real-world safety problems saroku solves | 1x/week |
| **research** | Safety benchmarks, research findings | 1x/week |

## Engagement Philosophy

**DO:**
- Answer genuine technical questions
- Share insights about agent safety
- Discuss tradeoffs and limitations honestly
- Link to specific GitHub examples
- Engage with other safety researchers

**DON'T:**
- Spam or mention saroku unprompted
- Act like a bot (use natural language)
- Oversell or mislead
- Engage in flamewars
- Post more than once per day (unless special event)

## Scheduling (Optional)

To run autonomously on a schedule:

```bash
# Add to crontab
0 9 * * 1,3,5 cd /home/karan/saroku && python twitter_agent.py post >> ~/.saroku/agent.log 2>&1
```

This posts a thread Mon/Wed/Fri at 9am UTC.

## History & Tracking

Posts are tracked in `~/.saroku/twitter_posts.json`:
```json
{
  "posted": [
    {
      "date": "2026-08-09T...",
      "strategy": "technical",
      "topic": "ExecutionEngine speculative strategy...",
      "first_tweet_id": "2086528170..."
    }
  ],
  "strategies_used": {
    "technical": 3,
    "tips": 2,
    "launch": 1
  }
}
```

Engagement tracked in `~/.saroku/twitter_engagement.json`.

## Metrics to Watch

- **Engagement rate** — Replies, retweets, likes per thread
- **Follower growth** — Compound growth week-over-week
- **Reply quality** — Do people ask good follow-up questions?
- **Conversions** — Stars on GitHub, PRs, email signups
- **Sentiment** — Are people positive about saroku?

## Troubleshooting

**"xurl command not found"**
```bash
npm install -g @xdevplatform/xurl
```

**"ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY=sk-...
```

**"Token expired"**
Refresh X OAuth tokens in `/home/karan/saroku/.env`

**"No tweets generated"**
Check Claude API availability and rate limits

## Next Steps

1. **Test locally** — Run with `--dry-run` a few times
2. **Enable scheduling** — Set up cron job for 3x/week posts
3. **Monitor metrics** — Track engagement, adjust strategies
4. **Add mention monitoring** — Integrate X API for replies
5. **Refine voice** — Tune prompts based on what resonates

## Files

- `twitter_agent.py` — Main agent script
- `TWITTER_AGENT.md` — This guide
- `.claude/agents/twitter-content-agent.md` — Agent definition for scheduled runs
- `~/.saroku/twitter_posts.json` — Post history
- `~/.saroku/twitter_engagement.json` — Engagement history
