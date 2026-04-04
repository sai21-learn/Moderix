# Content Moderation OpenEnv — Decision Tree

**Use this to make fast, high-impact choices. Answer each Q, follow the arrow.**

---

## **DECISION 1: Where do I get real posts?**

```
Should I use a real API or synthetic data?

├─ OPTION A: Real API (Reddit, HuggingFace, Twitter)
│  ├─ Pros: Judges love it, realistic, impressive
│  ├─ Cons: API keys, rate limits, takes 1–2 hours
│  ├─ Implementation:
│  │  • Use PRAW (Python Reddit API Wrapper)
│  │  • Or HF datasets library: `datasets.load_dataset('social_bias_frames')`
│  │  • Filter 50 posts, save to data/training_set.json
│  ├─ Time cost: 2 hours
│  └─ ✅ RECOMMENDATION: Do this if you finish foundation by hour 12
│
└─ OPTION B: Synthetic Dataset (hand-written + AI-generated)
   ├─ Pros: Full control, no API keys, instant
   ├─ Cons: Less impressive but still valid
   ├─ Implementation:
   │  • Write 15 posts manually (diverse, realistic)
   │  • Use Claude/ChatGPT to generate 35 more
   │  • Save to data/training_set.json
   ├─ Time cost: 1 hour
   └─ ✅ RECOMMENDATION: Do this if time is tight (< hour 12 left)

DECISION: Real API (you have 3 days). Start here, fallback to synthetic if API fails.
```

---

## **DECISION 2: Which LLM model for baseline?**

```
Which model should I use in inference.py?

├─ OPTION A: Qwen2.5-72B-Instruct (FREE via HF Router)
│  ├─ Cost: $0
│  ├─ Quality: Very good, instruction-following
│  ├─ Speed: ~1–2 sec per post
│  ├─ Setup: Just set HF_TOKEN (free HF account)
│  ├─ Reliability: Stable, widely tested
│  ├─ Baseline score: ~0.70–0.75 reward avg
│  └─ ✅ RECOMMENDATION: Use this (unless you have Claude credits)
│
├─ OPTION B: Claude (if you have credits)
│  ├─ Cost: ~$0.50 per episode (8 posts)
│  ├─ Quality: Excellent, very reliable
│  ├─ Speed: ~0.5–1 sec per post
│  ├─ Setup: Set ANTHROPIC_API_KEY
│  ├─ Reliability: Very stable, good at reasoning
│  ├─ Baseline score: ~0.75–0.82 reward avg (better)
│  └─ ✅ RECOMMENDATION: Use this if you have credits + score is competitive
│
├─ OPTION C: Local model (Mistral 7B)
│  ├─ Cost: $0
│  ├─ Quality: OK, but slower
│  ├─ Speed: 3–5 sec per post (slow)
│  ├─ Setup: Download ~4GB, run locally
│  ├─ Reliability: Can hallucinate JSON
│  ├─ Baseline score: ~0.50–0.60 reward avg (lower)
│  └─ ❌ NOT RECOMMENDED: Too slow for 72-hour sprint
│
└─ OPTION D: Small model (Phi-2, TinyLLaMA)
   ├─ Cost: $0
   ├─ Quality: Poor at classification tasks
   ├─ Speed: 1–2 sec (OK)
   ├─ Baseline score: ~0.40–0.50 reward avg (low)
   └─ ❌ NOT RECOMMENDED: Low baseline = low phase 2 score

DECISION:
  If you have $5+ in HF credits: Qwen2.5-72B ← START HERE
  If you have Claude credits: Use Claude for better score
  Do NOT use local models in 72-hour sprint (too slow, too flaky)
```

---

## **DECISION 3: How do I structure my reward function?**

```
How complex should my grading logic be?

├─ OPTION A: Simple (Accuracy only)
│  ├─ Implementation:
│  │  • decision matches gold label? +1.0 reward
│  │  • else +0.0 reward
│  │  • (OR: partial credit for "close" decisions)
│  ├─ Pros: Fast to implement (30 min), easy to debug
│  ├─ Cons: Doesn't reward confidence calibration
│  ├─ Baseline: ~0.55–0.65 avg reward
│  └─ ✅ RECOMMENDATION: START HERE (hours 13–18)
│
├─ OPTION B: Moderate (Accuracy + Confidence)
│  ├─ Implementation:
│  │  • If correct: reward = 0.8 + 0.2 * confidence
│  │  • If incorrect: reward = 0.2 * (1 - distance_from_correct)
│  │  • Penalize overconfidence when wrong
│  ├─ Pros: Rewards well-calibrated agents, realistic
│  ├─ Cons: More complex grading logic (2 hours)
│  ├─ Baseline: ~0.65–0.75 avg reward
│  └─ ✅ RECOMMENDATION: Use this (better competitive edge)
│
└─ OPTION C: Complex (Accuracy + Confidence + Task-specific metrics)
   ├─ Implementation:
   │  • Toxicity: sigmoid(1 - |pred_score - gold_score|)
   │  • Spam: F1-score with recall weight 2x
   │  • NSFW: macro-F1 across 4 categories
   │  • Reasoning: semantic similarity (embedding cosine)
   ├─ Pros: State-of-the-art, impressive
   ├─ Cons: Hard to debug, takes 4–6 hours
   ├─ Baseline: ~0.70–0.80 avg reward
   └─ ❌ NOT RECOMMENDED: Too time-consuming for sprint
           (only if you finish early)

DECISION: Go with OPTION B (Moderate).
  • Implement by hour 18
  • Shows thoughtful design
  • Competitive baseline score
  • Still time to debug
```

---

## **DECISION 4: How should I structure my episode?**

```
How many posts per episode? Fixed or variable length?

├─ OPTION A: Fixed 8 posts per episode (template recommendation)
│  ├─ Pros: Reproducible, fast (< 1 min per episode), easy to test
│  ├─ Cons: None really
│  ├─ Episodes take: ~30–60 seconds
│  └─ ✅ RECOMMENDATION: Use this (default)
│
├─ OPTION B: Variable length (stop when agent is confident)
│  ├─ Pros: Agents learn early stopping
│  ├─ Cons: Hard to make reproducible, unpredictable duration
│  ├─ Episodes take: 10–90 seconds
│  └─ ❌ NOT RECOMMENDED: Adds variability, harder to grade
│
└─ OPTION C: Longer episodes (16+ posts)
   ├─ Pros: Agents see more data, longer horizon
   ├─ Cons: Slower episodes, harder infrastructure
   ├─ Episodes take: 2–3 minutes
   └─ ❌ NOT RECOMMENDED: Resource-constrained in 72 hours

DECISION: Fixed 8 posts per episode. Lock this down, don't change.
```

---

## **DECISION 5: How do I handle the real API integration?**

```
Should I query Reddit/HF live or cache posts?

├─ OPTION A: Cached dataset (what you'll do)
│  ├─ Implementation:
│  │  • Download posts once
│  │  • Save to data/training_set.json
│  │  • Read from disk in reset()
│  ├─ Pros: Reproducible, fast, no API calls during eval
│  ├─ Cons: Static dataset (not "real-time")
│  ├─ Time: 1 hour to set up
│  └─ ✅ RECOMMENDATION: This is what to do
│
├─ OPTION B: Live API queries
│  ├─ Implementation:
│  │  • Call Reddit API / HF API in reset()
│  │  • Filter + cache for 8 posts
│  │  • Async requests
│  ├─ Pros: "Real-time" flavor
│  ├─ Cons: API failures break grading, rate limits, slow
│  ├─ Time: 3–4 hours
│  └─ ❌ NOT RECOMMENDED: Too risky in sprint
│
└─ OPTION C: Hybrid (cache seed, generate variations)
   ├─ Implementation:
   │  • Load base 50 posts
   │  • Use LLM to generate variations (5x more)
   │  • Cache all variations
   ├─ Pros: Bigger dataset, feels fresh
   ├─ Cons: Takes 2 hours, may not be worth it
   └─ ❌ NOT RECOMMENDED: Nice-to-have, cut it

DECISION: OPTION A (Cached dataset). Load 50 real posts, save to JSON, done.
```

---

## **DECISION 6: How do I test locally before deploying?**

```
What's the minimal test checklist?

├─ UNIT TESTS (skip for now, you have 3 days)
│  └─ ❌ Too slow to implement
│
├─ MANUAL TESTS (do this)
│  ├─ Step 1: Test my_env.py locally
│  │  $ python my_env.py
│  │  Check: reset() returns Observation
│  │  Check: step(Action) returns (obs, reward, done, info)
│  │
│  ├─ Step 2: Test inference.py locally (5 min)
│  │  $ HF_TOKEN=xxx python inference.py
│  │  Check: Produces [START], [STEP], [END] logs
│  │  Check: Final reward is 0.0–1.0
│  │  Check: No crashes
│  │
│  ├─ Step 3: Test Docker build (10 min)
│  │  $ docker build -t content-mod .
│  │  $ docker run -e HF_TOKEN=xxx content-mod
│  │  Check: Runs without error
│  │  Check: Same logs as local run
│  │
│  └─ Step 4: Test HF Space (5 min)
│     Push to HF, wait for build
│     Check: URL returns 200
│     Check: Space logs show [END] with success=true/false
│
└─ ✅ RECOMMENDATION: Do manual tests. Skip unit tests. Takes 30 min total.

DECISION:
  Hour 60: Run all 4 manual tests above
  Hour 65: Fix any failures
  Hour 70: Final commit + submit
  Hour 72: Celebrate 🎉
```

---

## **DECISION 7: What if something breaks at hour 70?**

```
Debugging priority list (fastest fixes first):

1. [STEP] format wrong (e.g., missing field)
   └─ Fix: Check log_step() function, fix typo
   └─ Time: 5 min

2. Inference script hangs / crashes
   └─ Check: Is HF_TOKEN set? Is model available?
   └─ Fix: Test LLM call separately, add timeout
   └─ Time: 10 min

3. Docker build fails
   └─ Check: requirements.txt up to date? Dockerfile syntax?
   └─ Fix: Copy working local version, rebuild
   └─ Time: 10 min

4. Grader always returns same score
   └─ Check: gold_labels loading? Grading logic in _grade_decision()?
   └─ Fix: Add print statements, trace logic
   └─ Time: 15 min

5. HF Space won't deploy
   └─ Check: Dockerfile at root? .gitignore not hiding it?
   └─ Fix: Push again, wait 5 min
   └─ Time: 10 min

6. Baseline scores too low (< 0.40 avg)
   └─ Check: Is grading too harsh? Model picking wrong decisions?
   └─ Fix: Soften grading thresholds OR use better model
   └─ Time: 20 min

⚠️ If broken at hour 71: Don't panic.
  • SUBMIT what works (even if score is low)
  • Judges value effort over perfection
  • A working baseline at 0.50 beats non-submitted 0.95
```

---

## **DECISION 8: How do I allocate my 72 hours optimally?**

```
Timeline guide (with buffer):

Hours 1–6:   Project setup + Antigravity IDE prompts
Hours 7–18:  Core environment + graders
Hours 19–24: Data prep + local testing ✅ CHECKPOINT 1
Hours 25–36: Baseline inference script + Docker
Hours 37–48: openenv.yaml + validation ✅ CHECKPOINT 2
Hours 49–60: HF Space deployment + final polish
Hours 61–66: Pre-submission checklist + fixes
Hours 67–72: Buffer (for unexpected failures + final review)

CRITICAL PATHS (if you run out of time):
├─ If stuck at hour 36:
│  └─ Skip fancy Docker setup, just use local inference.py
│
├─ If stuck at hour 48:
│  └─ Generate openenv.yaml with Antigravity, copy as-is
│
├─ If stuck at hour 60:
│  └─ Push to GitHub instead of HF Space (GitHub is still valid)
│
└─ If stuck at hour 66:
   └─ Submit as-is. A working baseline beats nothing.

DECISION: Stick to timeline. Hit 3 checkpoints.
  ✅ Hour 24: Environment + graders working locally
  ✅ Hour 48: Baseline scores reproducible
  ✅ Hour 66: Deployed + all logs correct
```

---

## **Quick Answers (Just Tell Me What To Do)**

### Q: "Should I use a real API?"
**A:** Yes. Use HuggingFace datasets library. Takes 1 hour. Adds 10 points to your score.

### Q: "Which model?"
**A:** Qwen2.5-72B via HF Router. Free, good, reliable. If stuck, use Claude.

### Q: "How complex should my grading be?"
**A:** Moderate complexity (accuracy + confidence calibration). Not simple, not fancy.

### Q: "How many posts per episode?"
**A:** 8 fixed posts. Don't overthink. Reproducible > creative.

### Q: "What if my baseline score is low?"
**A:** Normal. Baseline doesn't need to be high. Phase 2 eval uses a better agent.

### Q: "Do I need unit tests?"
**A:** No. Skip them. Manual testing only.

### Q: "Should I add extra features?"
**A:** No. Focus on core 4 tasks + clean implementation. Breadth > depth.

### Q: "Can I use async/await or should I use sync?"
**A:** Use async (that's the spec). But if you get stuck, sync also works.

### Q: "What if I run out of time?"
**A:** Submit what you have. A working baseline at 0.50 is better than no submission.

---

## **Final Reminder**

**You have exactly 72 hours. Use this decision tree to avoid decision fatigue.**

- ✅ Pick OPTION A or B for each decision
- ✅ Don't second-guess yourself after hour 30
- ✅ Hit the 3 checkpoints
- ✅ Test end-to-end at hour 60
- ✅ Submit by deadline

**You've got this.** Now stop reading and start building. 🚀
