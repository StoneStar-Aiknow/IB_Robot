# Addressee semantic gate (L3) — training toolkit

The agent stack can decide whether a transcript is addressed to the robot before
it reaches the planner. Three layers exist in `ibrobot_agent.addressee`: a wake
signal, a speaker anchor, and — the part this directory trains — a text
classifier over the transcript itself.

The classifier is a **veto**, never an oracle: it may drop an utterance that is
clearly not meant for the robot, and it never promotes one. Skill words from the
catalog and corrupted wake words override it, stop/confirm/cancel bypass it
entirely, and any failure (missing model, unreadable JSON) disables it unless
`semantic_gate_fail_open: false` says to fail the launch instead.

Design, measurements and rollout plan:
[`docs/aimdk_x2_addressee_semantic_gate_design_zh.md`](../../docs/aimdk_x2_addressee_semantic_gate_design_zh.md).

## Artefact

`models/addressee/addressee_zh_v1.json` — plain JSON, ~200 KB, standard library only
at runtime. The repository ignores `models/` (model bundles are deployed, not
committed), so the artefact travels with the deployment: train it from the corpus
with the commands below, or copy it to the board path named in the profile.

| field | meaning |
| --- | --- |
| `feature.ngrams` / `dim` / `hash` / `pad` | char 1–3 grams, dim 65536, FNV-1a 32, `<text>` padding |
| `wake_words` | wake spellings stripped before scoring (training and runtime share this list) |
| `bias`, `weights` | sparse `[[index, weight], …]` logistic regression |
| `training` | epochs, seed, corpus SHA-256, class counts |
| `cross_validation` | the threshold table measured with grouped 5-fold CV |

FNV-1a is part of the format because Python's `hash()` is randomised per
process: a model exported with it scores 0 everywhere in the loading process.
That regression is covered by `test_semantic_gate.py`.

Runtime cost: 45 µs per sentence, no allocations beyond the gram counter.

## Workflow

```bash
# 1. collect: the bridge appends one JSON line per transcript in shadow mode
#    (semantic_gate_log_path), e.g. /var/log/ibrobot/semantic_gate.jsonl

# 2. label: most-uncertain sentences first, unlabelled, plain text
python3 tools/addressee/build_dataset.py --logs /var/log/ibrobot/semantic_gate.jsonl \
    --labelled corpus.txt --out corpus_next.txt --candidates to_label.txt
#    append " 是" or " 不是" to a line in to_label.txt, move it into corpus.txt

# 3. train (needs numpy)
python3 tools/addressee/train_model.py --corpus corpus.txt \
    --out models/addressee/addressee_zh_v2.json --epochs 120

# 4. evaluate the full rule (model + catalog lexicon + wake prefixes)
python3 tools/addressee/eval_model.py --model models/addressee/addressee_zh_v2.json \
    --corpus corpus.txt --threshold 0.10

# 5. deploy: copy the artefact to the board path in the profile, set the mode
#    semantic_gate_mode: shadow   -> log only (use this first)
#    semantic_gate_mode: enforce  -> drop rejected utterances
```

`score_text.py` scores individual sentences with the runtime module, so a
debugging session and the bridge always agree:

```bash
python3 tools/addressee/score_text.py --model models/addressee/addressee_zh_v1.json \
    -- 灵犀灵犀挥挥手 帮我抓取橘子 你是谁 第二件事情呢就是我们成立了全域安全部专门设计
# 1.000  accept  灵犀灵犀挥挥手  (skill word)
# 1.000  accept  帮我抓取橘子  (model)
# 0.000  REJECT  你是谁  (score 0.000 below 0.10)
# 0.000  REJECT  第二件事情呢就是我们成立了全域安全部专门设计  (score 0.000 below 0.10)
```

(`--lexicon` overrides the catalog-derived list; without it the list comes from
the skill catalog, which is what the bridge uses.)

`emit_lexicon.py` regenerates the `semantic_gate_lexicon` list from
`src/agent/skill_catalog/config/skills/*/manifest.yaml` (`description.aliases_zh`,
minus ordinary dialogue such as 是的 / 同意, which as an override would only
ever accept):

```bash
python3 tools/addressee/emit_lexicon.py --yaml   # paste into the profile
```

## Corpus format

One example per line, either `<text> 是` / `<text> 不是`, or
`{"text": "...", "label": 0|1}`. Labels are the only thing that makes this
worth doing: the gate is trained on what people actually said to (and near) the
robot, including the ASR's own mangled spellings — those are the hard positives.

## Results of v1 (639 sentences, 211 addressed / 428 chatter)

Grouped 5-fold CV (near-duplicate paraphrases kept in one fold), model alone:

| threshold | accuracy | command recall | chatter dropped |
| --- | --- | --- | --- |
| 0.10 | 0.9296 | 0.8910 | 0.9486 |
| 0.30 | 0.9343 | 0.8863 | 0.9579 |
| 0.70 | 0.9405 | 0.8768 | 0.9720 |

Full rule at threshold 0.10 (model + catalog lexicon + wake prefixes): on the
same corpus command recall rises to 0.962 with chatter dropped 0.918, measured
with a model trained on that corpus, so treat it as an upper bound -- the
out-of-fold estimate is the 0.89 / 0.95 row above. The remaining misses are
sentences the ASR destroyed beyond recognition (回不少 / 我走 / 因此道听者);
they need the acoustic model, not this one.

## Limits

- Bag-of-grams generalises by vocabulary overlap. Out-of-corpus chatter built
  from words that also appear in addressed sentences scores high (今天天气不错 →
  1.0, 你好 → 1.0), while chatter sharing little vocabulary scores low (你是谁,
  现在几点了, 把窗户关一下 → 0.0). That is acceptable for a veto whose first
  stage is shadow, but it is the reason the next model should be retrained on
  the shadow log: real chatter *with* the robot's vocabulary is the negative set
  the corpus does not have enough of.
- Scores are bimodal (p50 = 0.000, p75 = 1.000). Confidence is useful for
  *which* sentences to label, not as a second threshold.
- Leave-one-skill-out recall drops to 0.51 for 抓取 and 0.45 for 照镜子: the model
  generalises by word overlap. That is why the catalog lexicon exists, and why
  the gate is a veto — a new skill must add its aliases (the emitter does).
- Adding one LLM call to re-check the ambiguous band (3 % of sentences) bought
  +1.5 pp accuracy for 424 ms per call. Not worth it on this board.
