# Baseline quality protocol, frozen before test scoring

The first and only proposed mixture is 3× WorldAlphabets training sentences and
1× unique Taskmaster USER training sentences, using unchanged n-gram weights.
The development gates are fixed in `quality-policy.json`; no test scores were
used to select this mixture. Exact prepared partition hashes are pinned in the
source manifest. Taskmaster uses its official dialogue-level splits; exact
normalized cross-partition sentence matches are removed before evaluation.

The general regression set retains the already-used v1 test partition. It is not
claimed to be newly unseen. The Taskmaster test partition has not been scored at
this checkpoint. Each conversational evaluation set samples up to 100 sentences
per task domain by hash, with 599 unique sentences after cross-domain deduplication.
The resulting benchmark is a task-request proxy, not a representative AAC study.

## Development decision

- `conversation_top5_gain`: 0.2501355013550135
- `general_top5_loss`: 0.011947218259629189
- `selection_savings_gain`: 0.11746972818168111
- `quality_passed`: True
- `latency_target_met`: True

The quality gates pass. The mixture and ranking are frozen for test scoring.
