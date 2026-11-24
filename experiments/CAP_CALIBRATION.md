# Output-length cap for EXP-006 (calibrated on training transcripts only)

cap(clip) = ceil(rate × clip seconds) + 4 tokens, rate = 99th percentile of tokens per second (microsoft/paza-whisper-large-v3-turbo tokenizer).

| lang | source | utterances | median tok/s | rate (cap) | cap for 4 s | cap for 10 s |
|---|---|--:|--:|--:|--:|--:|
| kik | badrex/anv_data_ke_kikuyu_scripted train, 6 shards | 18514 | 6.97 | 11.79 | 52 | 122 |
| swh | google/fleurs sw_ke train.tsv | 3070 | 3.37 | 5.40 | 26 | 59 |
| eng | google/fleurs en_us train.tsv | 2602 | 2.69 | 3.90 | 20 | 43 |
