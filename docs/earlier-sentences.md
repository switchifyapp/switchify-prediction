# Earlier sentences

Issue #37 lets generated words and phrases see earlier sentences, not only the
current one. Reranking and the statistical predictor are unchanged.

## Context

`generation_context(before)` keeps all of `before` as typed. A whitespace run
becomes one line break if it contains one, otherwise one space. Curly
apostrophes become `'`, and leading and trailing whitespace and trailing opening
quotes or brackets are dropped, as before. Sentence ends stay as typed, so the
text reads as prose.

## Window

The model sees at most 128 tokens (`CONTEXT_TOKENS`, `policy.context_tokens`),
the end-of-text token included; it saw 64 before. A longer context starts at the
earliest sentence that leaves at most 127 tokens. If a long sentence stands in
the way, the start is a multiple of 32 tokens after that sentence's start, so at
least 96 tokens remain. The start depends only on sentence starts and on the
end, so it stays put while words are added or the oldest text is clipped, until
a whole sentence or 32 tokens must go.

## Reuse

The worker keeps the prepared context for the session. A new request evaluates
only the tokens after the longest prefix it shares with the prepared one. A
request whose tokens equal the prepared ones is a cache hit and evaluates
nothing.

Quantized evaluation rounds slightly differently depending on how positions
are grouped into forward passes: one 26-token pass and a 21 + 5 split differ by
up to 0.6 in a logit. Context is therefore always evaluated in groups of 16
positions from the window start, and a request evaluates again from the start of
the group holding its first changed token, or its last token. An extended state
then holds exactly the values of a fresh one; tests compare them bitwise with a
small random model and with the pinned model. Grouping costs about 1% more for a
fresh 128-token context on the portable worker and 10% with AVX2.

Search clones the prepared state and continues at its length, as before.
`Refiner::cancel()` discards pending work and results but keeps the context, for
requests about the same text. `reset()` clears it.

## Results

Windows 11, AMD Ryzen AI 9 HX 370, four threads, release CLI round trip, 1,000
sampled frozen queries, each with a fresh context, as `generation_evaluate.py`
measures. `--history 512` precedes each query with earlier texts of its fixture
group, up to the desktop's 512-character buffer.

| Portable worker | Median | p95 | Max | Failures | Six-slot top six |
| --- | ---: | ---: | ---: | ---: | ---: |
| Natural context, before | 305 ms | 521 ms | 828 ms | 0 | 72.6% |
| Natural context, after | 306 ms | 532 ms | 813 ms | 0 | 72.6% |
| 512-character history, before | 307 ms | 545 ms | 1,019 ms | 0 | 72.6% |
| 512-character history, after | 1,261 ms | 1,600 ms | 1,838 ms | 3 | 78.9% |

The fixtures are single sentences, so natural context is unchanged. With
earlier sentences, accuracy rises by 6.3 points, but a fresh 128-token context
costs about 1.2 s on the portable worker (about 0.26 s with AVX2). Extending a
prepared one by a word costs 15 to 160 ms. The three failures were reply
timeouts. These are regression measurements on synthetic fixtures whose overlap
with model pretraining is unknown.
