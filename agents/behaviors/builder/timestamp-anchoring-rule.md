# behavior:builder:timestamp-anchoring-rule: Temporal Provenance & Timestamp Anchoring

**Category**: Governance Helix  
**Role**: role-builder  
**Status**: Enforced  

## Rule

1. **Mandatory Timestamps**: All distilled claims, key takeaways, and procedural steps extracted from video or audio sources must carry a temporal anchor.
2. **Anchor Format**: Anchors must link to the original video with seconds offset: `[MM:SS](https://youtu.be/...&t=...s)`.
3. **No Unverifiable Citations**: Any assertion not grounded in a timestamped speech segment or visual frame is classified as speculative hallucination and rejected by Tier-A validation.
4. **Anchors Resolve to Real Utterances**: Every anchor must resolve to the start of a segment that exists in the transcript, within a 3-second snap tolerance. Anchors are validated at emit time by `AnchorGate` (`proposals/emitter.py`). An anchor that is unparsable or that names a time with no nearby segment is **dropped**; the claim is then rendered with no evidence link. A missing link is acceptable, a link that points at the wrong moment is not.
5. **No Synthetic Segment Boundaries**: When a source publishes no chapters, the pipeline may batch the transcript for context-window reasons, but must not present those batch boundaries as topic boundaries, title them, or use them as anchors. Mechanical batches carry `boundary_source: mechanical` and `title: null`. Real semantic segmentation is a separate capability; until it runs, section titles in a digest are LLM-proposed and the digest must say so.
6. **Transcript Provenance Is Part of the Key**: A cached transcript is only valid for the source and engine that produced it. Caches are keyed on video id plus source, engine, and language (`acquisition/transcript_cache.py`), and emitted artifacts state which engine produced the timestamps they cite.

## Rationale

Timestamps are the only thing that makes a distilled claim checkable by a human reviewer in
bounded time. An anchor that looks precise but points at an arbitrary moment is worse than no
anchor: it spends the reviewer's trust and their time, and it silently converts a review into a
rubber stamp.
