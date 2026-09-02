# skill:builder:chapter-alignment-anchoring: Chapter Alignment & Timestamp Anchoring

**Category**: Structural Alignment  
**Role**: role-builder (MediaLoom Engine)  
**Status**: Production  
**Last Updated**: 2026-09-02  

## Purpose

Map transcribed speech segments onto native video chapter markers and generate clickable, hyperlinked timestamp URLs (`https://youtu.be/...&t=...s`) for every section and claim.

## Preconditions

- Transcribed segments with start/end seconds.
- Native video chapters list (or auto-synthesized 5-minute intervals).

## Procedure

1. If native chapters are missing, generate synthetic chapters partitioned at 300-second (5-minute) boundaries.
2. Group segments whose `start` or `end` falls within each chapter's `[start_time, end_time)`.
3. Concatenate text into cohesive chapter transcripts.
4. Construct hyperlink anchor URL:
   - Form: `{video_url}&t={int(start_time)}s`
5. Emit structured chapter blocks for LLM multi-perspective distillation.

## Verification

- Every chapter block contains non-empty text, a human-readable timestamp string (`HH:MM:SS`), and a verified anchor URL.
