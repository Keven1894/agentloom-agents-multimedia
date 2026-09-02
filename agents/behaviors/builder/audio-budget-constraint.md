# behavior:builder:audio-budget-constraint: Audio Budget & Size Constraints

**Category**: Engine Governance  
**Role**: role-builder  
**Status**: Enforced  

## Rule

1. **Format 140 Preference**: Never download full video streams when ingesting multimedia for text distillation. Always extract format 140 (native m4a) or pure audio stream.
2. **25 MB Hard Ceilings**: Any audio file sent to OpenAI Whisper API must be strictly smaller than 24 MB. When audio approaches 24 MB, the agent must downsample to 16kHz mono or chunk into 15-minute intervals.
3. **Bandwidth & Storage Conservation**: Delete intermediate compressed chunk files after transcription completes; retain only the source audio in `.cache/audio/` and the segment JSON in `.cache/transcripts/`.
