# Episode characterization: separate follow-up work

The core detector returns inclusive observation boundaries and sufficient local
statistics for downstream analysis. Physical descriptors and episode-family
construction should consume those frozen boundaries in a separate module; they
must not change detection scores, calibration, or interval selection.

Reusable components identified in the research implementation are:

- Joint signal magnitude, episode RMS, and peak magnitude, calculated on the
  original response vectors inside each selected core.
- Temporal concentration (TC), as defined in the finalized manuscript: normalize
  each observation's joint magnitude to sum to one, then calculate
  `(n*sum(w.^2)-1)/(n-1)`. TC is distinct from a turning count. A future API must
  explicitly define the zero-magnitude and single-observation cases.
- Common-amplitude episode normalization (`shape_normalize` in the research
  `IntervalEpisodes` module), preserving signs, offsets, and channel composition.
- Variable-length multivariate DTW, retaining the alignment path and separate
  path-normalized shape and raw-amplitude discrepancies (`normalized_dtw` and
  `pairwise_dtw`).
- Complete-linkage grouping, medoid extraction, recurrence checks, and assignment
  of new episodes to frozen families (`complete_link_membership`,
  `recurring_families`, and the downstream family helpers).
- Exposure-based episode frequency, occupied time, and mean positive score excess,
  using caller-supplied exposure units and explicit empty-trip conventions.

Turning counts could be considered separately once their signal definition and
boundary convention are specified; no verified general turning-count function was
identified as part of the finalized interval detector. TC should not be renamed
or interpreted as a turning count.

Before extracting DTW code, specify alignment tie rules and acceptance semantics
and add synthetic tests. Dataset-specific family labels, physical thresholds,
classification code, fitted dictionaries, and research outputs do not belong in
this module.
