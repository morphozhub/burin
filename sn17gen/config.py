"""Committed, replay-safe knobs (part of the admitted source). All off = the base pipeline."""

# First final refinement walks the photo part by part (counts of repeated parts, the recognisable
# structure) before changing anything; the second one is the plain four-view refinement.
STRUCTURE_REFINE = False

# Light-base metal shade table appended to the craft rules (the judge's white neutral render darkens
# metals; "dark/black instead of silver" is its most frequent complaint).
METAL_TABLE = False

# Rules for glass, gemstones and liquids as tinted solids with visible rims and facets.
GLASS_RULES = False

# After the Astra panel, the two best chains duel on the judge's four single front views of the
# final (tilted) module, both presentation orders, the judge's own Stage-1 wording; the duel winner
# replaces the panel pick unless the mean gap is within the draw band.
S1_PICK = False

# One text call on the photo before generation: notes (object, visible side, proportions, parts with
# counts and colours, text) and a short checklist of what a judge would verify; attached to every
# generation, refinement and the panel.
CHECKLIST = False

# Reasoning effort "high" on the first (draft) call of every chain instead of "medium": the draft
# fixes the skeleton (part counts, proportions); refinements rarely restructure it.
DRAFT_HIGH = False

# Number of independent generation chains ranked by the panel (base 3). More chains = more
# candidates for the judge-style selection; ids stay below the panel block for up to 8 chains.
CHAINS = 3

# Model calls started at once inside one prompt (chains run in lockstep); keep at the witness
# capture-slot count so battle runs never queue captures behind their own deadline.
PARALLEL_CALLS = 3

# Number of final refinements of the picked module on the judge's four-camera sheet (base 2).
# Rabbit's round-1 chain winner ran 3 and beat its own sibling with 2 by +0.117; ids 950 + k.
FINAL_REFINES = 2

# Round-1 chain winner's selection: instead of the Astra panel, every pair of finalists duels on the
# judge's four single front views in both orders (Sol low, 8 calls per pair, penalties summed); with more
# than three finalists the panel first narrows them to three (24 duel calls, ids 910-933).
TOURNAMENT_PICK = False

# Model that judges the tournament duels: None = the generation model (Sol, the chain winner's choice),
# "gpt-6-astra" = the stronger vision model (our picker calibration: 40/40 on clear pairs).
TOURNAMENT_MODEL = None

# After every final refinement, duel the module before and after it on the judge's four views in both
# orders (8 calls, ids 860 + 8k) and keep the better one, so a refinement can never make the result worse.
GUARDED_REFINES = False

# Before every final refinement, a judge-style critic (the tournament's model and effort) rates the current
# module on each of the judge's four single front views and names the visible differences from the photo;
# its notes are appended to the refinement request (4 calls per refinement, ids 840 + 4k).
JUDGE_NOTES = False

# Guard only the LAST final refinement (one before/after duel on the judge's four views, both orders, 8 calls,
# ids 860 + 8k): the cheap form of GUARDED_REFINES, which never lost on the stand but needed 60 calls.
GUARD_LAST = False

# Detail kit: ready-made snippets for the features the judge most often reports missing (printed text and logos as
# pixel-font DataTextures, ridges/fluting, serrations, stitching, stripes, knurled or brushed surfaces), appended
# to the craft rules of every generation and refinement call.
DETAIL_KIT = False

# With more finalists than the tournament takes, narrow them by single-view judge scores (the tournament's model
# and effort, 4 calls per candidate, ids 800 + 4i) instead of the Astra panel; 0 = off, N = keep the best N.
SCORE_NARROW = 0

# Proportion note: the object's measured width:height in the photo versus the module's near-frontal render,
# appended to every final refinement when they differ by more than 15 percent (no model calls).
PROPORTION_NOTE = False

# Refinements inside every chain before the selection (base 2: generate, then refine the latest valid module
# on its four-camera sheet twice). More chain refinements polish every candidate before the tournament picks one;
# ids stay inside the chain's block (100 per chain).
CHAIN_REFINES = 2

# With four finalists, pick by a knockout bracket instead of a round-robin: two semifinals (chains 1-2, 3-4) and a
# final, each a both-order duel on the judge's four single front views (3 duels, 24 calls, ids 910-933: the price of
# the round-robin of three). With more than four chains the panel narrows them to four first.
KNOCKOUT = False

# Reasoning effort of the tournament judge (also the guard, the score narrowing and the critic): "low" is the
# round-1 chain winner's choice; "medium" spends credits, not calls, on a steadier verdict.
TOURNAMENT_EFFORT = "low"

# Side of each camera view on the four-view sheets that every refinement sees (base 512, the sheet is 2x2 views).
SHEET_PX = 512
