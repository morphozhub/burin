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
