# Identical-state unified graph (Item 1)

This document describes the **identical-state detection and unified graph** behaviour: when the same screen state appears multiple times in a trajectory, we merge into a single page node and attach any additional icons/actions to it.

---

## Current status

| Question | Answer |
|----------|--------|
| How is the graph stored? | **Yes.** Synthetic uses networkx in memory; trajectory/sim2real use a `pages` dict + `transitions`. Persistence is always `ui_structure.json`, `ui_structure_layer.json`, and `pages/*.png`. |
| Identical **icon (image)** algorithm? | **No.** We only have **identical state** (layout: element name + bbox). No pixel/image-based “same icon” logic. |
| Multi-trajectory merge? | **No.** Still one trajectory → one environment. |

**Spine vs effective spine**

- **`spine_page_ids`**: logical step identifiers (one per trajectory step).
- **`effective_spine_page_ids`**: after deduplication, which graph node each step points to. The same node can appear multiple times (e.g. revisiting the same state). So 18 spine steps can yield 17 unique spine nodes.

---

## What is implemented

1. **State** = same set of (element name, bbox) pairs, excluding back/home/page_title.  
   `state_fingerprint_layout(layout)` (and `state_fingerprint_page` for synthetic) returns a canonical string; same string ⇒ same state.

2. **Single-trajectory merge**  
   Before adding a new page, we compute the fingerprint. If it is already in `fp2page`, we reuse that `page_id`, redirect the previous step’s transition to it, and **attach** the new step’s spine transition (and, when merging, union layout elements) into the existing node. We also record which steps contributed in `source_steps`.

3. **Attach additional icons/actions**  
   When the same state is revisited, we do **not** create a new node. We add the newly observed layout entries (if any new names) and the new spine transition to the existing page’s `layout` and `transitions`. One representative image per merged node is kept.

**Implementation details (for reviewers)**

- **Layout union**: In the merge branch, `_build_spine_with_merge` copies layout elements from the merged-away family into the existing page’s `layout` (names not already present; same name with different bbox keeps the first-seen). The newly appended transition’s action name is also ensured to exist in the page’s `layout` so that transitions and layout stay consistent.
- **source_steps**: Created when a new page is added (`source_steps: [family_idx]`). On merge, we append the current `family_idx` to the existing page’s `source_steps` (see `existing_page.setdefault("source_steps", [])` and `src_list.append(family_idx)` in `tree.py`). The saved `ui_structure.json` includes `source_steps` per page.

**Limitation**

- Fingerprinting is **exact-match layout** (name + bbox). It may miss semantically identical states under small spatial or naming differences.

---

## Result summary (18-step episode)

- Episode `7872483543119388`: step 0 and step 5 were forced to the same layout so that step 5 merges into `page_0`.
- **effective_spine_page_ids[5]** = `page_0` (merge applied).
- **Unique spine nodes**: 17. **Unique branch nodes**: 3. **total_pages**: 20 (= 17 + 3).
- `page_0` has two spine transitions: `launcher_button` → `page_1` and `content_region` → `page_6`, and `source_steps: [0, 5]`.

Run `scripts/run_real_complex.py` to reproduce. See `tests/data_engine/test_tree_unified_graph.py` for unit and integration tests.

---

## Fixes applied (review feedback)

- **page_counter collision**: After merge, branch page IDs now use `max(existing page indices) + 1` instead of `len(pages)`, so the first branch page no longer overwrites an existing spine page (e.g. `page_17` in an 18-step trajectory).
- **canvas_size key**: Trajectory metadata now sets both `output_canvas_size` and `canvas_size`; `env_utils.py` reads `canvas_size` for bbox normalization.
- **Layout vs transition consistency**: Saved page `layout` uses the mutated layout (after `_choose_click_target`) so transition action names (e.g. `launcher_button`, `input_field`) exist in the page’s `layout`. On merge, the newly appended transition’s action name is added to the existing page’s `layout` if missing.
