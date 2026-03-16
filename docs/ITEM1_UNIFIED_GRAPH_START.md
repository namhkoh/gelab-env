# Item 1: identical‑state detection + unified graph — getting started guide

This document lays out the steps for implementing identical‑state detection and a unified graph.

---

## Goal (in one line)

- When two screens represent the **same state**, **do not create a new node**; instead, reuse the existing node and **only add edges (actions/icons)**.
- This keeps the graph compact and lets multiple paths through the app/home screen share the same state.

---

## Step 1: Define what “state” means (first and most important)

**Task:** Decide on a criterion for when two pages are the **same state**, and implement a function that computes it.

**Recommended first‑pass criterion**  
- **Look only at content**: the set of **non‑system elements** (icons/widgets) and their positions, ignoring back/home.  
- In other words, if “which actions (icons) exist at which bboxes” is identical, consider the states equivalent.  
- Do **not** include `page_id`, depth, or parent in the state definition; the same layout reached via different paths is still the same state.

**Suggested implementation location**  
- Put utility functions near the top of `data_engine/tree.py` (above the main classes) so they can be reused by both synthetic and trajectory paths.

**Two key functions**

1. **`state_fingerprint_page(page: UIPage) -> str`**  
   - For synthetic mode. `UIPage` has a `layout` (dict: func_desc → `(x1,y1,x2,y2)`) and `elements`.  
   - Ignore back, home, page_title and build a sorted list of **(icon name, bbox tuple)**, then hash or serialize to a normalized string.  
   - Example:  
     `sorted((name, tuple(bbox)) for name, bbox in page.layout.items() if name not in ('back','home','page_title'))` → `json.dumps(..., sort_keys=True)`.

2. **`state_fingerprint_layout(layout: Dict[str, List[int]]) -> str`**  
   - For trajectory mode, where we only have a layout dict (`pages[page_id]["layout"]`).  
   - Same idea: drop system/title keys, then sort `(name, bbox)` pairs and serialize or hash.

**Optional: bbox normalization**  
- If we want to avoid treating tiny pixel‑level shifts as different states, we can grid or normalize bboxes (e.g., to 0–1000) and round before fingerprinting.  
- The first version can simply use raw `(x1, y1, x2, y2)` coordinates.

**Sanity check**  
- In `tree.py`, create two `UIPage` instances with the same layout and verify `state_fingerprint_page` returns the same value.

---

## Step 2: Hook into synthetic generation (optional, can be done later)

**Current behavior:** In `TopologyGenerator.generate()`  
- Each call to `_create_page()` creates a **new `UIPage`**.  
- Nodes are wired up via `hierarchy[parent_id].append(child_id)` and `pages[child_id] = child` without deduplication.

**In theory:**  
- We could check whether a child page matches any existing page’s fingerprint and reuse that `page_id` if a match is found.  
- However, the current synthetic generator tends to assign different icons to every page, so perfectly identical layouts may rarely appear.  
- For this reason, it is fine to **start with Step 1 only** and prioritize trajectory mode (Step 3).

---

## Step 3: Hook into trajectory generation (main application)

**Location:** `data_engine/tree.py`, function **`generate_trajectory_family_environment()`**.

**Current flow:**  
- Spine: for each step, create a new `page_id` and `page_record`, then store `pages[page_id] = page_record`.  
- Branch: allocate `page_{N}` from `page_counter` and store `pages[branch_page_id] = ...`.  
- Even when layouts are identical (same family layout), separate nodes are created for spine and branches.

**Target flow:**  
1. **Spine**
   - Before inserting `page_record`, compute `state_fingerprint_layout(page_record["layout"])`.  
   - Look up the fingerprint in a shared dict `fp2page: Dict[str, str>`.  
   - If found:  
     - Do **not** create a new `page_id` / `page_record`.  
     - Use the existing `page_id` as the page for this step, and only **add or update transitions** from the previous node to this existing node.  
   - If not found:  
     - Create a new `page_id` as today, store it in `pages`, and register `fp2page[fp] = page_id`.

2. **Branch**
   - Branch pages share the canonical layout for the family.  
   - Before creating a new branch node, compute its fingerprint and check whether a spine (or existing branch) node already has this layout.  
   - If a node exists:  
     - Do **not** create a new `branch_page_id`.  
     - Point the canonical page’s outgoing edge’s `target_page` to the existing `page_id`, and if needed, add merge edges in that existing node’s `transitions`.  
   - If not:  
     - Create a new branch node exactly as before.

**Caveats:**  
- The same state can be reached via many paths, so a single node may have **multiple incoming edges**.  
- Outgoing transitions should represent **all actions possible from that state**, aggregated from every occurrence.  
- When appending to `transitions`, we may need a policy for “duplicate actions” (overwrite vs ignore vs keep both).

**Concrete edit points:**  
- Inside `generate_trajectory_family_environment`:  
  - The spine loop that currently does `pages[page_id] = page_record` (around lines 1802–1822).  
  - The branch loop that does `pages[branch_page_id] = { ... }` (around lines 1850–1865).  
- Immediately before each assignment, compute the fingerprint, consult `fp2page`, and either reuse or insert.

---

## Step 4: Save/load compatibility

- `ui_structure.json` / `ui_structure_layer.json` contain **`page_id`** and **`transitions`**.  
- After identical‑state merge, **the number of nodes decreases**, and some `page_id` values will have multiple incoming edges.  
- If the previous assumption was “one page = one image file”, we now just keep **one representative image** for each merged node (first occurrence or a chosen canonical).  
- `save_environment_data` and layer‑building code already operate on `pages` dicts, so as long as merge is done correctly, storage does not need major changes — we only need to ensure each `page_id` maps to exactly one image choice.

---

## Implementation order checklist

| Step | Task | File / location |
|------|------|-----------------|
| 1 | Add **two state fingerprint functions** (for `UIPage` and layout dict) | Utility section at the top of `tree.py` |
| 2 | Add a **simple test or sanity check** that two identical layouts produce the same fingerprint | In `tree.py` or under `tests/` |
| 3 | In **trajectory generation**, reuse existing nodes based on fingerprints for spine/branches | `generate_trajectory_family_environment()` |
| 4 | (Optional) Try identical‑state merge in synthetic generation | `TopologyGenerator.generate()` |
| 5 | (If needed) Adjust save/layer generation to clarify which image represents each merged node | `save_environment_data` and related code |

---

## First commit you can make immediately

- Implement **`state_fingerprint_page(page: UIPage) -> str`**.  
- Implement **`state_fingerprint_layout(layout: Dict[str, List[int]]) -> str`**.  
- Add short docstrings explaining that these are for **Item 1: identical‑state detection**, ignoring back/home/page_title and normalizing as `(name, bbox)` pairs.  
- Add a quick check (in a test or under `if __name__ == "__main__":`) that two pages with the same layout produce identical fingerprints.

Once the notion of “state” is codified this way, Step 3 (trajectory‑side merging) can be implemented on top of it.
