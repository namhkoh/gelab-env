# Item 1: merge strategy and current status

This document answers three questions and summarizes where we are.

---

## 1. How is the graph stored — do we understand it?

**Yes,** the storage format is understood; it just differs slightly by path.

### Synthetic (`tree.py` — `DynamicTopoEnv`)


| Stage     | What exists                             | Where                                                                                                       |
| --------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| In‑memory | `transition_graph` (networkx `DiGraph`) | Nodes: `page_id`, with `node["page"]` = `UIPage` object. Edges: `(u, v)` with `data["action"]` = icon name. |
| Save      | `save_environment_data()`               | Builds `pages_data` per node as `{ image, depth, layout, transitions }`.                                    |
| Files     | `ui_structure.json`                     | `{"pages": pages_data, "metadata": ...}`                                                                    |
|           | `ui_structure_layer.json`               | Hierarchical structure (parent‑child) derived from `pages_data`.                                            |
|           | `pages/<page_id>.png`                   | One image per node.                                                                                         |


- `layout`: `{ icon_name: { "bbox": [x1,y1,x2,y2], "type": "normal"|"system" } }`
- `transitions`: `[ { "action", "target_page", "icon_bbox" }, ... ]`

### Trajectory (`tree.py` — `generate_trajectory_family_environment`)


| Stage     | What exists                        | Where                                                                                            |
| --------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| In‑memory | `pages` dict, `tree_children` dict | `pages[page_id]` uses the same schema (image, depth, layout, transitions, …). No networkx graph. |
| Save      | At the end of the same function    | Directly dumps `ui_structure.json` and `ui_structure_layer.json`.                                |
| Files     | `pages/<page_id>.png`              | Per‑family page images.                                                                          |


- Spine: creates `page_0`, `page_1`, … for each step.
- Branch: creates `page_{N}` and links it back to the next spine node via a merge edge.

### `sim2real_compose.py`

- `pages_data` is a list `[ { page_id, layout, ... }, ... ]` ordered by trajectory step.
- `build_structure(pages_data, trajectory, ...)` builds `ui_structure["pages"]` and writes `ui_structure.json` / `ui_structure_layer.json` in the same schema.

**Summary:**  

- The “graph” is represented as a **networkx graph** in the synthetic path, and as `**pages` dict + `transitions`** in trajectory/sim2real.  
- **Persistence is always** via `ui_structure.json` (flat pages + transitions) and `ui_structure_layer.json` (hierarchy), with `pages/*.png` for images.  
→ This is enough to say the storage format is understood.

---

## 2. Do we already have an algorithm for “identical icon (image)”?

**No.**  
What we implemented is **“identical state”**, not “identical icon (image)”.

- **What exists:**
  - `state_fingerprint_page` / `state_fingerprint_layout`
  - **Layout‑based criterion**: a page is the same state if the set of (**element name**, **bbox**) pairs matches.
  - That is, we only look at **which elements are where**; we do **not** look at pixels or images.
- **What does not exist:**
  - Any algorithm to decide whether two icon images are visually the same (same app icon, duplicate crop, …).
  - Examples that are **not implemented**: perceptual hashes, embedding similarity, or “this crop and that crop are the same icon”.

So:

- **“Identical state”** = **same layout (name + bbox)** — detected via the current fingerprint functions.
- **“Identical icon (image)”** — **no algorithm yet**.

---

## 3. Is there an algorithm to merge multiple trajectories into one graph?

**No.**  
Currently, it is strictly **one trajectory → one environment**.

- `tree.py`: accepts exactly one `--trajectory_id`, and `generate_trajectory_family_environment(args, output_dir)` builds a single `pages` dict / `ui_structure.json`.
- `sim2real_compose.py`: likewise processes a single `trajectory_id`.

For **merging multiple trajectories into one graph** (e.g., taking several trajectories, building pages for each, merging nodes with the same fingerprint, and combining edges from different trajectories):

- This is **not implemented yet**.  
→ The correct answer today is: **no, there is no multi‑trajectory merge algorithm.**

---

## 4. Then how should we merge — strategy

Below is the **strategy**; once agreed, it can be implemented where indicated.

### 4.1 State definition (already done)

- **Same state** = same set of (**element name**, **bbox**) pairs, excluding back/home/page_title.  
- Detection: compute a string via `state_fingerprint_layout(layout)` (or `state_fingerprint_page` for `UIPage`). If the strings match, they are the same state.

### 4.2 Merging within a single trajectory (what we implemented first)

**Situation:**  

- A single trajectory may revisit the same screen at different steps (e.g., step 2 and step 5 have the same layout).  
- Previously we always created new nodes like `page_2`, `page_5` even when layouts were identical.

**Strategy:**  

1. Maintain a mapping `**fp2page: Dict[str, str]`** from fingerprint to `page_id`.
2. **Right before adding a new page (node):**
  - Compute `fp = state_fingerprint_layout(layout)` from this page’s `layout`.  
  - If `fp in fp2page`:
    - Do **not** create a new `page_id`.  
    - Use the existing `page_id = fp2page[fp]` as “the page for this step”.  
    - **Only add edges**: add a transition from the current parent (or previous node) to this existing `page_id` (e.g., updating the parent’s `transitions` with `target_page = page_id`).
  - Else:
    - Create a new `page_id`, store `pages[page_id] = ...`, and set `fp2page[fp] = page_id`.
3. **Images:**
  - Since multiple steps may map to the same fingerprint, a state can be referenced many times.  
  - A single **representative image per fingerprint** is enough (e.g., from the first step where that state appears).

**Where to hook this in:**  

- In `tree.py`’s `generate_trajectory_family_environment`:  
  - In the spine construction loop (around lines 1789–1822).  
  - In the branch construction loop (around 1832–1865).
- Just before `pages[page_id] = ...`, compute the fingerprint, check `fp2page`, and either reuse or insert.

### 4.3 Merging multiple trajectories into one graph (future work)

**Situation:**  

- We may want a single graph representing multiple trajectories A, B, C for the same app/environment.

**Strategy:**  

1. Use **shared data structures**:
  - One global `pages: Dict[str, ...>`  
  - One global `fp2page: Dict[str, str>`
2. For each trajectory:
  - Build “pages + transitions” as now, but **for each page insert**:  
    - Compute `state_fingerprint_layout(layout)`.  
    - If `fp in fp2page`: reuse `page_id = fp2page[fp]` and only add edges from that trajectory’s parents to this existing node.  
    - Else: create a new `page_id`, store it in `pages`, and register `fp2page[fp] = page_id`.
3. **Roots / layers:**
  - With multiple trajectories there may be multiple roots.  
  - Either define a synthetic “super‑root” or allow multiple roots and reflect that in `ui_structure_layer.json` (policy‑dependent).

**Where to put this:**  

- Right now the entry points accept only a single `trajectory_id`.  
- A multi‑trajectory merge would need a new entry point, e.g. `generate_merged_trajectory_environment(trajectory_ids, ...)`, which internally loops trajectories against the shared `pages` + `fp2page`.  
- That should come **after** the single‑trajectory merge (4.2) is stable.

---

## 5. Status recap


| Question                                                | Answer                                                                                                                                                                                                         |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Do we understand how the graph is stored?               | **Yes.** Synthetic / trajectory / sim2real all end up in the same `ui_structure*.json` + `pages/*.png` format.                                                                                                 |
| Do we have an algorithm for identical icons (images)?   | **No.** The only thing implemented is “identical state” via layout‐based fingerprints.                                                                                                                         |
| Do we have an algorithm to merge multiple trajectories? | **No.** It is still 1 trajectory → 1 environment.                                                                                                                                                              |
| How will we merge?                                      | **Strategy:** (1) Within a single trajectory, use fingerprints to reuse existing nodes and only add edges. (2) Later, add a multi‑trajectory entry point that reuses nodes using a shared `pages` + `fp2page`. |


**Next concrete implementation steps:**  

- **First:** follow 4.2 and implement fingerprint‑based node reuse within `generate_trajectory_family_environment`.  
- **Then (optional):** add the 4.3 multi‑trajectory merge entry point and algorithm.

