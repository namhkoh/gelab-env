# GE-Lab V1: Realistic GUI Environment Extension

## 1. Problem Statement

The current GE-Lab environment uses a simplified tree-based navigation model where:
- Each page contains icons that link to other pages
- Only CLICK actions are supported
- Icons are abstract/random with no semantic meaning
- Navigation is purely tree-based (parent-child relationships)

Real device interactions (as evidenced by the Sequence data) involve:
- Multiple action types: CLICK, TEXT, SCROLL, KEY_HOME, COMPLETE
- Complex UI components: search bars, text inputs, lists, buttons, tabs
- Multi-app workflows: home screen -> browser -> shopping app
- Contextual state: search history, cart contents, tab management

## 2. Goals for V1

Create a more realistic GE-Lab environment that:
1. Supports expanded action space (CLICK, TEXT, SCROLL, KEY_HOME)
2. Renders realistic UI components (search bars, buttons, lists)
3. Maintains tree-based navigation structure for evaluation consistency
4. Enables multi-app simulation within a single environment

## 3. Analysis of Real Trajectory (Sequence/2451545728360121.json)

### 3.1 Action Space Breakdown

| Action | Count | Description | Parameters |
|--------|-------|-------------|------------|
| CLICK | 12 | Tap on UI element | coordinates (x, y) |
| TEXT | 2 | Type text input | string content |
| SCROLL | 4 | Swipe gesture | start/end coords + trajectory |
| KEY_HOME | 1 | System home button | none |
| COMPLETE | 1 | Task completion | none |

### 3.2 UI Component Types Required

1. **Home Screen**
   - App icon grid (4-5 columns)
   - App labels
   - Navigation bar at bottom

2. **Browser (Chrome-like)**
   - Address/search bar (typable)
   - Tab counter with overview button
   - New tab button
   - Content area (scrollable webview)
   - Bottom navigation bar

3. **Search Interface**
   - Search input field
   - Search suggestions/history dropdown
   - Keyboard (virtual, for TEXT actions)
   - Search/enter button

4. **Search Results Page**
   - Result items with titles
   - Snippet text
   - Clickable links
   - Scrollable list

5. **Article/Content Page**
   - Title/header
   - Body text (scrollable)
   - Embedded product cards
   - External links

6. **Shopping App (Amazon-like)**
   - Search bar with clear button
   - Product grid/list view
   - Product detail page
   - Price, ratings, images
   - Add to cart button
   - Cart icon with count

### 3.3 State Transitions

```
HOME_SCREEN
    |-- [CLICK app_icon] --> APP_MAIN
    |-- [KEY_HOME] --> HOME_SCREEN

BROWSER_MAIN
    |-- [CLICK search_bar] --> BROWSER_SEARCH_INPUT
    |-- [CLICK tab_button] --> BROWSER_TAB_VIEW
    |-- [SCROLL] --> BROWSER_MAIN (scrolled)
    |-- [KEY_HOME] --> HOME_SCREEN

BROWSER_SEARCH_INPUT
    |-- [TEXT query] --> BROWSER_SEARCH_INPUT (text entered)
    |-- [CLICK search_btn] --> BROWSER_RESULTS

BROWSER_RESULTS
    |-- [CLICK result] --> BROWSER_ARTICLE
    |-- [SCROLL] --> BROWSER_RESULTS (scrolled)

SHOPPING_APP
    |-- [CLICK search_bar] --> SHOPPING_SEARCH
    |-- [CLICK product] --> PRODUCT_DETAIL
    
PRODUCT_DETAIL
    |-- [SCROLL] --> PRODUCT_DETAIL (scrolled)
    |-- [CLICK add_to_cart] --> PRODUCT_DETAIL (cart updated)
```

## 4. Proposed Architecture

### 4.1 Environment Structure

```
ui_environment_v1/
├── config.json              # Environment configuration
├── apps/                    # App definitions
│   ├── home/
│   │   ├── manifest.json    # Home screen config
│   │   └── pages/
│   │       └── main.json    # App grid layout
│   ├── browser/
│   │   ├── manifest.json
│   │   └── pages/
│   │       ├── main.json
│   │       ├── search.json
│   │       ├── results.json
│   │       └── article_*.json
│   └── shopping/
│       ├── manifest.json
│       └── pages/
│           ├── main.json
│           ├── search.json
│           ├── results.json
│           └── product_*.json
├── assets/
│   ├── icons/               # App and UI icons
│   ├── templates/           # UI component templates
│   └── fonts/
└── pages/                   # Rendered page images
```

### 4.2 Page Definition Schema

```json
{
  "page_id": "browser_main",
  "app": "browser",
  "type": "main",
  "components": [
    {
      "id": "search_bar",
      "type": "text_input",
      "bbox": [50, 60, 700, 110],
      "placeholder": "Search or type URL",
      "value": "",
      "actions": {
        "click": {"target": "browser_search_input"},
        "text": {"update_value": true}
      }
    },
    {
      "id": "tab_button",
      "type": "button",
      "bbox": [750, 60, 850, 110],
      "icon": "tabs",
      "label": "2",
      "actions": {
        "click": {"target": "browser_tabs"}
      }
    },
    {
      "id": "content_area",
      "type": "scrollable",
      "bbox": [0, 120, 900, 1800],
      "scroll_content": "article_content_1",
      "actions": {
        "scroll": {"update_scroll_position": true}
      }
    }
  ],
  "transitions": {
    "KEY_HOME": "home_main"
  }
}
```

### 4.3 Action Handler

```python
class ActionHandler:
    def execute(self, action_type, params, current_state):
        if action_type == "CLICK":
            component = self.find_component_at(params["x"], params["y"])
            return self.handle_click(component, current_state)
        
        elif action_type == "TEXT":
            active_input = current_state.get("active_input")
            return self.handle_text(active_input, params["text"], current_state)
        
        elif action_type == "SCROLL":
            scrollable = self.find_scrollable_at(params["start"])
            return self.handle_scroll(scrollable, params, current_state)
        
        elif action_type == "KEY_HOME":
            return {"next_page": "home_main", "state": current_state}
```

## 5. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. **Define new page schema**
   - Component types: button, text_input, scrollable, icon, label
   - Action bindings per component
   - State management for inputs

2. **Implement page renderer**
   - Render components to image
   - Support dynamic text in inputs
   - Handle scroll position

3. **Implement action handler**
   - CLICK: coordinate to component mapping
   - TEXT: input field updates
   - SCROLL: scroll position tracking
   - KEY_HOME: return to home screen

### Phase 2: App Templates (Week 2)

1. **Home Screen**
   - 4x5 app icon grid
   - App icons from assets
   - Click transitions to apps

2. **Browser App**
   - Main page with search bar
   - Tab view
   - Article content pages
   - Search results template

3. **Shopping App**
   - Search interface
   - Product listing
   - Product detail page

### Phase 3: Task Generation (Week 3)

1. **Task templates**
   - Research + Purchase (like the example)
   - Single app navigation
   - Multi-step form completion

2. **Trajectory generation**
   - Automatic path finding
   - Action sequence synthesis
   - Ground truth annotation

### Phase 4: Integration (Week 4)

1. **Training data format**
   - Extend existing format for new actions
   - Include TEXT and SCROLL in solutions

2. **Evaluation**
   - Action type accuracy
   - Coordinate accuracy for CLICK
   - Text matching for TEXT
   - Trajectory matching for SCROLL

## 6. Key Design Decisions

### 6.1 Maintaining Tree Structure

To preserve the theoretical foundation of GE-Lab while adding realism:
- Each app is a subtree rooted at its main page
- Home screen is the global root
- Cross-app navigation only through home (or explicit deep links)
- Scrollable content treated as depth extension (not separate pages)

### 6.2 Action Space Extension

| Action | Current GE-Lab | V1 Extension |
|--------|----------------|--------------|
| CLICK | Icon click only | Component click with context |
| TEXT | Not supported | Input field typing |
| SCROLL | Not supported | Vertical scroll with trajectory |
| KEY_HOME | Not supported | System key press |
| COMPLETE | Not supported | Task completion signal |

### 6.3 Reward Functions

Extend existing reward functions:
- `action_type_match`: Exact match of action type
- `coordinate_match`: For CLICK actions (existing bbox check)
- `text_match`: For TEXT actions (exact or fuzzy string match)
- `scroll_direction_match`: For SCROLL (direction + magnitude)

## 7. Component Library

### 7.1 Base Components

```python
COMPONENT_TYPES = {
    "button": {
        "clickable": True,
        "has_icon": True,
        "has_label": True
    },
    "text_input": {
        "clickable": True,
        "typable": True,
        "has_placeholder": True,
        "has_clear_button": True
    },
    "icon": {
        "clickable": True,
        "has_icon": True,
        "has_label": True
    },
    "scrollable": {
        "scrollable": True,
        "has_content": True,
        "scroll_bounds": True
    },
    "label": {
        "clickable": False,
        "has_text": True
    },
    "image": {
        "clickable": True,
        "has_src": True
    },
    "list_item": {
        "clickable": True,
        "has_title": True,
        "has_subtitle": True,
        "has_icon": True
    }
}
```

### 7.2 App-Specific Components

**Browser:**
- AddressBar (text_input + icon buttons)
- TabButton (button with counter)
- SearchResult (list_item with snippet)
- ArticleCard (image + text)

**Shopping:**
- ProductCard (image + price + rating)
- SearchBar (text_input + voice icon)
- CartButton (button with badge)
- AddToCartButton (button with confirmation)

## 8. Data Format Extension

### 8.1 Training Sample

```json
{
  "idx": 0,
  "image": "pages/browser_main.png",
  "problem": "<image>Task: Search for smart light bulbs. History: Null",
  "solution": "Explain: Click on the search bar to enter a query.\tAction: click(start_box='<|box_start|>(400,85)<|box_end|>')",
  "action_type": "CLICK",
  "target_component": "search_bar"
}
```

### 8.2 TEXT Action Sample

```json
{
  "idx": 1,
  "image": "pages/browser_search_input.png",
  "problem": "<image>Task: Search for smart light bulbs. History: step1: click search_bar",
  "solution": "Explain: Type the search query.\tAction: text('smart light bulbs')",
  "action_type": "TEXT",
  "text_content": "smart light bulbs"
}
```

### 8.3 SCROLL Action Sample

```json
{
  "idx": 2,
  "image": "pages/article_page_1.png",
  "problem": "<image>Task: Find product recommendations. History: step1: click search_bar; step2: text('smart light bulbs'); step3: click search",
  "solution": "Explain: Scroll down to see more content.\tAction: scroll(start=(500,800), end=(500,300))",
  "action_type": "SCROLL",
  "scroll_vector": [0, -500]
}
```

## 9. Success Metrics

1. **Action Type Accuracy**: Model predicts correct action type
2. **CLICK Accuracy**: Coordinates within target component bbox
3. **TEXT Accuracy**: Exact or high similarity text match
4. **SCROLL Accuracy**: Correct direction and reasonable magnitude
5. **Task Completion Rate**: End-to-end trajectory success

## 10. Next Steps

1. [ ] Review and finalize this plan
2. [ ] Set up directory structure for V1
3. [ ] Implement base component renderer
4. [ ] Create home screen template
5. [ ] Implement CLICK action handler with component detection
6. [ ] Add TEXT action support
7. [ ] Add SCROLL action support
8. [ ] Create browser app template
9. [ ] Create shopping app template
10. [ ] Generate sample trajectories
11. [ ] Integrate with training pipeline

## 11. Open Questions

1. Should SCROLL be simplified to discrete steps (up/down) or support continuous trajectory?
2. How to handle keyboard rendering for TEXT actions?
3. Should we support long-press and other gesture types?
4. How to balance realism vs. training efficiency?
5. Should app state persist across sessions or reset?

---

**Document Version**: 1.0
**Created**: 2026-02-02
**Branch**: koh-dev/emulator-expansion
