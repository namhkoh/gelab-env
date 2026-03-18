# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_12
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14.png
# step_index: 12/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page
# Assumes `canvas` (1440x2960 PIL Image) and `draw` (ImageDraw.Draw) are already provided.

# Colors
BG_WHITE = (255, 255, 255)
STATUS_BAR_GRAY = (189, 189, 189)       # top status bar
HEADER_DIVIDER = (239, 236, 245)        # subtle divider under header
CARD_BG = (250, 248, 252)               # very light card background
CARD_BORDER = (224, 219, 231)           # card border
SECTION_SEPARATOR = (232, 229, 238)     # inner separators
SHADOW = (230, 226, 235)

W = 1440
H = 2960

# Fill overall background (canvas may already be white, but fill explicitly)
draw.rectangle([(0, 0), (W, H)], fill=BG_WHITE)

# Status bar area (top)
STATUS_BAR_HEIGHT = 72
draw.rectangle([(0, 0), (W, STATUS_BAR_HEIGHT)], fill=STATUS_BAR_GRAY)

# Header region (title area). Keep it visually distinct with a subtle bottom divider.
HEADER_TOP = STATUS_BAR_HEIGHT
HEADER_BOTTOM = 168
draw.rectangle([(0, HEADER_TOP), (W, HEADER_BOTTOM)], fill=BG_WHITE)
# bottom divider line across the content width (leave small margins)
draw.line([(48, HEADER_BOTTOM), (W - 48, HEADER_BOTTOM)], fill=HEADER_DIVIDER, width=2)

# Date selection card (rounded container behind the Start/End rows)
CARD_LEFT = 36
CARD_RIGHT = W - 36
CARD_TOP = 220
CARD_BOTTOM = 640
CARD_RADIUS = 20
draw.rounded_rectangle(
    [(CARD_LEFT, CARD_TOP), (CARD_RIGHT, CARD_BOTTOM)],
    radius=CARD_RADIUS,
    fill=CARD_BG,
    outline=CARD_BORDER,
    width=2
)

# Sub-section separator inside the date card to separate Start and End areas
# Place it roughly between the two rows (matches visual spacing, but not drawing text)
SEPARATOR_Y = 420
draw.line([(CARD_LEFT + 20, SEPARATOR_Y), (CARD_RIGHT - 20, SEPARATOR_Y)], fill=SECTION_SEPARATOR, width=1)

# Soft shadow under the card to lift it slightly
SHADOW_HEIGHT = 6
shadow_top = CARD_BOTTOM + 4
draw.rectangle(
    [(CARD_LEFT + 6, shadow_top), (CARD_RIGHT - 6, shadow_top + SHADOW_HEIGHT)],
    fill=SHADOW
)

# Large empty content area remains white — draw a faint horizontal guide near the bottom
# to separate content from the bottom action area (do not draw the button itself).
BOTTOM_GUIDE_Y = 2720
draw.line([(48, BOTTOM_GUIDE_Y), (W - 48, BOTTOM_GUIDE_Y)], fill=SECTION_SEPARATOR, width=2)

# Bottom safe area background (very subtle, behind the area where a control will be pasted).
# This is a structural background band only; it deliberately stays behind the button area.
SAFE_BAND_TOP = BOTTOM_GUIDE_Y + 8
SAFE_BAND_BOTTOM = H
draw.rectangle([(0, SAFE_BAND_TOP), (W, SAFE_BAND_BOTTOM)], fill=BG_WHITE)

# Add a faint rounded container outline to hint the button area (outline only, no fill),
# but do not fill or draw any inner button content (the actual button will be pasted later).
BUTTON_HINT_LEFT = 40
BUTTON_HINT_RIGHT = W - 40
BUTTON_HINT_TOP = 2748  # sits slightly above the detected button area to avoid duplication
BUTTON_HINT_BOTTOM = 2936
BUTTON_HINT_RADIUS = 12
draw.rounded_rectangle(
    [(BUTTON_HINT_LEFT, BUTTON_HINT_TOP), (BUTTON_HINT_RIGHT, BUTTON_HINT_BOTTOM)],
    radius=BUTTON_HINT_RADIUS,
    outline=CARD_BORDER,
    width=2,
    fill=None
)

# Subtle vertical padding markers (visual structure guides) left side
# (thin vertical line to indicate content column — decorative structure only)
draw.line([(48, HEADER_BOTTOM + 20), (48, BUTTON_HINT_TOP - 20)], fill=SECTION_SEPARATOR, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/01_icon_7.19.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (182, 2), _c1)
except Exception:
    pass
layout["7.19"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 48, 70)
    canvas.paste(_c2, (1155, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1155, 0, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 60)
    canvas.paste(_c3, (310, 4), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 4, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/04_icon_7.19.png
try:
    _c4 = get_crop(4, 56, 64)
    canvas.paste(_c4, (116, 2), _c4)
except Exception:
    pass
layout["7.19"] = [116, 2, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (249, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 5, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/06_icon_7.19.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (12, 72), _c6)
except Exception:
    pass
layout["7.19"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 96, 70)
    canvas.paste(_c7, (1211, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1211, 0, 1307, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 66)
    canvas.paste(_c8, (1325, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/09_icon_What_date.png
try:
    _c9 = get_crop(9, 318, 72)
    canvas.paste(_c9, (558, 112), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 112, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/10_icon_7.19.png
try:
    _c10 = get_crop(10, 90, 63)
    canvas.paste(_c10, (17, 2), _c10)
except Exception:
    pass
layout["7.19"] = [17, 2, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 62)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 580, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/13_text_End_Date.png
try:
    _c13 = get_crop(13, 580, 144)
    canvas.paste(_c13, (48, 313), _c13)
except Exception:
    pass
layout["End_Date"] = [48, 313, 628, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_12_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-14/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
