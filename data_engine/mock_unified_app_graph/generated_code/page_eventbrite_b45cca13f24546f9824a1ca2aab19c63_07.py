# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_07
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9.png
# step_index: 7/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 canvas using provided
# `canvas` (PIL Image) and `draw` (PIL ImageDraw). Fonts are available but unused.
w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
status_gray = (189, 189, 191)       # status bar background
status_line = (160, 160, 164)       # subtle bottom line of status bar
header_blue = (43, 82, 255)         # bright accent blue used for underline
divider_gray = (230, 230, 235)      # very light divider for sections
card_border = (220, 220, 225)       # card border / shadow hint
card_fill = (255, 255, 255)         # card fills (keeps overall white)

# Clear canvas to background white
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_gray)
# subtle bottom separator under status bar
draw.line([(0, status_h), (w, status_h)], fill=status_line, width=1)

# Header area (toolbar) below status bar
header_top = status_h
header_bottom = 220
# header background stays white to match app; draw a faint separator shadow
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_white)
# blue underline under header (accent)
underline_y = header_bottom - 4
draw.line([(48, underline_y), (w - 48, underline_y)], fill=header_blue, width=4)

# Very subtle hairline above the underline (gives crisp separation)
draw.line([(48, underline_y - 8), (w - 48, underline_y - 8)], fill=divider_gray, width=1)

# Large empty content area intentionally left blank (for pasted elements)
content_top = header_bottom + 20
# Add a light global divider to visually separate header from content
draw.line([(24, content_top), (w - 24, content_top)], fill=divider_gray, width=1)

# "Nearby" item card background (rounded rectangle)
# Positioned to leave space for pasted icon and text; do not draw icons/text.
card_x1, card_y1 = 30, 440
card_x2, card_y2 = w - 30, 560
card_radius = 16
# subtle border/shadow outline
draw.rounded_rectangle([(card_x1 + 2, card_y1 + 4), (card_x2 + 2, card_y2 + 4)],
                       radius=card_radius, outline=card_border, width=1, fill=None)
# main card (white) - acts as background behind pasted row content
draw.rounded_rectangle([(card_x1, card_y1), (card_x2, card_y2)],
                       radius=card_radius, outline=(245,245,246), width=1, fill=card_fill)

# Additional subtle separators for future sections (do not overlap detected text/icons)
sep1_y = card_y2 + 40
draw.line([(24, sep1_y), (w - 24, sep1_y)], fill=divider_gray, width=1)

# Right-side thin divider near top header's action area (visual cue behind pasted X icon)
# Keep it faint and do not draw actual icon content.
right_divider_x = w - 72
draw.line([(right_divider_x, header_top + 18), (right_divider_x, header_bottom - 18)], fill=divider_gray, width=1)

# Bottom area left intentionally blank (white canvas) for pasted content
# Add a faint bottom margin line to emulate app chrome
bottom_margin_y = h - 24
draw.line([(24, bottom_margin_y), (w - 24, bottom_margin_y)], fill=(245,245,246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/01_icon_7.06.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["7.06"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 62, 61)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/03_icon_7.06.png
try:
    _c3 = get_crop(3, 62, 64)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["7.06"] = [179, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 87, 63)
    canvas.paste(_c4, (1215, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [1215, 1, 1302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 79, 85)
    canvas.paste(_c5, (1314, 291), _c5)
except Exception:
    pass
layout["icon_5"] = [1314, 291, 1393, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/06_icon_7.06.png
try:
    _c6 = get_crop(6, 60, 65)
    canvas.paste(_c6, (115, 1), _c6)
except Exception:
    pass
layout["7.06"] = [115, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 55)
    canvas.paste(_c7, (249, 7), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 7, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 44, 56)
    canvas.paste(_c8, (1324, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1324, 4, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 42, 59)
    canvas.paste(_c9, (1271, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1271, 3, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/10_text_7.06.png
try:
    _c10 = get_crop(10, 91, 45)
    canvas.paste(_c10, (20, 15), _c10)
except Exception:
    pass
layout["7.06"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/11_text_New_York.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_07_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-9/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]
