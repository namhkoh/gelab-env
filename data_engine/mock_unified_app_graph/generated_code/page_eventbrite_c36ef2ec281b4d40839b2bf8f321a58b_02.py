# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_02
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4.png
# step_index: 2/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page.
# Uses provided 'canvas' (1440x2960) and 'draw' (ImageDraw).

# Colors
status_color = (189, 189, 189)        # top status bar gray
divider_color = (159, 146, 170)       # subtle purple/gray divider
card_bg = (245, 247, 255)             # pale blue card behind chips
card_outline = (235, 232, 238)        # very light outline
section_divider = (238, 235, 240)     # faint section separators
bottom_banner = (250, 250, 252)       # near-white content background

w, h = canvas.size

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_color)

# Toolbar area (beneath status) - keep white but draw a subtle top shadow to separate
toolbar_top = status_h
toolbar_bottom = 180
# light translucent shadow line under status to indicate toolbar separation
draw.rectangle((0, toolbar_top, w, toolbar_bottom), fill=(255,255,255))
draw.line((0, toolbar_bottom-1, w, toolbar_bottom-1), fill=(230,230,230), width=1)

# Search / header divider
# The detected "Find events in..." text block occupies y ~264..(264+129)=393.
# Draw a thin divider a little below that to match screenshot structure.
search_divider_y = 400
draw.line((48, search_divider_y, w-48, search_divider_y), fill=divider_color, width=2)

# Card behind "Nearby" / "Online events" chips
chips_card_top = 450
chips_card_bottom = 600
chips_card_left = 40
chips_card_right = w - 40
draw.rounded_rectangle(
    (chips_card_left, chips_card_top, chips_card_right, chips_card_bottom),
    radius=20,
    fill=card_bg,
    outline=card_outline,
    width=1
)

# Subtle divider below chips card separating from browsing section
browse_divider_y = chips_card_bottom + 24
draw.line((48, browse_divider_y, w-48, browse_divider_y), fill=section_divider, width=1)

# Light background block for the browsing section header area
browse_block_top = browse_divider_y + 18
browse_block_bottom = browse_block_top + 220
draw.rectangle((40, browse_block_top, w-40, browse_block_bottom), fill=bottom_banner)

# Very subtle horizontal rule under the browsing header/title
draw.line((48, browse_block_bottom, w-48, browse_block_bottom), fill=section_divider, width=1)

# Large content area background (remaining page)
content_top = browse_block_bottom + 20
draw.rectangle((0, content_top, w, h), fill=(255,255,255))

# Optional faint vertical margins to frame content area (left/right)
margin_x = 40
draw.line((margin_x, toolbar_bottom, margin_x, h), fill=section_divider, width=1)
draw.line((w - margin_x, toolbar_bottom, w - margin_x, h), fill=section_divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 59, 60)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/02_icon_5.12.png
try:
    _c2 = get_crop(2, 58, 62)
    canvas.paste(_c2, (181, 2), _c2)
except Exception:
    pass
layout["5.12"] = [181, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/03_icon_5.12.png
try:
    _c3 = get_crop(3, 57, 62)
    canvas.paste(_c3, (116, 2), _c3)
except Exception:
    pass
layout["5.12"] = [116, 2, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/04_icon_5.12.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["5.12"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 59)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/10_icon_5.12.png
try:
    _c10 = get_crop(10, 97, 64)
    canvas.paste(_c10, (14, 1), _c10)
except Exception:
    pass
layout["5.12"] = [14, 1, 111, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_02_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-4/17_text_Chicago.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 816, 1440, 954]
