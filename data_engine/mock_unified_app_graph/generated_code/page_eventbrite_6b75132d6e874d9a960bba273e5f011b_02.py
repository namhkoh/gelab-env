# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_02
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4.png
# step_index: 2/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint the page background and structural UI elements for the mobile UI (1440x2960)
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw), and fonts: font_sm, font_md, font_lg, font_xl

# Background base (canvas starts as white, but fill to ensure consistent color)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# subtle top edge highlight and bottom separator for status bar
draw.line([(0, 0), (1440, 0)], fill="#EDEDED", width=1)
draw.line([(0, status_h - 1), (1440, status_h - 1)], fill="#B7B7B7", width=1)

# App header / toolbar area below status bar
toolbar_y0 = status_h
toolbar_y1 = 156
draw.rectangle([(0, toolbar_y0), (1440, toolbar_y1)], fill="#FFFFFF")
# faint shadow under toolbar
draw.line([(0, toolbar_y1), (1440, toolbar_y1)], fill="#E9E6EF", width=2)
draw.line([(0, toolbar_y1 + 2), (1440, toolbar_y1 + 2)], fill="#F6F4F7", width=1)

# A subtle horizontal divider separating header from content (placed in a safe band)
# Pick Y that does not overlap detected text crops (safe area around y=440)
divider_y = 440
draw.line([(48, divider_y), (1440 - 48, divider_y)], fill="#E6E1E8", width=1)

# Light background band for the main content area (lower half) to indicate scannable region
band_y0 = 1000
band_y1 = 1600
draw.rectangle([(0, band_y0), (1440, band_y1)], fill="#FBFBFD")

# Card-style rounded rectangles inside the content band (layout placeholders, do not overlap detected elements)
card_margin_x = 48
card_w = 1344
card_h = 220
card_gap = 36

# First card
c1_x0 = card_margin_x
c1_y0 = band_y0 + 40
c1_x1 = c1_x0 + card_w
c1_y1 = c1_y0 + card_h
draw.rounded_rectangle([(c1_x0, c1_y0), (c1_x1, c1_y1)], radius=20, fill="#FFFFFF", outline="#EFEAF2", width=1)

# Second card below
c2_x0 = card_margin_x
c2_y0 = c1_y1 + card_gap
c2_x1 = c2_x0 + card_w
c2_y1 = c2_y0 + card_h
draw.rounded_rectangle([(c2_x0, c2_y0), (c2_x1, c2_y1)], radius=20, fill="#FFFFFF", outline="#EFEAF2", width=1)

# Small separator lines to define sections further down (safe areas)
draw.line([(card_margin_x, c2_y1 + 40), (1440 - card_margin_x, c2_y1 + 40)], fill="#F0EDF2", width=1)
draw.line([(card_margin_x, c2_y1 + 100), (1440 - card_margin_x, c2_y1 + 100)], fill="#F0EDF2", width=1)

# Footer subtle area near bottom to balance composition
footer_y0 = 2720
footer_y1 = 2960
draw.rectangle([(0, footer_y0), (1440, footer_y1)], fill="#FFFFFF")
draw.line([(0, footer_y0), (1440, footer_y0)], fill="#F2F0F4", width=1)

# Decorative subtle vertical rule on the left to imply content column (does not overlap detected text blocks)
draw.line([(40, toolbar_y1 + 20), (40, footer_y0 - 40)], fill="#F3F1F5", width=2)

# End of structural drawing - icons and text will be pasted on top by the pipeline.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 58, 60)
    canvas.paste(_c1, (311, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [311, 4, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/02_icon_8.11.png
try:
    _c2 = get_crop(2, 60, 65)
    canvas.paste(_c2, (113, 1), _c2)
except Exception:
    pass
layout["8.11"] = [113, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/03_icon_8.11.png
try:
    _c3 = get_crop(3, 59, 62)
    canvas.paste(_c3, (180, 2), _c3)
except Exception:
    pass
layout["8.11"] = [180, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/04_icon_8.11.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["8.11"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 59)
    canvas.paste(_c6, (1323, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/10_icon_8.11.png
try:
    _c10 = get_crop(10, 91, 64)
    canvas.paste(_c10, (14, 1), _c10)
except Exception:
    pass
layout["8.11"] = [14, 1, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_02_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-4/17_text_Chicago.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 816, 1440, 954]
