# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_02
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-4.png
# step_index: 2/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and UI structure for the provided canvas using PIL draw object.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Overall background (slightly off-white to match the screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 251))

# Status bar at the very top (neutral gray)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(170, 170, 170))
# Thin darker bottom edge for status bar
draw.line([(0, status_h), (1440, status_h)], fill=(150, 150, 150), width=2)

# Header area divider line under the "Find events in..." header block.
# The detected header text block top is at y=264 with height 129 (so bottom y = 393).
header_text_top = 264
header_text_h = 129
header_div_y = header_text_top + header_text_h  # 393
# Draw a subtle divider across the content width with side padding
draw.line([(48, header_div_y), (1440 - 48, header_div_y)], fill=(200, 196, 204), width=2)

# Two option cards ("Nearby" and "Online events") - draw soft rounded rectangles as backgrounds
# Left card (behind the 'Nearby' option area)
left_card_x = 40
left_card_y = 445
left_card_w = 435  # slightly larger than detected content width to make a pill background
left_card_h = 144
draw.rounded_rectangle(
    [(left_card_x, left_card_y), (left_card_x + left_card_w, left_card_y + left_card_h)],
    radius=24,
    fill=(236, 249, 255),
    outline=None
)
# Light inner highlight on left card (very subtle)
draw.line(
    [(left_card_x + 8, left_card_y + 8), (left_card_x + left_card_w - 8, left_card_y + 8)],
    fill=(245, 252, 255),
    width=2
)

# Right card (behind the 'Online events' option area)
right_card_x = 510
right_card_y = 445
right_card_w = 472
right_card_h = 144
draw.rounded_rectangle(
    [(right_card_x, right_card_y), (right_card_x + right_card_w, right_card_y + right_card_h)],
    radius=24,
    fill=(241, 246, 255),
    outline=None
)
draw.line(
    [(right_card_x + 8, right_card_y + 8), (right_card_x + right_card_w - 8, right_card_y + 8)],
    fill=(249, 251, 255),
    width=2
)

# Subtle shadow under both cards to give separation from background
shadow_color = (235, 235, 235)
draw.rectangle(
    [(left_card_x + 6, left_card_y + left_card_h + 6), (left_card_x + left_card_w - 6, left_card_y + left_card_h + 9)],
    fill=shadow_color
)
draw.rectangle(
    [(right_card_x + 6, right_card_y + right_card_h + 6), (right_card_x + right_card_w - 6, right_card_y + right_card_h + 9)],
    fill=shadow_color
)

# Thin separator line between the options area and the "Browsing in" section
sep_y = right_card_y + right_card_h + 30  # a bit of spacing below the option cards
draw.line([(48, sep_y), (1440 - 48, sep_y)], fill=(240, 239, 242), width=1)

# Large subtle area background for the "Browsing in / San Francisco" block.
# The detected "San Francisco" block covers the full width; draw a subtle left accent column
# to visually separate the section without duplicating text or icons.
accent_x = 48
accent_y = 760
accent_w = 8
accent_h = 220
draw.rectangle([(accent_x, accent_y), (accent_x + accent_w, accent_y + accent_h)], fill=(244, 239, 255))

# Another faint horizontal divider below the browsing header area
browse_block_top = 742  # per detected "Browsing in" y
browse_div_y = browse_block_top + 220
draw.line([(48, browse_div_y), (1440 - 48, browse_div_y)], fill=(245, 245, 247), width=1)

# Footer subtle baseline near the bottom of the visible area to anchor the page
footer_y = 2800
draw.line([(0, footer_y), (1440, footer_y)], fill=(245, 245, 247), width=1)

# Done: structural/background elements painted.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/02_icon_7.28.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.28"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/03_icon_7.28.png
try:
    _c3 = get_crop(3, 59, 64)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["7.28"] = [115, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/04_icon_7.28.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.28"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 64)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 43, 63)
    canvas.paste(_c7, (1270, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 59)
    canvas.paste(_c8, (1322, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/09_icon_7.28.png
try:
    _c9 = get_crop(9, 92, 64)
    canvas.paste(_c9, (15, 1), _c9)
except Exception:
    pass
layout["7.28"] = [15, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/17_text_San_Francisco.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["San_Francisco"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_02_2024_4_23_19_27_45f56b06f31541079045047b6d542613-4/18_text_California.png
try:
    _c18 = get_crop(18, 188, 50)
    canvas.paste(_c18, (42, 902), _c18)
except Exception:
    pass
layout["California"] = [42, 902, 230, 952]
