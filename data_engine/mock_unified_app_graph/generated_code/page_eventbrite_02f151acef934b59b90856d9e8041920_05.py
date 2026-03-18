# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_05
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7.png
# step_index: 5/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page

# Fill entire background (clean white)
draw.rectangle([(0, 0), canvas.size], fill=(255, 255, 255))

# Status bar area (top) - light grey background to sit behind status icons
status_bar_height = 72
draw.rectangle([(0, 0), (canvas.size[0], status_bar_height)], fill=(200, 200, 200))

# subtle divider under status bar
draw.line([(0, status_bar_height), (canvas.size[0], status_bar_height)], fill=(190, 190, 190), width=2)

# Header/toolbar area (under status bar) - white with subtle bottom divider
header_top = status_bar_height
header_height = 88
header_bottom = header_top + header_height
draw.rectangle([(0, header_top), (canvas.size[0], header_bottom)], fill=(255, 255, 255))
# header bottom divider
draw.line([(24, header_bottom), (canvas.size[0]-24, header_bottom)], fill=(235, 229, 242), width=2)

# Group card background behind the list of time options
# (rounded rectangle spanning the list region so pasted text/icons appear on top)
list_card_top = 200
list_card_bottom = 1280
list_card_margin = 40
draw.rounded_rectangle(
    [(list_card_margin, list_card_top), (canvas.size[0]-list_card_margin, list_card_bottom)],
    radius=24,
    fill=(252, 251, 252),  # very subtle off-white/pale tint
    outline=(240, 236, 244),
    width=2
)

# Draw separators between the list items as thin subtle lines
# Detected text item top positions (do not draw text) used only to place separators visually
item_tops = [234, 414, 594, 774, 954, 1134]
item_height = 144
sep_color = (238, 233, 244)
for y in item_tops:
    sep_y = y + item_height - 6  # a little above the bottom of each item box
    # ensure separator is inside the list card bounds
    if list_card_top < sep_y < list_card_bottom:
        draw.line([(list_card_margin+12, sep_y), (canvas.size[0]-list_card_margin-12, sep_y)], fill=sep_color, width=2)

# Add subtle left edge accent line on the list card (visual boundary, not a UI element)
accent_x = list_card_margin + 8
draw.line([(accent_x, list_card_top + 12), (accent_x, list_card_bottom - 12)], fill=(245, 243, 249), width=2)

# Light bottom shadow under the list card to give a raised card feel
shadow_top = list_card_bottom
shadow_height = 18
for i in range(shadow_height):
    alpha = int(12 * (1 - i / shadow_height))  # fading shadow
    color = (0, 0, 0, alpha)
    # draw as progressively lighter grey lines (canvas is RGB, so approximate shadow with grey)
    grey = 240 - i
    draw.line([(list_card_margin+4, shadow_top + i), (canvas.size[0]-list_card_margin-4, shadow_top + i)], fill=(grey, grey, grey))

# Right-side subtle visual divider near top-right for toolbar affordance (purely structural)
draw.line([(canvas.size[0]-92, header_top+18), (canvas.size[0]-92, header_bottom-18)], fill=(245,245,245), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/00_icon_5.25.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["5.25"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 63, 61)
    canvas.paste(_c1, (309, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [309, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/02_icon_5.25.png
try:
    _c2 = get_crop(2, 57, 64)
    canvas.paste(_c2, (116, 2), _c2)
except Exception:
    pass
layout["5.25"] = [116, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/03_icon_5.25.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["5.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/08_icon_5.25.png
try:
    _c8 = get_crop(8, 91, 62)
    canvas.paste(_c8, (17, 3), _c8)
except Exception:
    pass
layout["5.25"] = [17, 3, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_05_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
