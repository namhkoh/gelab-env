# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_03
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6.png
# step_index: 3/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas (1440x2960)
# Available variables:
# - canvas: PIL Image (RGB)
# - draw: PIL ImageDraw
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
status_bg = (245, 245, 245)        # light gray status bar
muted_gray = (238, 238, 238)       # subtle divider / card border
divider_gray = (224, 224, 224)     # separators
search_fill = (250, 250, 250)      # search bar background
card_bg = (252, 252, 253)          # very subtle off-white for section cards
nav_shadow = (230, 230, 230)       # nav bar top border shadow

# Fill overall background (canvas is initially white; do it explicitly)
draw.rectangle([0, 0, w, h], fill=bg_white)

# Status bar area at top (~0..64)
status_h = 64
draw.rectangle([0, 0, w, status_h], fill=status_bg)

# Search bar (rounded rect). Positioned to avoid drawing icons/text that will be pasted on top.
# Use a roomy rounded rect that covers the search field area. (Will be underneath icons/text.)
search_left = 32
search_top = 88
search_right = w - 32
search_bottom = 232
search_radius = 22
draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                       radius=search_radius, fill=search_fill, outline=muted_gray, width=1)

# subtle shadow line below search bar (very light)
draw.line([(search_left, search_bottom + 8), (search_right, search_bottom + 8)], fill=divider_gray, width=1)

# Thin divider directly under the search block (spanning content width)
divider_y = search_bottom + 28
draw.line([(24, divider_y), (w - 24, divider_y)], fill=divider_gray, width=1)

# Recent searches / list separators
# Based on detected tops for list items; draw separators at bottoms of each list item.
recent_item_tops = [471, 639, 807, 975, 1143]  # detected top positions
item_height = 168
content_left = 32
content_right = w - 32
for top in recent_item_tops:
    bottom = top + item_height
    # draw subtle separator line across the content area
    draw.line([(content_left, bottom), (content_right, bottom)], fill=divider_gray, width=1)

# Heavier section divider after recent searches (make a bit darker, matches screenshot)
if recent_item_tops:
    last_section_bottom = recent_item_tops[-1] + item_height
    draw.line([(20, last_section_bottom + 8), (w - 20, last_section_bottom + 8)], fill=(220,220,220), width=1)

# Suggestions card background (subtle off-white rounded area behind suggestions)
# Place it below recent searches; keep it tall enough to cover suggestion items
suggestions_top = 1400
suggestions_bottom = 2000
suggestions_left = 24
suggestions_right = w - 24
draw.rounded_rectangle([suggestions_left, suggestions_top, suggestions_right, suggestions_bottom],
                       radius=14, fill=card_bg, outline=None)

# Separators for suggestion items (based on detected suggestion tops)
suggestion_item_tops = [1520, 1688, 1856]
for top in suggestion_item_tops:
    bottom = top + item_height
    draw.line([(content_left, bottom), (content_right, bottom)], fill=divider_gray, width=1)

# Content-area long divider (a faint rule across full width near mid page)
mid_div_y = 1311  # approximate consolidated divider location
draw.line([(16, mid_div_y), (w - 16, mid_div_y)], fill=divider_gray, width=1)

# Bottom navigation bar background and top shadow line
nav_top = 2792
nav_bottom = h
draw.rectangle([0, nav_top, w, nav_bottom], fill=bg_white)
# top shadow / border line for nav bar
draw.line([(0, nav_top), (w, nav_top)], fill=nav_shadow, width=2)
# subtle second line for depth
draw.line([(0, nav_top + 2), (w, nav_top + 2)], fill=(245,245,245), width=1)

# Additional subtle left/right content gutters (visual structure)
gutter_x = 24
draw.line([(gutter_x, status_h + 8), (gutter_x, h - 200)], fill=(250,250,250), width=1)
draw.line([(w - gutter_x, status_h + 8), (w - gutter_x, h - 200)], fill=(250,250,250), width=1)

# Slight top card stroke under the search to indicate raised input
draw.rectangle([search_left + 1, search_bottom + 1, search_right - 1, search_bottom + 2], fill=divider_gray)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/00_icon_4.59_my.png
try:
    _c0 = get_crop(0, 168, 144)
    canvas.paste(_c0, (48, 120), _c0)
except Exception:
    pass
layout["4.59_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/01_icon_Tracking.png
try:
    _c1 = get_crop(1, 288, 168)
    canvas.paste(_c1, (864, 2792), _c1)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 46, 70)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 66, 62)
    canvas.paste(_c3, (242, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 3, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/04_icon_Browse.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/05_icon_Miami_Dolphins.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 807), _c5)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/06_icon_Just_Announced_by_My_Performers.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 1688), _c6)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 95, 68)
    canvas.paste(_c8, (1217, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1217, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/09_icon_Clear.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 120), _c9)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/10_icon_Miami_Dolphins.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 639), _c10)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/11_icon_Events_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1520), _c11)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/12_icon_Taylor_Swift.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["Taylor_Swift"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/13_icon_Wicked.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 471), _c13)
except Exception:
    pass
layout["Wicked"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 68)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 60)
    canvas.paste(_c16, (315, 5), _c16)
except Exception:
    pass
layout["icon_16"] = [315, 5, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/17_icon_4.59_my.png
try:
    _c17 = get_crop(17, 47, 63)
    canvas.paste(_c17, (186, 1), _c17)
except Exception:
    pass
layout["4.59_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/18_icon_Search.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/19_icon_Performer_event_or_venue.png
try:
    _c19 = get_crop(19, 1032, 144)
    canvas.paste(_c19, (216, 120), _c19)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/20_icon_Rolling_Stones.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1143), _c20)
except Exception:
    pass
layout["Rolling_Stones"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/22_icon_Just_Announced_by_My_Performers.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1856), _c22)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/23_text_4.59_my.png
try:
    _c23 = get_crop(23, 153, 52)
    canvas.paste(_c23, (19, 9), _c23)
except Exception:
    pass
layout["4.59_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/24_text_Recent_searches.png
try:
    _c24 = get_crop(24, 168, 144)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_03_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-6/25_text_Suggestions.png
try:
    _c25 = get_crop(25, 331, 74)
    canvas.paste(_c25, (40, 1423), _c25)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
