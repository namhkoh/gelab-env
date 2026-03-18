# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_02
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5.png
# step_index: 2/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw app background and UI structural elements for the provided canvas/draw
# Assumes variables provided: canvas (PIL.Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors (match the screenshot's subtle greys and white)
BG = (255, 255, 255)
STATUS_BG = (247, 247, 247)        # top status bar background (very light grey)
SEARCH_BG = (250, 250, 250)        # search bar background (slightly off-white)
SEARCH_OUTLINE = (233, 233, 233)   # search bar outline / subtle border
DIVIDER = (230, 230, 230)          # thin divider lines
BOTTOM_BG = (255, 255, 255)        # bottom navigation background (white)
CONTENT_SHADOW = (242, 242, 242)   # faint shadow line under header

# Fill background (canvas starts white but explicitly ensure)
draw.rectangle([0, 0, W, H], fill=BG)

# Status bar area at top (~0-72 px)
status_height = 72
draw.rectangle([0, 0, W, status_height], fill=STATUS_BG)

# subtle bottom hairline under status bar
draw.line([(0, status_height), (W, status_height)], fill=DIVIDER, width=1)

# Search bar area (rounded rectangle)
search_left = 48
search_top = 60
search_right = W - 48
search_bottom = 192
search_radius = 30
draw.rounded_rectangle(
    [search_left, search_top, search_right, search_bottom],
    radius=search_radius,
    fill=SEARCH_BG,
    outline=SEARCH_OUTLINE,
    width=1
)

# Slight inner highlight/shadow for search bar to match screenshot subtle depth
# top highlight
draw.line([(search_left + 2, search_top + 2), (search_right - 2, search_top + 2)], fill=(255,255,255), width=1)
# bottom subtle shadow
draw.line([(search_left + 2, search_bottom - 2), (search_right - 2, search_bottom - 2)], fill=CONTENT_SHADOW, width=1)

# Divider line below search / header area
divider_y = search_bottom + 24
draw.line([(48, divider_y), (W - 48, divider_y)], fill=DIVIDER, width=2)

# Section separator between "Recent searches" list and "Suggestions"
# Based on detected element positions, place a thin divider roughly under the recent list
sep_y = 1310
draw.line([(48, sep_y), (W - 48, sep_y)], fill=DIVIDER, width=2)

# Subtle top border for the suggestions section to give separation
draw.line([(48, sep_y + 1), (W - 48, sep_y + 1)], fill=(245,245,245), width=1)

# Bottom navigation bar background and top divider
bottom_nav_top = 2720
draw.rectangle([0, bottom_nav_top, W, H], fill=BOTTOM_BG)
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=DIVIDER, width=1)

# Additional subtle separators for content groups (do not draw icons/text)
# Light horizontal guides to imply grouping without adding duplicated content
group_lines = [360, 588, 816, 1044, 1270]  # approximate y positions near list items
for y in group_lines:
    # Draw very faint lines with some spacing to avoid conflicting with item artwork
    draw.line([(48, y), (W - 48, y)], fill=(248, 248, 248), width=1)

# Slight rounded card background behind the "Suggestions" area to hint grouping (very subtle)
suggest_card_top = 1400
suggest_card_bottom = 1960
suggest_card_left = 32
suggest_card_right = W - 32
draw.rounded_rectangle(
    [suggest_card_left, suggest_card_top, suggest_card_right, suggest_card_bottom],
    radius=12,
    fill=BG,
    outline=(250,250,250),
    width=1
)

# Small top shadow under the page header (below status/search area) to match screenshot subtlety
shadow_y = divider_y + 6
draw.rectangle([48, shadow_y, W - 48, shadow_y + 2], fill=(245,245,245))

# Save into canvas (drawing is direct on provided canvas); nothing to return

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 49, 70)
    canvas.paste(_c0, (1152, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1152, 0, 1201, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/01_icon_Tracking.png
try:
    _c1 = get_crop(1, 288, 168)
    canvas.paste(_c1, (864, 2792), _c1)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/02_icon_Browse.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (0, 2792), _c2)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 97, 69)
    canvas.paste(_c3, (1216, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1216, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/04_icon_Miami_Dolphins.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 807), _c4)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 66, 63)
    canvas.paste(_c5, (242, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 2, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/06_icon_4.59_Wy.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["4.59_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/09_icon_Clear.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 120), _c9)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/10_icon_Account.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (1152, 2792), _c10)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/11_icon_Miami_Dolphins.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 639), _c11)
except Exception:
    pass
layout["Miami_Dolphins"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 69)
    canvas.paste(_c12, (1319, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/13_icon_4.59_Wy.png
try:
    _c13 = get_crop(13, 49, 63)
    canvas.paste(_c13, (185, 1), _c13)
except Exception:
    pass
layout["4.59_Wy"] = [185, 1, 234, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/14_icon_Taylor_Swift.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 975), _c14)
except Exception:
    pass
layout["Taylor_Swift"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/16_icon_Wicked.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 471), _c16)
except Exception:
    pass
layout["Wicked"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/17_icon_4.59_Wy.png
try:
    _c17 = get_crop(17, 59, 65)
    canvas.paste(_c17, (112, 0), _c17)
except Exception:
    pass
layout["4.59_Wy"] = [112, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/18_icon_Search.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/19_icon_Rolling_Stones.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 1143), _c19)
except Exception:
    pass
layout["Rolling_Stones"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/20_icon_Performer_event_or_venue.png
try:
    _c20 = get_crop(20, 1032, 144)
    canvas.paste(_c20, (216, 120), _c20)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/21_icon_Just_Announced_by_My_Performers.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1856), _c21)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/22_icon_Beyonce.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 639), _c22)
except Exception:
    pass
layout["Beyonce"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/23_text_Recent_searches.png
try:
    _c23 = get_crop(23, 168, 144)
    canvas.paste(_c23, (48, 120), _c23)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_02_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-5/24_text_Suggestions.png
try:
    _c24 = get_crop(24, 331, 74)
    canvas.paste(_c24, (40, 1423), _c24)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
