# page_id: page_seatgeek_094b5cdb02e246858451240263e6ef7f_05
# screenshot: 2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8.png
# step_index: 5/9
# task: Open SeatGeek. Find the soonest upcoming NBA game in Boston with "Celtics". What is the highest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for SeatGeek-like mobile page
# Assumes variables provided in environment:
# - canvas: PIL.Image (1440x2960 RGB)
# - draw: PIL.ImageDraw.Draw(canvas)
# - font_sm, font_md, font_lg, font_xl available (not used for text here)

# Colors
BG_OFFWHITE = "#F6F6F6"
WHITE = "#FFFFFF"
LIGHT_DIV = "#E6E6E6"
CARD_BG = "#FFFFFF"
HERO_TOP = "#15381E"      # deep green/teal to suggest arena photo area
HERO_BOTTOM = "#2A6B3A"
STATUS_BAR = "#0A0A0A"
SUBTLE_SHADOW = (0, 0, 0, 18)  # not directly used (RGB canvas), simulate with darker line

W, H = canvas.size

# 1) Base background
draw.rectangle((0, 0, W, H), fill=BG_OFFWHITE)

# 2) Status bar area (top ~56px)
status_h = 56
draw.rectangle((0, 0, W, status_h), fill=STATUS_BAR)

# 3) Hero image background (large stadium photo placeholder)
# Keep it as a horizontal gradient to emulate photo tonal variety.
hero_top = status_h
hero_bottom = 560  # approximate bottom of hero/photo area
# Simple vertical gradient
for i in range(hero_top, hero_bottom):
    # Interpolate between HERO_TOP and HERO_BOTTOM
    t = (i - hero_top) / max(1, (hero_bottom - hero_top))
    r1, g1, b1 = (int(HERO_TOP[1:3], 16), int(HERO_TOP[3:5], 16), int(HERO_TOP[5:7], 16))
    r2, g2, b2 = (int(HERO_BOTTOM[1:3], 16), int(HERO_BOTTOM[3:5], 16), int(HERO_BOTTOM[5:7], 16))
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    draw.line([(0, i), (W, i)], fill=(r, g, b))

# Add a subtle dark vignette on hero edges (left & right)
vignette_width = 140
vignette_color = (10, 10, 10)
for x in range(vignette_width):
    alpha = int(20 * (1 - x / vignette_width))
    draw.line([(x, hero_top), (x, hero_bottom)], fill=(vignette_color[0], vignette_color[1], vignette_color[2]))
    draw.line([(W - 1 - x, hero_top), (W - 1 - x, hero_bottom)], fill=(vignette_color[0], vignette_color[1], vignette_color[2]))

# 4) White content area starting slightly overlapping the hero (rounded top edge)
content_top = hero_bottom - 28  # small overlap to create rounded-top feel
draw.rectangle((0, content_top, W, H), fill=WHITE)

# Soft rounded arc at top of content area by drawing a white rounded rectangle overlapping hero
corner_radius = 28
draw.rounded_rectangle((0, content_top - corner_radius, W, content_top + 2 * corner_radius),
                       radius=corner_radius, fill=WHITE)

# 5) Divider line under hero / header separation
draw.line([(24, content_top + 56), (W - 24, content_top + 56)], fill=LIGHT_DIV, width=1)

# 6) Header card area (title and guarantee badge area) as subtle white card with border
header_card_top = content_top + 20
header_card_bottom = header_card_top + 160
header_pad = 24
draw.rounded_rectangle((header_pad, header_card_top, W - header_pad, header_card_bottom),
                       radius=12, fill=CARD_BG, outline=LIGHT_DIV, width=1)

# 7) Thin separator line below header card
sep_y = header_card_bottom + 18
draw.line([(24, sep_y), (W - 24, sep_y)], fill=LIGHT_DIV, width=1)

# 8) "No Games" callout card area (large card) - a white rounded card
no_games_top = sep_y + 28
no_games_bottom = no_games_top + 210
draw.rounded_rectangle((24, no_games_top, W - 24, no_games_bottom),
                       radius=14, fill=CARD_BG, outline=LIGHT_DIV, width=1)

# Subtle inner divider inside the No Games card (for spacing), do not draw any text or buttons
draw.line([(40, no_games_top + 110), (W - 40, no_games_top + 110)], fill="#F2F2F2", width=1)

# 9) Separator under the No Games card
draw.line([(24, no_games_bottom + 22), (W - 24, no_games_bottom + 22)], fill=LIGHT_DIV, width=1)

# 10) Section header area for "All Games" (just background spacing)
all_games_title_top = no_games_bottom + 48
# Leave title text area blank; draw a subtle horizontal rule beneath
draw.line([(24, all_games_title_top + 80), (W - 24, all_games_title_top + 80)], fill=LIGHT_DIV, width=1)

# 11) List background (entire list area is white; we already have base white, add slight outer padding)
list_start = all_games_title_top + 100
list_item_height = 260  # approximate height for each list row as seen
list_gap = 28
num_preview_items = 4
current_y = list_start

# Draw three list item cards with rounded rectangle backgrounds (keeps content areas clear for pasted elements)
for i in range(num_preview_items):
    # Each item is a card with left date badge area; do not draw the badge itself (icons will be pasted)
    item_top = current_y
    item_bottom = item_top + list_item_height
    draw.rounded_rectangle((24, item_top, W - 24, item_bottom),
                           radius=14, fill=CARD_BG, outline=LIGHT_DIV, width=1)
    # Add a light drop-shadow line at bottom of each card
    shadow_y = item_bottom
    draw.line([(36, shadow_y), (W - 36, shadow_y)], fill="#F4F4F4", width=2)
    current_y = item_bottom + list_gap

# 12) Final page bottom separator
draw.line([(0, H - 220), (W, H - 220)], fill=LIGHT_DIV, width=1)

# 13) Floating subtle bottom sheet background (light)
bottom_sheet_top = H - 200
draw.rectangle((0, bottom_sheet_top, W, H), fill=BG_OFFWHITE)

# Note: This code intentionally avoids drawing any specific icons, text, or button shapes
# that correspond to the detected overlay elements. It only lays out the primary background,
# header/toolbar areas, section cards, separators, and list item card backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (60, 1376), _c0)
except Exception:
    pass
layout["Track_Now"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/01_icon_24.png
try:
    _c1 = get_crop(1, 1440, 367)
    canvas.paste(_c1, (0, 1785), _c1)
except Exception:
    pass
layout["24"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/02_icon_29.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 2519), _c2)
except Exception:
    pass
layout["29"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/03_icon_Boston_MA.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 1785), _c3)
except Exception:
    pass
layout["Boston,_MA"] = [0, 1785, 1440, 2152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/04_icon_Miami_FL.png
try:
    _c4 = get_crop(4, 1440, 367)
    canvas.paste(_c4, (0, 2152), _c4)
except Exception:
    pass
layout["Miami,_FL"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/05_icon_27.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 2152), _c5)
except Exception:
    pass
layout["27"] = [0, 2152, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/06_icon_Eastern_Conference_First_Round_Boston_Ce.png
try:
    _c6 = get_crop(6, 1440, 367)
    canvas.paste(_c6, (0, 2519), _c6)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 2519, 1440, 2886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/07_icon_5.00_Wy.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 84), _c7)
except Exception:
    pass
layout["5.00_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/08_icon_Track_this_performer.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1104, 84), _c8)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/09_icon_Share_this_performer.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1260, 84), _c9)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/10_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c10 = get_crop(10, 1440, 126)
    canvas.paste(_c10, (0, 933), _c10)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/11_icon_5.00_Wy.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (36, 84), _c11)
except Exception:
    pass
layout["5.00_Wy"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 56, 65)
    canvas.paste(_c12, (1316, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [1316, 4, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 64)
    canvas.paste(_c13, (1149, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1149, 5, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 100, 69)
    canvas.paste(_c14, (1217, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1217, 2, 1317, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 83, 106)
    canvas.paste(_c15, (1307, 951), _c15)
except Exception:
    pass
layout["icon_15"] = [1307, 951, 1390, 1057]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/16_icon_5.00_Wy.png
try:
    _c16 = get_crop(16, 51, 63)
    canvas.paste(_c16, (186, 2), _c16)
except Exception:
    pass
layout["5.00_Wy"] = [186, 2, 237, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/17_icon_Mon.png
try:
    _c17 = get_crop(17, 210, 54)
    canvas.paste(_c17, (56, 2906), _c17)
except Exception:
    pass
layout["Mon"] = [56, 2906, 266, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 64)
    canvas.paste(_c18, (245, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [245, 2, 301, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 61, 80)
    canvas.paste(_c19, (377, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [377, 1, 438, 81]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/20_text_No_Games_near_Los_Angeles_CA.png
try:
    _c20 = get_crop(20, 1440, 126)
    canvas.paste(_c20, (0, 933), _c20)
except Exception:
    pass
layout["No_Games_near_Los_Angeles"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/21_text_Track_Boston_Celtics_for_event_updates.png
try:
    _c21 = get_crop(21, 337, 153)
    canvas.paste(_c21, (60, 1376), _c21)
except Exception:
    pass
layout["Track_Boston_Celtics_for_"] = [60, 1376, 397, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/22_text_AIl_Games.png
try:
    _c22 = get_crop(22, 268, 60)
    canvas.paste(_c22, (59, 1682), _c22)
except Exception:
    pass
layout["AIl_Games"] = [59, 1682, 327, 1742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/094b5cdb02e246858451240263e6ef7f/step_05_2024_4_22_16_59_094b5cdb02e246858451240263e6ef7f-8/23_text_Fastern_Conference_First_Round_Miami_Hea.png
try:
    _c23 = get_crop(23, 1440, 367)
    canvas.paste(_c23, (0, 2519), _c23)
except Exception:
    pass
layout["Fastern_Conference_First_"] = [0, 2519, 1440, 2886]
