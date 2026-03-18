# page_id: page_seatgeek_1cc69540849e491bb4fc78ed1f09c554_03
# screenshot: 2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6.png
# step_index: 3/7
# task: Open SeatGeek. Search "Madison Square Garden". Select the next upcoming event. Who are the performers of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the provided mobile UI canvas.
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (PIL.ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Colors
STATUS_BAR_COLOR = (238, 238, 238)   # light gray for status bar
SEARCH_BG = (245, 245, 245)          # search box background
SEARCH_BORDER = (225, 225, 225)      # border for search box
DIVIDER = (236, 236, 236)            # thin dividers
CARD_BG = (250, 250, 251)            # subtle card background for suggestion section
BOTTOM_BORDER = (230, 230, 230)      # top border for bottom nav
SHADOW = (245, 245, 245)             # subtle shadow line

W = canvas.width
H = canvas.height

# 1) Overall background (canvas starts white, but reinforce to near-white)
draw.rectangle([(0, 0), (W, H)], fill=(255, 255, 255))

# 2) Status bar (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BAR_COLOR)
# subtle bottom border for status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=DIVIDER, width=1)

# 3) App header / toolbar area background (under status bar)
# Keep it white but ensure separation for search area
header_bottom = 110
draw.rectangle([(0, status_h), (W, header_bottom)], fill=(255, 255, 255))

# 4) Search box (rounded rectangle). Coordinates chosen to match detected layout margins.
search_left = 48
search_top = 120
search_right = W - 48
search_bottom = search_top + 144
search_radius = 20
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=SEARCH_BG,
    outline=SEARCH_BORDER,
    width=1
)
# subtle lower shadow line under search box
draw.line([(search_left + 4, search_bottom + 1), (search_right - 4, search_bottom + 1)], fill=SHADOW, width=1)

# 5) Thin divider under search area (full width inset to match content margins)
divider_y = search_bottom + 24
draw.line([(search_left, divider_y), (search_right, divider_y)], fill=DIVIDER, width=1)

# 6) Separator lines between list items (these are structural dividers only).
# Based on detected list item blocks; draw across content margins (left/right same as search)
divider_positions = [
    639,  # bottom of first recent item group (471 + 168)
    807,  # bottom of second
    975,  # bottom of third
    1143, # bottom of fourth
    1311, # bottom of fifth
    1856, # bottom of later section item
    2024, # following item bottom
]
for y in divider_positions:
    # keep dividers subtle and inset from full bleed
    draw.line([(search_left, y), (search_right, y)], fill=DIVIDER, width=1)

# 7) Suggestions section card background (rounded rectangle behind the suggestions group)
# Position chosen to start a little above the "Suggestions" heading and extend downward
suggest_card_top = 1420
suggest_card_bottom = 1920
suggest_card_radius = 12
draw.rounded_rectangle(
    [(search_left - 8, suggest_card_top), (search_right + 8, suggest_card_bottom)],
    radius=suggest_card_radius,
    fill=CARD_BG,
    outline=None
)
# subtle top divider above suggestions (reinforce separation)
draw.line([(search_left, suggest_card_top - 8), (search_right, suggest_card_top - 8)], fill=DIVIDER, width=1)

# 8) Large thin section divider (between main lists and suggestions area)
big_div_y = 1416
draw.line([(search_left, big_div_y), (search_right, big_div_y)], fill=DIVIDER, width=1)

# 9) Bottom navigation bar background (area at bottom where icons will be pasted)
bottom_nav_top = 2792
bottom_nav_rect = [(0, bottom_nav_top), (W, H)]
# Keep it white but draw a top border and light shadow
draw.rectangle(bottom_nav_rect, fill=(255, 255, 255))
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=BOTTOM_BORDER, width=1)
# add a subtle inner shadow under the top border
draw.line([(0, bottom_nav_top + 2), (W, bottom_nav_top + 2)], fill=SHADOW, width=1)

# 10) Additional subtle structural lines: small separators to imply section groupings
# a) top area under header
draw.line([(search_left, header_bottom + 12), (search_right, header_bottom + 12)], fill=DIVIDER, width=1)
# b) small divider under last recent-search item cluster (near 1316)
draw.line([(search_left, 1316), (search_right, 1316)], fill=DIVIDER, width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/00_icon_Madison_Square_Garden.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/01_icon_Bruno_Mars.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Bruno_Mars"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 47, 70)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/03_icon_7.44_W.png
try:
    _c3 = get_crop(3, 168, 144)
    canvas.paste(_c3, (48, 120), _c3)
except Exception:
    pass
layout["7.44_W"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/06_icon_L_Olympia_Olympia_Theatre.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 807), _c6)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/07_icon_L_Olympia_Olympia_Theatre.png
try:
    _c7 = get_crop(7, 1440, 168)
    canvas.paste(_c7, (0, 639), _c7)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 61, 63)
    canvas.paste(_c8, (243, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [243, 2, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/10_icon_L_Olympia_Olympia_Theatre.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 975), _c10)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 69)
    canvas.paste(_c11, (1216, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/12_icon_Coldplay.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1143), _c12)
except Exception:
    pass
layout["Coldplay"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/13_icon_Just_Announced_by_My_Performers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 1688), _c13)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/14_icon_7.44_W.png
try:
    _c14 = get_crop(14, 88, 63)
    canvas.paste(_c14, (18, 2), _c14)
except Exception:
    pass
layout["7.44_W"] = [18, 2, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/15_icon_7.44_W.png
try:
    _c15 = get_crop(15, 53, 64)
    canvas.paste(_c15, (116, 1), _c15)
except Exception:
    pass
layout["7.44_W"] = [116, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/16_icon_Clear.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 120), _c16)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/17_icon_7.44_W.png
try:
    _c17 = get_crop(17, 45, 63)
    canvas.paste(_c17, (187, 1), _c17)
except Exception:
    pass
layout["7.44_W"] = [187, 1, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 53, 68)
    canvas.paste(_c19, (1319, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/20_icon_Coldplay.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 975), _c20)
except Exception:
    pass
layout["Coldplay"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/21_icon_Events_by_My_Performers.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1520), _c21)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 64)
    canvas.paste(_c22, (313, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [313, 2, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/25_icon_Denver_Nuggets.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1143), _c25)
except Exception:
    pass
layout["Denver_Nuggets"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/26_icon_Just_Announced_by_My_Performers.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1856), _c26)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1cc69540849e491bb4fc78ed1f09c554/step_03_2024_4_22_19_44_1cc69540849e491bb4fc78ed1f09c554-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
