# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_02
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5.png
# step_index: 2/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page

# Colors
bg_color = (255, 255, 255)            # page background (white)
status_color = (238, 238, 238)        # top status bar (light gray)
search_bg = (247, 247, 247)           # search box background
search_border = (225, 225, 225)       # subtle border for search
card_bg = (255, 255, 255)             # card background (white)
card_outline = (230, 230, 230)        # card outline / shadow
divider = (220, 220, 220)             # thin separators
bottom_shadow = (245, 245, 245)       # shadow above bottom nav

W, H = canvas.size

# Fill full background (in case canvas isn't pure white)
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top area with icons)
status_h = 96
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)
# subtle bottom hairline under status bar
draw.line([(24, status_h - 1), (W - 24, status_h - 1)], fill=divider, width=1)

# Search bar (rounded pill)
search_left = 40
search_right = W - 40
search_top = 80
search_bottom = 200
search_radius = 40
# background
draw.rounded_rectangle([(search_left, search_top), (search_right, search_bottom)],
                       radius=search_radius, fill=search_bg, outline=search_border, width=1)

# subtle inner divider inside search area (to hint icon area) - light line near left area
icon_hint_x = search_left + 96
draw.line([(icon_hint_x, search_top + 18), (icon_hint_x, search_bottom - 18)], fill=(240,240,240), width=2)

# horizontal divider under search (separating header from content)
divider_y = search_bottom + 40
draw.line([(24, divider_y), (W - 24, divider_y)], fill=divider, width=1)

# Recent searches card container (rounded white card with faint outline)
recent_card_top = divider_y + 20
recent_card_left = 20
recent_card_right = W - 20
# cover the area where recent items are located (but leave content to be pasted)
recent_card_bottom = 1320
draw.rounded_rectangle([(recent_card_left, recent_card_top), (recent_card_right, recent_card_bottom)],
                       radius=16, fill=card_bg, outline=card_outline, width=1)

# subtle separators between list blocks inside the recent card (light hairlines)
# these are only structure lines; actual list items will be pasted on top
# place a few separators roughly matching item spacing
sep_x0 = recent_card_left + 24
sep_x1 = recent_card_right - 24
seps = [recent_card_top + 220, recent_card_top + 380, recent_card_top + 540, recent_card_top + 700]
for y in seps:
    draw.line([(sep_x0, y), (sep_x1, y)], fill=divider, width=1)

# big divider between Recent searches and Suggestions
big_div_y = recent_card_bottom + 16
draw.line([(24, big_div_y), (W - 24, big_div_y)], fill=divider, width=1)

# Suggestions section card (rounded, mostly white area to host suggestion rows)
suggest_top = big_div_y + 20
suggest_left = 20
suggest_right = W - 20
suggest_bottom = 2200
draw.rounded_rectangle([(suggest_left, suggest_top), (suggest_right, suggest_bottom)],
                       radius=16, fill=card_bg, outline=card_outline, width=1)

# separators inside Suggestions section
s_seps = [suggest_top + 120, suggest_top + 240, suggest_top + 360]
for y in s_seps:
    draw.line([(suggest_left + 24, y), (suggest_right - 24, y)], fill=divider, width=1)

# Large empty content area remains white; optionally draw a faint large divider before footer
content_div_y = 2600
draw.line([(24, content_div_y), (W - 24, content_div_y)], fill=(245,245,245), width=1)

# Bottom navigation background and top shadow
bottom_nav_top = 2792
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=card_bg)
# subtle shadow line above bottom nav
draw.rectangle([(0, bottom_nav_top - 8), (W, bottom_nav_top)], fill=bottom_shadow)

# final small accents: rounded corners on full-width separators near top (visual polish)
draw.line([(24, recent_card_top - 20), (W - 24, recent_card_top - 20)], fill=(250,250,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/01_icon_Justin_Bieber.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 49, 69)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/03_icon_Madison_Square_Garden.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 975), _c3)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 64)
    canvas.paste(_c5, (242, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 2, 305, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 97, 69)
    canvas.paste(_c7, (1216, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1216, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/10_icon_Los_Angeles_Lakers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 471), _c10)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/11_icon_Seattle_Mariners.png
try:
    _c11 = get_crop(11, 135, 129)
    canvas.paste(_c11, (43, 828), _c11)
except Exception:
    pass
layout["Seattle_Mariners"] = [43, 828, 178, 957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/13_icon_7.56_my.png
try:
    _c13 = get_crop(13, 47, 63)
    canvas.paste(_c13, (186, 1), _c13)
except Exception:
    pass
layout["7.56_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 68)
    canvas.paste(_c14, (1319, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/15_icon_7.56_my.png
try:
    _c15 = get_crop(15, 168, 144)
    canvas.paste(_c15, (48, 120), _c15)
except Exception:
    pass
layout["7.56_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/17_icon_Events_by_My_Performers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1520), _c17)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 61, 63)
    canvas.paste(_c18, (313, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [313, 2, 374, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/19_icon_Madison_Square_Garden.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 807), _c19)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/20_icon_Los_Angeles_Lakers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 639), _c20)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/21_icon_7.56_my.png
try:
    _c21 = get_crop(21, 57, 65)
    canvas.paste(_c21, (113, 0), _c21)
except Exception:
    pass
layout["7.56_my"] = [113, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/22_icon_Madison_Square_Garden.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/25_text_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_02_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-5/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
