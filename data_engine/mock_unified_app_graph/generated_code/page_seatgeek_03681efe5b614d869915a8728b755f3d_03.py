# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_03
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6.png
# step_index: 3/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas using provided canvas and draw objects.
# Available: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors (match screenshot subtle palette)
BG = "#FFFFFF"
STATUS_BAR = "#efefef"        # very light grey status bar
SEARCH_BG = "#f7f7f7"         # search field background
SEARCH_BORDER = "#e6e6e6"     # search field border
DIVIDER = "#e9e9e9"           # subtle dividers
CARD_BG = "#fbfbfb"           # subtle card background for grouped sections
BOTTOM_SHADOW = "#ededed"     # top line of bottom nav

w, h = canvas.size

# Base background (canvas is already white but set explicitly)
draw.rectangle((0, 0, w, h), fill=BG)

# Status bar area at top (~50-80px)
status_h = 80
draw.rectangle((0, 0, w, status_h), fill=STATUS_BAR)
# subtle bottom line of status bar
draw.line((0, status_h-1, w, status_h-1), fill=DIVIDER, width=1)

# Search bar background (rounded) - underlying background only (icons/text will be pasted on top)
search_x0, search_x1 = 40, w - 40
search_y0, search_y1 = 96, 96 + 168  # aligns with detected search-area vertical region
search_radius = 40
draw.rounded_rectangle((search_x0, search_y0, search_x1, search_y1),
                       radius=search_radius,
                       fill=SEARCH_BG,
                       outline=SEARCH_BORDER,
                       width=2)

# thin divider under the search area (full width with side padding)
divider_y = search_y1 + 16
draw.line((search_x0, divider_y, search_x1, divider_y), fill=DIVIDER, width=1)

# Recent searches card background (grouping area) - subtle rounded card behind list rows
recent_card_y0 = divider_y + 32
recent_card_y1 = 1320  # end of recent searches area before suggestions header
draw.rounded_rectangle((24, recent_card_y0, w-24, recent_card_y1),
                       radius=16,
                       fill=CARD_BG,
                       outline=None)

# subtle horizontal separators to indicate section breaks within content
# Separator between recent searches and suggestions (visible as a thin rule)
sep1_y = 960
draw.line((40, sep1_y, w-40, sep1_y), fill=DIVIDER, width=1)

# Separator before "Suggestions" header area (a wider gap in UI)
sep2_y = 1320
draw.line((40, sep2_y, w-40, sep2_y), fill=DIVIDER, width=1)

# Suggestions area card background (rounded) - underlying background for suggestion items
suggest_card_y0 = sep2_y + 24
suggest_card_y1 = suggest_card_y0 + 520
draw.rounded_rectangle((24, suggest_card_y0, w-24, suggest_card_y1),
                       radius=16,
                       fill=CARD_BG,
                       outline=None)

# Bottom navigation bar background and top divider/shadow
bottom_nav_top = 2792  # matches provided detected icon y for nav row
draw.rectangle((0, bottom_nav_top, w, h), fill=BG)
# top shadow / divider for nav area
draw.line((0, bottom_nav_top, w, bottom_nav_top), fill=BOTTOM_SHADOW, width=2)

# Additional subtle vertical padding lines for layout alignment (left margin guide visuals)
# These are very faint and intended only as structural guides similar to the app's subtle guides
left_margin = 40
right_margin = w - 40
draw.line((left_margin, status_h, left_margin, bottom_nav_top), fill="#ffffff00")  # transparent/no-op
draw.line((right_margin, status_h, right_margin, bottom_nav_top), fill="#ffffff00")  # transparent/no-op

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/00_icon_Justin_Bieber.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/01_icon_Madison_Square_Garden.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 975), _c1)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/02_icon_Los_Angeles_Lakers.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 46, 70)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/05_icon_Recent_searches.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 471), _c5)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/06_icon_7.56_my.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["7.56_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 61, 64)
    canvas.paste(_c10, (243, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [243, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/11_icon_Seattle_Mariners.png
try:
    _c11 = get_crop(11, 128, 129)
    canvas.paste(_c11, (45, 828), _c11)
except Exception:
    pass
layout["Seattle_Mariners"] = [45, 828, 173, 957]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 100, 67)
    canvas.paste(_c12, (1215, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1215, 0, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/14_icon_Events_by_My_Performers.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 1520), _c14)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/15_icon_Madison_Square_Garden.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 807), _c15)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 66)
    canvas.paste(_c17, (1327, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/18_icon_Los_Angeles_Lakers.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 639), _c18)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 58, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/20_icon_7.56_my.png
try:
    _c20 = get_crop(20, 45, 63)
    canvas.paste(_c20, (187, 1), _c20)
except Exception:
    pass
layout["7.56_my"] = [187, 1, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/22_icon_Madison_Square_Garden.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/23_icon_Suggestions.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/25_icon_Search.png
try:
    _c25 = get_crop(25, 288, 162)
    canvas.paste(_c25, (288, 2792), _c25)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/26_text_7.56_my.png
try:
    _c26 = get_crop(26, 153, 52)
    canvas.paste(_c26, (19, 9), _c26)
except Exception:
    pass
layout["7.56_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/27_text_Performer_event_or_venue.png
try:
    _c27 = get_crop(27, 1032, 144)
    canvas.paste(_c27, (216, 120), _c27)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_03_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-6/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
