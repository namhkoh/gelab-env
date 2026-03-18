# page_id: page_seatgeek_2c6b8c5734894f77ba798a927b118406_01
# screenshot: 2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4.png
# step_index: 1/5
# task: Open SeatGeek. Search "Wembley Stadium". Show the next five football matches. Add to watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural UI background and separators for the mobile page

# Colors
status_bar_color = "#efefef"      # light grey status bar
divider_color = "#e6e6e6"         # subtle dividers
card_shadow = (0, 0, 0, 20)       # not used directly (no alpha brush), keep for reference
muted_bg = "#ffffff"              # main background (white)
subtle_grey = "#fbfbfb"           # very subtle panel bg
trim_grey = "#f5f5f5"             # slightly darker than background
accent_edge = "#f7f5f3"           # warm accent for sections

W, H = canvas.size

# 1) Status bar area
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)
# thin bottom border under status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=divider_color, width=1)

# 2) Header area (toolbar) - leave content blank but draw bottom divider
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (W, header_bottom)], fill=muted_bg)
draw.line([(24, header_bottom), (W - 24, header_bottom)], fill=divider_color, width=1)

# 3) Large hero/feature area is represented by detected icon "Knicks" and MUST NOT be redrawn.
# We'll add a very subtle shadow strip under where that hero sits (so content pasted on top looks natural)
# Hero top is visually around y ~300.. we add a faint shadow band below it.
hero_shadow_top = 1160  # keep well below; subtle separator before "Just for you"
draw.line([(24, hero_shadow_top), (W - 24, hero_shadow_top)], fill=trim_grey, width=1)

# 4) "Just for you" section area background (subtle)
# Provide a soft white backdrop and a thin divider under the row of cards.
just_section_top = 1240
just_section_bottom = 2010
# subtle panel (almost white) to indicate section grouping
draw.rectangle([(0, just_section_top), (W, just_section_bottom)], fill=subtle_grey)
# top and bottom separators for the section
draw.line([(24, just_section_top), (W - 24, just_section_top)], fill=divider_color, width=1)
draw.line([(24, just_section_bottom - 1), (W - 24, just_section_bottom - 1)], fill=divider_color, width=1)

# 5) Trending events area (list background)
trending_top = just_section_bottom
trending_bottom = 2760
draw.rectangle([(0, trending_top), (W, trending_bottom)], fill=muted_bg)

# Trending header separator (keep space for the "Trending events" title which will be pasted)
trending_header_y = trending_top + 48
draw.line([(24, trending_header_y), (W - 24, trending_header_y)], fill=divider_color, width=1)

# 6) Draw list separators for trending rows (approx positions based on detected text positions)
# We'll draw 4 separators to divide items into rows.
row_sep_y = [2260, 2490, 2720]  # approximate Y positions for separators between list items
for y in row_sep_y:
    # draw a thin hairline across the content inset by left/right margins
    draw.line([(24, y), (W - 24, y)], fill=divider_color, width=1)

# 7) Left inset guide column (visual alignment) - very subtle vertical line where list content aligns
# This is only a faint guide to match layout structure, not any text or icon.
left_guide_x = 96
draw.line([(left_guide_x, trending_header_y + 12), (left_guide_x, trending_bottom - 40)], fill=accent_edge, width=1)

# 8) Bottom navigation bar background and top border
nav_top = 2792
draw.rectangle([(0, nav_top), (W, H)], fill=muted_bg)
draw.line([(24, nav_top), (W - 24, nav_top)], fill=divider_color, width=1)

# 9) Safe area inset (a slight overlay band just above nav to create separation)
safe_band_top = nav_top - 18
draw.rectangle([(0, safe_band_top), (W, nav_top)], fill="#ffffff")
draw.line([(24, safe_band_top), (W - 24, safe_band_top)], fill=trim_grey, width=1)

# 10) Small decorative horizontal rhythm lines to suggest grouped cards (non-intrusive)
# These are subtle and do not duplicate any app text/icons.
rhythm_y = [just_section_top + 48, just_section_top + 120]
for y in rhythm_y:
    draw.line([(48, y), (W - 48, y)], fill=accent_edge, width=1)

# Note: All interactive elements (icons, thumbnails, text) will be pasted on top of this structure.
# This script intentionally avoids drawing any icon, label, or button content detected elsewhere.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/00_icon_Knicks.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Knicks"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/01_icon_BOOK_OF.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 1431), _c1)
except Exception:
    pass
layout["BOOK_OF"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/02_icon_August_Wilson_Theatre.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 2183), _c2)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 2183, 1309, 2419]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/03_icon_Yankee_Stadium.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2419), _c3)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2419, 1309, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/04_icon_S116.png
try:
    _c4 = get_crop(4, 396, 519)
    canvas.paste(_c4, (1044, 1431), _c4)
except Exception:
    pass
layout["S116+"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/05_icon_S94.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (546, 1431), _c5)
except Exception:
    pass
layout["S94+"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 99, 152)
    canvas.paste(_c6, (1341, 2464), _c6)
except Exception:
    pass
layout["icon_6"] = [1341, 2464, 1440, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/07_icon_View_all.png
try:
    _c7 = get_crop(7, 98, 149)
    canvas.paste(_c7, (1342, 2228), _c7)
except Exception:
    pass
layout["View_all"] = [1342, 2228, 1440, 2377]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 61, 58)
    canvas.paste(_c8, (243, 5), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/09_icon_May.png
try:
    _c9 = get_crop(9, 264, 183)
    canvas.paste(_c9, (1176, 2000), _c9)
except Exception:
    pass
layout["May"] = [1176, 2000, 1440, 2183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/10_icon_888.png
try:
    _c10 = get_crop(10, 99, 63)
    canvas.paste(_c10, (1214, 1), _c10)
except Exception:
    pass
layout["888"] = [1214, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/11_icon_7_05_my.png
try:
    _c11 = get_crop(11, 54, 57)
    canvas.paste(_c11, (115, 5), _c11)
except Exception:
    pass
layout["7:05_my"] = [115, 5, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/12_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (864, 2792), _c12)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/13_icon_888.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (1260, 72), _c13)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/14_icon_7_05_my.png
try:
    _c14 = get_crop(14, 47, 57)
    canvas.paste(_c14, (185, 5), _c14)
except Exception:
    pass
layout["7:05_my"] = [185, 5, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 63)
    canvas.paste(_c15, (1320, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [1320, 2, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/16_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/17_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (576, 2792), _c17)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 59)
    canvas.paste(_c18, (314, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [314, 5, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 46, 65)
    canvas.paste(_c19, (1154, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [1154, 1, 1200, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 99, 119)
    canvas.paste(_c20, (1341, 2698), _c20)
except Exception:
    pass
layout["icon_20"] = [1341, 2698, 1440, 2817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/21_icon_Browse.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/23_icon_Andrew_Schulz.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (546, 1431), _c23)
except Exception:
    pass
layout["Andrew_Schulz"] = [546, 1431, 1008, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 116, 127)
    canvas.paste(_c24, (1138, 2484), _c24)
except Exception:
    pass
layout["icon_24"] = [1138, 2484, 1254, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 390, 86)
    canvas.paste(_c25, (40, 119), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [40, 119, 430, 205]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/26_icon_The.png
try:
    _c26 = get_crop(26, 91, 102)
    canvas.paste(_c26, (36, 1427), _c26)
except Exception:
    pass
layout["The"] = [36, 1427, 127, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/27_text_date.png
try:
    _c27 = get_crop(27, 114, 52)
    canvas.paste(_c27, (137, 208), _c27)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/28_text_Just_for_you.png
try:
    _c28 = get_crop(28, 306, 66)
    canvas.paste(_c28, (38, 1310), _c28)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 1248), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/30_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (576, 2792), _c30)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_01_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-4/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (906, 1455), _c32)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
