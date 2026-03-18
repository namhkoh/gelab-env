# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_04
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7.png
# step_index: 4/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page

# Overall background (light neutral)
draw.rectangle([(0, 0), canvas.size], fill="#faf9f7")

# Status bar area at top (~0-80px)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#efefef")
# thin bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#e3e3e3", width=1)

# Search bar background (rounded) under status bar
search_left = 40
search_right = 1400
search_top = 70
search_bottom = 170
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=28,
    fill="#ffffff",
    outline="#ececec",
    width=1
)
# subtle shadow under search bar
draw.line([(search_left+6, search_bottom+2), (search_right-6, search_bottom+2)], fill="#f0f0f0", width=2)

# Primary section separators (full-width subtle rules)
separators = [
    210,   # below search area
    640,   # end of top results block
    1060,  # end of events block
    1700,  # end of venues block
    2360,  # end of recent searches block header area
]
for y in separators:
    draw.line([(24, y), (1440-24, y)], fill="#e7e7e7", width=1)

# Section "cards" / group backgrounds as rounded white panels
panel_margin_x = 36
# Top results panel (behind the list items)
top_panel = (panel_margin_x, 300, 1440 - panel_margin_x, 640)
draw.rounded_rectangle([top_panel[0:2], top_panel[2:4]], radius=12, fill="#ffffff", outline="#efefef", width=1)
# small internal dividers for list rows inside top panel
draw.line([(top_panel[0]+18, 430), (top_panel[2]-18, 430)], fill="#f0f0f0", width=1)
draw.line([(top_panel[0]+18, 520), (top_panel[2]-18, 520)], fill="#f0f0f0", width=1)

# Events panel
events_panel = (panel_margin_x, 1000, 1440 - panel_margin_x, 1400)
draw.rounded_rectangle([events_panel[0:2], events_panel[2:4]], radius=12, fill="#ffffff", outline="#efefef", width=1)
draw.line([(events_panel[0]+18, 1150), (events_panel[2]-18, 1150)], fill="#f0f0f0", width=1)
draw.line([(events_panel[0]+18, 1280), (events_panel[2]-18, 1280)], fill="#f0f0f0", width=1)

# Venues panel
venues_panel = (panel_margin_x, 1640, 1440 - panel_margin_x, 1960)
draw.rounded_rectangle([venues_panel[0:2], venues_panel[2:4]], radius=12, fill="#ffffff", outline="#efefef", width=1)
draw.line([(venues_panel[0]+18, 1800), (venues_panel[2]-18, 1800)], fill="#f0f0f0", width=1)

# Recent searches panel
recent_panel = (panel_margin_x, 2320, 1440 - panel_margin_x, 2720)
draw.rounded_rectangle([recent_panel[0:2], recent_panel[2:4]], radius=12, fill="#ffffff", outline="#efefef", width=1)
draw.line([(recent_panel[0]+18, 2550), (recent_panel[2]-18, 2550)], fill="#f0f0f0", width=1)

# Thin separators between groups (slightly inset)
inset = 36
group_dividers = [
    (inset, 620, 1440-inset, 620),
    (inset, 1040, 1440-inset, 1040),
    (inset, 1680, 1440-inset, 1680),
    (inset, 2360, 1440-inset, 2360),
]
for x1, y1, x2, y2 in group_dividers:
    draw.line([(x1, y1), (x2, y2)], fill="#e9e9e9", width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e6", width=1)
# subtle shadow lines above the nav bar for depth
draw.line([(0, nav_top-2), (1440, nav_top-2)], fill="#f3f3f3", width=1)
draw.line([(0, nav_top-6), (1440, nav_top-6)], fill="#fafafa", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 58, 61)
    canvas.paste(_c0, (245, 3), _c0)
except Exception:
    pass
layout["icon_0"] = [245, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/01_icon_Los_Angeles_CA.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/02_icon_Iustin_Timherlake.png
try:
    _c2 = get_crop(2, 288, 162)
    canvas.paste(_c2, (288, 2792), _c2)
except Exception:
    pass
layout["Iustin_Timherlake"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/03_icon_The_Fonda_Theatrel.png
try:
    _c3 = get_crop(3, 1032, 144)
    canvas.paste(_c3, (216, 120), _c3)
except Exception:
    pass
layout["The_Fonda_Theatrel"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/04_icon_8.11_Wy.png
try:
    _c4 = get_crop(4, 55, 61)
    canvas.paste(_c4, (114, 2), _c4)
except Exception:
    pass
layout["8.11_Wy"] = [114, 2, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 53, 61)
    canvas.paste(_c5, (315, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [315, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/06_icon_New_York.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1575), _c6)
except Exception:
    pass
layout["New_York;"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/07_icon_CABARET.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 1575), _c7)
except Exception:
    pass
layout["CABARET"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 93, 68)
    canvas.paste(_c8, (1219, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1219, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 69)
    canvas.paste(_c9, (1155, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1155, 0, 1198, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/10_icon_CABARET.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 1396), _c10)
except Exception:
    pass
layout["CABARET"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/11_icon_8.11_Wy.png
try:
    _c11 = get_crop(11, 44, 61)
    canvas.paste(_c11, (187, 2), _c11)
except Exception:
    pass
layout["8.11_Wy"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/12_icon_8.11_Wy.png
try:
    _c12 = get_crop(12, 168, 144)
    canvas.paste(_c12, (48, 120), _c12)
except Exception:
    pass
layout["8.11_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/13_icon_Lunt-Fontanne_Theatre.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 1963), _c13)
except Exception:
    pass
layout["Lunt-Fontanne_Theatre"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/14_icon_Los_Angeles_CA.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1217), _c14)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/15_icon_Tickets.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (576, 2792), _c15)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 64)
    canvas.paste(_c16, (1326, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1326, 3, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/17_icon_IcABARET.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 829), _c17)
except Exception:
    pass
layout["IcABARET"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/18_icon_Clear.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 120), _c18)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/19_icon_Tracking.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (864, 2792), _c19)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/20_icon_IcABARET.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 650), _c20)
except Exception:
    pass
layout["IcABARET"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/21_icon_New_York_NY.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 829), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/22_icon_Browse.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (0, 2792), _c22)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/23_icon_Account.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (1152, 2792), _c23)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/24_icon_Lunt-Fontanne_Theatre.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 2142), _c24)
except Exception:
    pass
layout["Lunt-Fontanne_Theatre"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/25_icon_Events.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 1217), _c25)
except Exception:
    pass
layout["Events"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/26_icon_Los_Angeles_CA.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 650), _c26)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/27_icon_Mannequin_Pussy.png
try:
    _c27 = get_crop(27, 1440, 179)
    canvas.paste(_c27, (0, 1575), _c27)
except Exception:
    pass
layout["Mannequin_Pussy"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/28_icon_Mkgee.png
try:
    _c28 = get_crop(28, 163, 56)
    canvas.paste(_c28, (233, 683), _c28)
except Exception:
    pass
layout["Mkgee"] = [233, 683, 396, 739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/29_text_Top_results.png
try:
    _c29 = get_crop(29, 295, 72)
    canvas.paste(_c29, (40, 373), _c29)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/30_text_Events.png
try:
    _c30 = get_crop(30, 177, 54)
    canvas.paste(_c30, (46, 1122), _c30)
except Exception:
    pass
layout["Events"] = [46, 1122, 223, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/31_text_Venues.png
try:
    _c31 = get_crop(31, 195, 56)
    canvas.paste(_c31, (43, 1868), _c31)
except Exception:
    pass
layout["Venues"] = [43, 1868, 238, 1924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/32_text_Recent_searches.png
try:
    _c32 = get_crop(32, 436, 54)
    canvas.paste(_c32, (44, 2435), _c32)
except Exception:
    pass
layout["Recent_searches"] = [44, 2435, 480, 2489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/33_text_Madison_Square_Garden.png
try:
    _c33 = get_crop(33, 1440, 168)
    canvas.paste(_c33, (0, 2530), _c33)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_04_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-7/34_text_Iustin_Timherlake.png
try:
    _c34 = get_crop(34, 288, 162)
    canvas.paste(_c34, (288, 2792), _c34)
except Exception:
    pass
layout["Iustin_Timherlake"] = [288, 2792, 576, 2954]
