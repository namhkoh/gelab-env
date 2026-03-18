# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_04
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7.png
# step_index: 4/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# fill overall background to match the app's very light gray canvas
draw.rectangle([(0, 0), canvas.size], fill="#FBFBFB")

# STATUS BAR
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#F3F3F3")
# subtle bottom divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#E6E6E6", width=1)

# SEARCH BAR (rounded background only — icons/text will be pasted on top)
search_left, search_top = 24, 100
search_right, search_bottom = 1416, 244
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=36,
    fill="#FFFFFF",
    outline="#EDEDED",
    width=1
)
# subtle shadow line under search bar
draw.line([(search_left+8, search_bottom+2), (search_right-8, search_bottom+2)], fill="#F0F0F0", width=2)

# MAIN SECTION GROUP CARDS (subtle white cards on light gray background)
# Top Results card area (background only)
draw.rounded_rectangle([(24, 300), (1416, 700)], radius=12, fill="#FFFFFF", outline="#F0F0F0", width=1)
# Performers card area
draw.rounded_rectangle([(24, 960), (1416, 1288)], radius=12, fill="#FFFFFF", outline="#F0F0F0", width=1)
# Events list area
draw.rounded_rectangle([(24, 1440), (1416, 2080)], radius=12, fill="#FFFFFF", outline="#F0F0F0", width=1)
# Recent searches card
draw.rounded_rectangle([(24, 2360), (1416, 2680)], radius=12, fill="#FFFFFF", outline="#F0F0F0", width=1)

# SEPARATOR LINES between major sections (thin, subtle)
separator_color = "#EFEFF0"
separators = [300, 720, 960, 1288, 1440, 2080, 2360, 2680]
for y in separators:
    draw.line([(24, y), (1416, y)], fill=separator_color, width=1)

# Additional subtle dividers across full width to match screenshot rhythm
extra_divs = [256, 360, 820, 1120, 1400, 1960, 2140, 2440]
for y in extra_divs:
    draw.line([(0, y), (1440, y)], fill="#F6F6F6", width=1)

# BOTTOM NAV BAR area (background + top divider)
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#E8E8E8", width=1)
# tiny shadow above nav to separate from content
draw.line([(0, nav_top-2), (1440, nav_top-2)], fill="#F7F7F7", width=1)

# subtle left/right edge inner padding guides (very faint) to imply content area margins
draw.line([(24, status_h+8), (24, 2960-nav_top//6)], fill="#FBFBFB", width=0)  # no-op visually, keep margins consistent
draw.line([(1416, status_h+8), (1416, 2960-nav_top//6)], fill="#FBFBFB", width=0)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/00_icon_lion_kingl.png
try:
    _c0 = get_crop(0, 1032, 144)
    canvas.paste(_c0, (216, 120), _c0)
except Exception:
    pass
layout["lion_kingl"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/01_icon_Top.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Top"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/02_icon_Performers.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1217), _c2)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/03_icon_8_events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1396), _c3)
except Exception:
    pass
layout["8_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/04_icon_Recent_searches.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 2530), _c4)
except Exception:
    pass
layout["Recent_searches"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/05_icon_8_events.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 650), _c5)
except Exception:
    pass
layout["8_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/06_icon_Tonight.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 829), _c6)
except Exception:
    pass
layout["Tonight"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/07_icon_7.40_W.png
try:
    _c7 = get_crop(7, 168, 144)
    canvas.paste(_c7, (48, 120), _c7)
except Exception:
    pass
layout["7.40_W"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/08_icon_Rrooklvn_Nets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Rrooklvn_Nets"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 45, 69)
    canvas.paste(_c9, (1154, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1154, 0, 1199, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 93, 68)
    canvas.paste(_c10, (1219, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1219, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/11_icon_Omaha_NE.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 1784), _c11)
except Exception:
    pass
layout["Omaha,_NE"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/12_icon_Events.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 1784), _c12)
except Exception:
    pass
layout["Events"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/14_icon_Omaha.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 650), _c14)
except Exception:
    pass
layout["Omaha"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 67)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/16_icon_Tracking.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (864, 2792), _c16)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/17_icon_Omaha.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 471), _c17)
except Exception:
    pass
layout["Omaha"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/18_icon_Rrooklvn_Nets.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Rrooklvn_Nets"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/19_icon_Account.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (1152, 2792), _c19)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/20_icon_The_Lion.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 1963), _c20)
except Exception:
    pass
layout["The_Lion"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/21_icon_GK.png
try:
    _c21 = get_crop(21, 62, 57)
    canvas.paste(_c21, (305, 4), _c21)
except Exception:
    pass
layout["GK"] = [305, 4, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/22_icon_Tickets.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (576, 2792), _c22)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/23_icon_New_York.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 2142), _c23)
except Exception:
    pass
layout["New_York"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/24_icon_7.40_W.png
try:
    _c24 = get_crop(24, 53, 62)
    canvas.paste(_c24, (182, 3), _c24)
except Exception:
    pass
layout["7.40_W"] = [182, 3, 235, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/25_icon_Hamilton.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (0, 2792), _c25)
except Exception:
    pass
layout["Hamilton"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/26_icon_The_Lion.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 2142), _c26)
except Exception:
    pass
layout["The_Lion"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/27_icon_King.png
try:
    _c27 = get_crop(27, 1440, 179)
    canvas.paste(_c27, (0, 1217), _c27)
except Exception:
    pass
layout["King"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/28_icon_Omaha_NE.png
try:
    _c28 = get_crop(28, 1440, 179)
    canvas.paste(_c28, (0, 829), _c28)
except Exception:
    pass
layout["Omaha,_NE"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/29_icon_The_Lion.png
try:
    _c29 = get_crop(29, 295, 60)
    canvas.paste(_c29, (234, 1247), _c29)
except Exception:
    pass
layout["The_Lion"] = [234, 1247, 529, 1307]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/30_text_7.40_W.png
try:
    _c30 = get_crop(30, 153, 49)
    canvas.paste(_c30, (19, 12), _c30)
except Exception:
    pass
layout["7.40_W"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/31_text_results.png
try:
    _c31 = get_crop(31, 190, 63)
    canvas.paste(_c31, (144, 374), _c31)
except Exception:
    pass
layout["results"] = [144, 374, 334, 437]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/32_text_Performers.png
try:
    _c32 = get_crop(32, 293, 54)
    canvas.paste(_c32, (44, 1122), _c32)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/33_text_Lion_King_Jr.png
try:
    _c33 = get_crop(33, 260, 56)
    canvas.paste(_c33, (234, 1432), _c33)
except Exception:
    pass
layout["Lion_King_Jr:"] = [234, 1432, 494, 1488]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/34_text_8_events.png
try:
    _c34 = get_crop(34, 173, 43)
    canvas.paste(_c34, (237, 1497), _c34)
except Exception:
    pass
layout["8_events"] = [237, 1497, 410, 1540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/35_text_Events.png
try:
    _c35 = get_crop(35, 179, 52)
    canvas.paste(_c35, (44, 1691), _c35)
except Exception:
    pass
layout["Events"] = [44, 1691, 223, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/36_text_Recent_searches.png
try:
    _c36 = get_crop(36, 436, 54)
    canvas.paste(_c36, (44, 2435), _c36)
except Exception:
    pass
layout["Recent_searches"] = [44, 2435, 480, 2489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/37_text_Hamilton.png
try:
    _c37 = get_crop(37, 205, 52)
    canvas.paste(_c37, (236, 2588), _c37)
except Exception:
    pass
layout["Hamilton"] = [236, 2588, 441, 2640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_04_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-7/38_text_Rrooklvn_Nets.png
try:
    _c38 = get_crop(38, 288, 162)
    canvas.paste(_c38, (288, 2792), _c38)
except Exception:
    pass
layout["Rrooklvn_Nets"] = [288, 2792, 576, 2954]
