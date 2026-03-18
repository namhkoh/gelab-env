# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_04
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7.png
# step_index: 4/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background fill (subtle off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFB")

# Status bar area (top ~56px) - light gray
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#ECECEC")

# Search bar area (rounded white card)
search_left = 48
search_top = 60
search_right = 1392
search_bottom = 148
search_radius = 28
# subtle underline shadow (very light)
draw.rectangle((search_left, search_bottom + 2, search_right, search_bottom + 4), fill="#F2F2F2")
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom), radius=search_radius, fill="#FFFFFF")

# Divider under search area
draw.line((32, 200, 1408, 200), fill="#E6E6E6", width=1)

# Define major content blocks (white panels) with slight spacing
blocks = [
    (24, 210, 1416, 830),    # Top results block
    (24, 850, 1416, 1396),   # Performers block
    (24, 1416, 1416, 1860),  # Events block
    (24, 1880, 1416, 2788),  # Recent searches block
]
for bbox in blocks:
    left, top, right, bottom = bbox
    draw.rounded_rectangle((left, top, right, bottom), radius=10, fill="#FFFFFF")

# Section separators (thin lines) - between logical groups (reinforce the UI separators)
separator_ys = [830, 1396, 1860, 2788]
for y in separator_ys:
    draw.line((24, y, 1416, y), fill="#E6E6E6", width=1)

# Subtle inset dividers inside blocks to suggest list rows (only structural, no text/icons)
# We'll add faint row separators spaced to match typical row heights (~179px) without drawing content
row_height = 179
for block in blocks:
    left, top, right, bottom = block
    # start a little below the block title area
    y = top + 80
    while y + 2 < bottom - 24:
        draw.line((left + 16, y, right - 16, y), fill="#F0F0F0", width=1)
        y += row_height

# Bottom navigation bar background (area reserved for nav icons)
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
# top divider for nav
draw.line((24, nav_top, 1416, nav_top), fill="#E6E6E6", width=1)

# Subtle left and right safe-area vertical guides (very faint) to mimic app padding
draw.line((24, 0, 24, 2960), fill="#FFFFFF", width=1)
draw.line((1416, 0, 1416, 2960), fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/00_icon_New_York_NY.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1963), _c0)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/01_icon_The_Book_Of_Mormon.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["The_Book_Of_Mormon"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/02_icon_New_York_NY.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 2142), _c2)
except Exception:
    pass
layout["New_York,_NY"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/03_icon_New_York_NY.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["New_York,_NY"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 62, 59)
    canvas.paste(_c4, (244, 4), _c4)
except Exception:
    pass
layout["icon_4"] = [244, 4, 306, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 43, 70)
    canvas.paste(_c5, (1155, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/06_icon_Vin.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (576, 2792), _c6)
except Exception:
    pass
layout["Vin~"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 93, 69)
    canvas.paste(_c7, (1219, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/08_icon_No_events.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 1575), _c8)
except Exception:
    pass
layout["No_events"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/09_icon_6.50_my.png
try:
    _c9 = get_crop(9, 168, 144)
    canvas.paste(_c9, (48, 120), _c9)
except Exception:
    pass
layout["6.50_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/10_icon_Vin.png
try:
    _c10 = get_crop(10, 288, 162)
    canvas.paste(_c10, (288, 2792), _c10)
except Exception:
    pass
layout["Vin~"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 56)
    canvas.paste(_c11, (318, 7), _c11)
except Exception:
    pass
layout["icon_11"] = [318, 7, 365, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/12_icon_Tracking.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (864, 2792), _c12)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/13_icon_6.50_my.png
try:
    _c13 = get_crop(13, 46, 61)
    canvas.paste(_c13, (185, 3), _c13)
except Exception:
    pass
layout["6.50_my"] = [185, 3, 231, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 44, 66)
    canvas.paste(_c14, (1326, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [1326, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/15_icon_The_Book_Of_Mormon.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 1396), _c15)
except Exception:
    pass
layout["The_Book_Of_Mormon"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/16_icon_New_York_NY.png
try:
    _c16 = get_crop(16, 1440, 179)
    canvas.paste(_c16, (0, 829), _c16)
except Exception:
    pass
layout["New_York,_NY"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/17_icon_The_Broadway_Musical_-_The_Book_Of_Mormo.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1575), _c17)
except Exception:
    pass
layout["The_Broadway_Musical_-_Th"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/18_icon_Clear.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 120), _c18)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/19_icon_Account.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (1152, 2792), _c19)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/20_icon_New_York_NY.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 2321), _c20)
except Exception:
    pass
layout["New_York;_NY"] = [0, 2321, 1440, 2500]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/21_icon_TL_T_r.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (0, 2792), _c21)
except Exception:
    pass
layout["TL^T:r"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/22_icon_Mori.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 471), _c22)
except Exception:
    pass
layout["Mori"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/23_icon_Marmi.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 1217), _c23)
except Exception:
    pass
layout["Marmi"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/24_icon_Mormi.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 829), _c24)
except Exception:
    pass
layout["Mormi"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/25_text_6.50_my.png
try:
    _c25 = get_crop(25, 153, 49)
    canvas.paste(_c25, (19, 12), _c25)
except Exception:
    pass
layout["6.50_my"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/26_text_Book_of_Mormon.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Book_of_Mormon"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/27_text_Top_results.png
try:
    _c27 = get_crop(27, 295, 72)
    canvas.paste(_c27, (40, 373), _c27)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/28_text_Performers.png
try:
    _c28 = get_crop(28, 293, 54)
    canvas.paste(_c28, (44, 1122), _c28)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/29_text_Events.png
try:
    _c29 = get_crop(29, 181, 57)
    canvas.paste(_c29, (43, 1868), _c29)
except Exception:
    pass
layout["Events"] = [43, 1868, 224, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/30_text_Recent_searches.png
try:
    _c30 = get_crop(30, 288, 168)
    canvas.paste(_c30, (0, 2792), _c30)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_04_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-7/31_clickable_The_Book_of_Mormon.png
try:
    _c31 = get_crop(31, 1440, 179)
    canvas.paste(_c31, (0, 650), _c31)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 650, 1440, 829]
