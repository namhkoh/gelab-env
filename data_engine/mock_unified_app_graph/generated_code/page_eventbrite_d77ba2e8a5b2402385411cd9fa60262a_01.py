# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_01
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3.png
# step_index: 1/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly warm white to match screenshot)
draw.rectangle([(0, 0), canvas.size], fill="#ffffff")

W, H = canvas.size

# --- Status bar (top) ---
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill="#d0d0d0")  # light gray status bar

# subtle inner top line to mimic device bezel
draw.line([(0, status_h), (W, status_h)], fill="#c8c8c8", width=1)

# --- Header / Search area ---
search_top = status_h + 18
search_bottom = search_top + 82
search_left = 72
search_right = W - 72
search_radius = 46

# Outer subtle border for the search field
draw.rounded_rectangle(
    [(search_left-2, search_top-2), (search_right+2, search_bottom+2)],
    radius=search_radius+2,
    fill="#f6f4f8",
    outline=None,
)

# Actual search bar background (white) with faint border
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill="#ffffff",
    outline="#e7e3ec",
)

# Divider below header
divider_y = search_bottom + 18
draw.line([(48, divider_y), (W-48, divider_y)], fill="#f0edf2", width=2)

# --- Section title background (behind "More events you'll love") ---
# subtle top padding area (no text drawn)
title_block_top = divider_y + 18
title_block_bottom = title_block_top + 80
draw.rectangle([(0, title_block_top), (W, title_block_bottom)], fill="#ffffff")

# --- Event list separators and subtle grouped card backgrounds ---
row_x1 = 48
row_x2 = W - 48
row_height = 200
row_gap = 28
first_row_top = title_block_bottom + 18

# Draw a soft drop shadow bar and faint rounded card backgrounds for rows
for i in range(7):
    top = first_row_top + i * (row_height + row_gap)
    bottom = top + row_height
    card_rect = [(row_x1, top), (row_x2, bottom)]
    # subtle shadow/backing rectangle
    shadow_rect = [(row_x1+4, top+6), (row_x2+4, bottom+6)]
    draw.rounded_rectangle(shadow_rect, radius=12, fill="#fbfbfc")
    # main card background (very slightly off-white)
    draw.rounded_rectangle(card_rect, radius=12, fill="#ffffff", outline="#f1edf3")

    # thin separator line below each card
    sep_y = bottom + int(row_gap/2)
    draw.line([(row_x1+8, sep_y), (row_x2-8, sep_y)], fill="#f2eef4", width=1)

# --- A slightly darker full-width background block for a late content area (e.g., image banner) ---
# placed near the lower half like the large image area in the screenshot
banner_top = first_row_top + 5 * (row_height + row_gap) + 40
banner_bottom = banner_top + 220
draw.rectangle([(0, banner_top), (W, banner_bottom)], fill="#faf8fb")

# subtle inner rounded panel centered horizontally (for a featured card background)
feat_left = 48
feat_right = W - 48
feat_top = banner_top + 18
feat_bottom = banner_bottom - 18
draw.rounded_rectangle([(feat_left, feat_top), (feat_right, feat_bottom)], radius=14, fill="#ffffff", outline="#efe9f2")

# --- Bottom navigation bar area ---
nav_h = 120
nav_top = H - nav_h
draw.rectangle([(0, nav_top), (W, H)], fill="#ffffff")

# top divider / shadow for nav bar
draw.line([(0, nav_top), (W, nav_top)], fill="#e6e2e8", width=2)
# subtle soft shadow gradient band above nav (simulated by a thin semi-transparent stripe)
draw.rectangle([(0, nav_top-6), (W, nav_top)], fill="#fbfafb")

# Slight background circle behind center area to suggest raised button area (no icons drawn)
center_x = W // 2
center_y = nav_top - 24
draw.ellipse([(center_x-64, center_y-32), (center_x+64, center_y+32)], fill="#ffffff", outline="#efe6ee")

# --- Additional subtle separators for large content groups ---
# A few more section divider lines to match layout rhythm
for y in [first_row_top + 2*(row_height+row_gap), first_row_top + 4*(row_height+row_gap), banner_bottom + 16]:
    if 0 < y < H:
        draw.line([(48, y), (W-48, y)], fill="#f3eff4", width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/00_icon_New_York.png
try:
    _c0 = get_crop(0, 405, 117)
    canvas.paste(_c0, (518, 2651), _c0)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/01_icon_City.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 886), _c1)
except Exception:
    pass
layout["City,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/02_icon_Q_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/03_icon_New_York.png
try:
    _c3 = get_crop(3, 144, 139)
    canvas.paste(_c3, (1140, 747), _c3)
except Exception:
    pass
layout["New_York"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/04_icon_Conference_Connections.png
try:
    _c4 = get_crop(4, 144, 139)
    canvas.paste(_c4, (1140, 1935), _c4)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/05_icon_New_York.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1284, 1159), _c5)
except Exception:
    pass
layout["New_York"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/06_icon_Conference_Connections.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 2347), _c6)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/07_icon_New_York.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1159), _c7)
except Exception:
    pass
layout["New_York"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/08_icon_New_York.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1284, 747), _c8)
except Exception:
    pass
layout["New_York"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1284, 2347), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/10_icon_Good_Afternoon_New_York.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1678), _c10)
except Exception:
    pass
layout["Good_Afternoon_New_York"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/11_icon_Pier_36.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Pier_36"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1935), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1555), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/14_icon_Medical_Hair_Loss_Therapy_Training.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 886), _c14)
except Exception:
    pass
layout["Medical_Hair_Loss_Therapy"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/15_icon_139_creator_followers.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1282), _c15)
except Exception:
    pass
layout["139_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/16_icon_Free.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1282), _c16)
except Exception:
    pass
layout["Free"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/17_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 490), _c17)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/18_icon_VOSCHINO.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1678), _c18)
except Exception:
    pass
layout["VOSCHINO"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1140, 1555), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/20_icon_ACHT_PART.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["'ACHT_PART"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 58)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/22_icon_6.48.png
try:
    _c22 = get_crop(22, 100, 96)
    canvas.paste(_c22, (43, 123), _c22)
except Exception:
    pass
layout["6.48"] = [43, 123, 143, 219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/23_icon_6.48.png
try:
    _c23 = get_crop(23, 55, 60)
    canvas.paste(_c23, (183, 2), _c23)
except Exception:
    pass
layout["6.48"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 60)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 47, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/26_icon_Free.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 490), _c26)
except Exception:
    pass
layout["Free"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 64, 58)
    canvas.paste(_c27, (1212, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [1212, 4, 1276, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/28_icon_6.48.png
try:
    _c28 = get_crop(28, 58, 61)
    canvas.paste(_c28, (115, 2), _c28)
except Exception:
    pass
layout["6.48"] = [115, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/29_icon_Free.png
try:
    _c29 = get_crop(29, 130, 74)
    canvas.paste(_c29, (244, 560), _c29)
except Exception:
    pass
layout["Free"] = [244, 560, 374, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/30_icon_Q_Search_events.png
try:
    _c30 = get_crop(30, 44, 57)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 55)
    canvas.paste(_c31, (1272, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/32_icon_8_1646_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["8_1646_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/33_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/34_icon_Primary_Ventures_Partners.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["Primary_Ventures_Partners"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/36_icon_HIPHOP_Dancehall_yacht_party_NEW_YORK.png
try:
    _c36 = get_crop(36, 1344, 346)
    canvas.paste(_c36, (48, 2470), _c36)
except Exception:
    pass
layout["HIPHOP_Dancehall_yacht_pa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/37_text_6.48.png
try:
    _c37 = get_crop(37, 89, 43)
    canvas.paste(_c37, (22, 15), _c37)
except Exception:
    pass
layout["6.48"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/39_text_Sat_Apr_27.png
try:
    _c39 = get_crop(39, 195, 43)
    canvas.paste(_c39, (390, 2525), _c39)
except Exception:
    pass
layout["Sat,_Apr_27"] = [390, 2525, 585, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/40_text_1_00_PM_EDT.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["1:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/41_text_PPHc.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["PPHc"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/42_text_ACHT_PART.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (0, 2804), _c42)
except Exception:
    pass
layout["'ACHT_PART"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_01_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-3/44_clickable_More.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (1152, 2804), _c44)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
