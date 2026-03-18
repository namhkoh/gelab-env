# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_01
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3.png
# step_index: 1/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle([(0, 0), canvas.size], fill="#FCFCFD")

# Status bar (top area)
status_h = 90
draw.rectangle([(0, 0), (1440, status_h)], fill="#9E9E9E")
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#E6E6E6", width=1)

# Header / toolbar background (below status bar)
header_top = status_h
header_bottom = 210
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# toolbar bottom divider / shadow
draw.line([(0, header_bottom), (1440, header_bottom)], fill="#F0F0F2", width=2)

# Main content area background (already set by overall fill)
# Draw section cards (rounded rectangles) for each list row
rows = [490, 886, 1282, 1678, 2074, 2470, 2804]
card_x1 = 48
card_x2 = 48 + 1344  # 1392
card_w = card_x2 - card_x1
card_h = 396
card_radius = 14

for y in rows:
    x1 = card_x1
    y1 = y
    x2 = card_x2
    y2 = y1 + card_h

    # subtle shadow (offset)
    shadow_offset = 6
    draw.rounded_rectangle(
        [(x1 + shadow_offset, y1 + shadow_offset), (x2 + shadow_offset, y2 + shadow_offset)],
        radius=card_radius, fill="#F3F4F6"
    )

    # card background
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=card_radius, fill="#FFFFFF", outline="#F4F4F6", width=1)

    # left thumbnail/content background inside card (placeholder background only)
    thumb_w = 180
    thumb_h = 180
    thumb_x1 = x1 + 8
    # vertically center thumbnail within the card
    thumb_y1 = y1 + (card_h - thumb_h) // 2
    thumb_x2 = thumb_x1 + thumb_w
    thumb_y2 = thumb_y1 + thumb_h
    draw.rectangle([(thumb_x1, thumb_y1), (thumb_x2, thumb_y2)], fill="#EDEFF3", outline="#E0E3E8")

    # small colored banner on top-left of thumbnail (simulates tag background)
    banner_w = 64
    banner_h = 28
    b_x1 = thumb_x1 + 8
    b_y1 = thumb_y1 + 8
    draw.rounded_rectangle([(b_x1, b_y1), (b_x1 + banner_w, b_y1 + banner_h)], radius=8, fill="#DCEFF6")

    # right-side subtle divider for each card (vertical spacing indicator)
    draw.line([(x2 + 2, y1 + 12), (x2 + 2, y2 - 12)], fill="#FFFFFF", width=1)

    # bottom separator line between list items
    draw.line([(x1 + 12, y2 - 1), (x2 - 12, y2 - 1)], fill="#EFEFF1", width=1)

# A subtle long divider under the first content block region (visual grouping)
first_group_bottom = rows[1] + card_h
draw.line([(24, first_group_bottom + 12), (1440 - 24, first_group_bottom + 12)], fill="#F5F5F7", width=1)

# Floating location/search suggestion area background near lower center
# (draw only a subtle shadow behind it so pasted control sits naturally)
loc_box = (440, 2600, 1000, 2730)  # area around the floating location control
draw.rounded_rectangle(
    [(loc_box[0] + 6, loc_box[1] + 6), (loc_box[2] + 6, loc_box[3] + 6)],
    radius=40, fill="#EDEEF0"
)

# Bottom navigation bar background
nav_h = 120
nav_top = canvas.size[1] - nav_h
draw.rectangle([(0, nav_top), (1440, canvas.size[1])], fill="#FFFFFF")
# top divider for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#E9E9EB", width=2)

# small instruction: draw faint indicators for empty content edges (visual polish)
# left and right safe area guides (very subtle)
draw.line([(12, header_bottom + 8), (12, canvas.size[1] - nav_h - 8)], fill="#FFFFFF", width=1)
draw.line([(1440 - 12, header_bottom + 8), (1440 - 12, canvas.size[1] - nav_h - 8)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/00_icon_ORK.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["'ORK"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/05_icon_The_DL.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 1951), _c5)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/06_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1282), _c6)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/07_icon_The_DL.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1539), _c7)
except Exception:
    pass
layout["The_DL"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/08_icon_Dtlaict.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 2074), _c8)
except Exception:
    pass
layout["Dtlaict"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/09_icon_The_DL_Rooftop.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 2347), _c9)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 763), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/11_icon_The_DL.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/13_icon_The_DL_Rooftop.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 2347), _c13)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 58)
    canvas.paste(_c14, (183, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1159), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/16_icon_The_DL.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["The_DL"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 55, 56)
    canvas.paste(_c17, (247, 5), _c17)
except Exception:
    pass
layout["icon_17"] = [247, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1159), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1284, 763), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/20_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/21_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1678), _c21)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/22_icon_Ary.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 52)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/24_icon_New_York.png
try:
    _c24 = get_crop(24, 405, 117)
    canvas.paste(_c24, (518, 2651), _c24)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/25_icon_9.11.png
try:
    _c25 = get_crop(25, 93, 101)
    canvas.paste(_c25, (46, 120), _c25)
except Exception:
    pass
layout["9.11"] = [46, 120, 139, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 67, 59)
    canvas.paste(_c26, (1211, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 4, 1278, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 59)
    canvas.paste(_c27, (311, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 4, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/29_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 42, 56)
    canvas.paste(_c30, (1272, 5), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/31_icon_TUmU_5i0.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 31, 48)
    canvas.paste(_c32, (913, 2687), _c32)
except Exception:
    pass
layout["icon_32"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/33_icon_Fireworks_July_4th_Rooftop_Party.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["Fireworks_July_4th_Roofto"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/34_icon_9.11.png
try:
    _c34 = get_crop(34, 52, 58)
    canvas.paste(_c34, (116, 3), _c34)
except Exception:
    pass
layout["9.11"] = [116, 3, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/35_icon_Free.png
try:
    _c35 = get_crop(35, 128, 75)
    canvas.paste(_c35, (245, 1352), _c35)
except Exception:
    pass
layout["Free"] = [245, 1352, 373, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/36_text_9.11.png
try:
    _c36 = get_crop(36, 89, 41)
    canvas.paste(_c36, (20, 17), _c36)
except Exception:
    pass
layout["9.11"] = [20, 17, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/37_text_More_events_you_II_love.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 490), _c37)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/38_text_Sun_Jun_23.png
try:
    _c38 = get_crop(38, 205, 49)
    canvas.paste(_c38, (388, 2554), _c38)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/39_text_3_00_PM_EDT.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/40_text_The_DL_Rooftop.png
try:
    _c40 = get_crop(40, 144, 123)
    canvas.paste(_c40, (1140, 2347), _c40)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/41_text_Ary.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/42_text_The_DL.png
try:
    _c42 = get_crop(42, 115, 38)
    canvas.paste(_c42, (394, 2693), _c42)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/43_text_TUmU_5i0.png
try:
    _c43 = get_crop(43, 405, 117)
    canvas.paste(_c43, (518, 2651), _c43)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_01_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
