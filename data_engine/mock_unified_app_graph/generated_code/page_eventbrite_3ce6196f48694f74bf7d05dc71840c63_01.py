# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_01
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3.png
# step_index: 1/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), canvas.size], fill="#FBFBFC")

# Status bar (approx 50px high)
status_h = 72
draw.rectangle([(0, 0), (canvas.size[0], status_h)], fill="#BFBFBF")

# Header area under status bar (toolbar / search background region)
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (canvas.size[0], header_top + header_h)], fill="#FFFFFF")
# subtle bottom divider under header
draw.line([(32, header_top + header_h), (canvas.size[0] - 32, header_top + header_h)], fill="#ECECEC", width=1)

# Search bar background (leave icons/text out; only draw the rounded background)
# Using detected search bar position: pos=(195,93) size=1179x144
search_x, search_y = 195, 93
search_w, search_h = 1179, 144
draw.rounded_rectangle(
    [(search_x, search_y), (search_x + search_w, search_y + search_h)],
    radius=36,
    fill="#F5F6FA",
    outline="#E6E6EA",
    width=2
)

# Main content area (already white) - draw a subtle left title area spacer (no text)
# Draw a large left margin area to visually separate content (very subtle)
draw.rectangle([(0, header_top + header_h), (48, canvas.size[1])], fill="#FBFBFC")

# Section dividers and row card backgrounds
# Approximate Y positions for the list rows based on visual layout
row_ys = [560, 886, 1282, 1678, 2074, 2470]
card_left = 32
card_right = canvas.size[0] - 32
card_width = card_right - card_left
card_h = 220

# Colors for image placeholders (varied so they aren't identical)
thumb_colors = ["#E76F51", "#2A9D8F", "#E9C46A", "#8ECAE6", "#F4A261", "#9C89B8"]

for i, y in enumerate(row_ys):
    top = y - 16
    bottom = top + card_h
    # card background (subtle white card with tiny border)
    draw.rounded_rectangle(
        [(card_left, top), (card_right, bottom)],
        radius=14,
        fill="#FFFFFF",
        outline="#F0F0F2",
        width=1
    )
    # thumbnail placeholder on the left (rounded square)
    thumb_x = 48
    thumb_y = top + 20
    thumb_w = 180
    thumb_h = 180
    draw.rounded_rectangle(
        [(thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h)],
        radius=8,
        fill=thumb_colors[i % len(thumb_colors)],
        outline="#E6E6E9",
        width=1
    )
    # subtle separator line below each card
    sep_y = bottom + 12
    draw.line([(card_left + 8, sep_y), (card_right - 8, sep_y)], fill="#F2F2F4", width=1)

# Additional content area backgrounds (e.g., colored banner for one of the rows)
# Draw a muted banner behind one lower row's thumbnail area (no text)
banner_top = row_ys[-1] + 80
banner_height = 120
draw.rectangle([(48, banner_top), (48 + 180, banner_top + banner_height)], fill="#FFF6E6", outline="#F0E6D8")

# Floating location pill background near lower center (background only, no icon/text)
pill_w, pill_h = 495, 117  # matches detected size for location pill background
pill_x = int((canvas.size[0] - pill_w) / 2)  # center horizontally
pill_y = 2651  # use detected y
# subtle shadow (a faint darker rounded rectangle slightly offset)
shadow_offset = 6
draw.rounded_rectangle(
    [(pill_x + shadow_offset, pill_y + shadow_offset), (pill_x + pill_w + shadow_offset, pill_y + pill_h + shadow_offset)],
    radius=40,
    fill="#EEEEF0"
)
# white pill on top
draw.rounded_rectangle(
    [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)],
    radius=40,
    fill="#FFFFFF",
    outline="#E6E6EA",
    width=1
)

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = canvas.size[1] - nav_h
draw.line([(0, nav_top), (canvas.size[0], nav_top)], fill="#EDEDED", width=1)
draw.rectangle([(0, nav_top), (canvas.size[0], canvas.size[1])], fill="#FFFFFF")

# Small indicator dots/separators in nav (background shapes only, no icons/text)
nav_icon_xs = [72, 360, 648, 936, 1224]
for x in nav_icon_xs:
    # draw faint circular hit areas for icons (no icons themselves)
    r = 36
    draw.ellipse([(x - r, nav_top + 24 - r), (x + r, nav_top + 24 + r)], outline="#F2F2F4", width=1, fill="#FFFFFF")

# Final subtle top shadow under status bar to separate it slightly
draw.line([(0, status_h), (canvas.size[0], status_h)], fill="#CFCFCF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/03_icon_Sat.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/04_icon_San_Francisco.png
try:
    _c4 = get_crop(4, 495, 117)
    canvas.paste(_c4, (473, 2651), _c4)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 139)
    canvas.paste(_c5, (1140, 747), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/06_icon_Reggaeton.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 2331), _c6)
except Exception:
    pass
layout["Reggaeton__"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/07_icon_City.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1539), _c7)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 1951), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/10_icon_RIEF_MEDICIN.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1282), _c10)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/11_icon_Bissa.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Bissa}"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/13_icon_City.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1539), _c13)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/14_icon_7.23.png
try:
    _c14 = get_crop(14, 111, 103)
    canvas.paste(_c14, (37, 120), _c14)
except Exception:
    pass
layout["7.23"] = [37, 120, 148, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/15_icon_City.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 1143), _c15)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1143), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/17_icon_7.23.png
try:
    _c17 = get_crop(17, 54, 60)
    canvas.paste(_c17, (184, 2), _c17)
except Exception:
    pass
layout["7.23"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/18_icon_SatvaonG.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["SatvaonG"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/19_icon_PDO_Thread_Training.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1282), _c19)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 58, 57)
    canvas.paste(_c20, (313, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/21_icon_Reggaeton.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 2331), _c21)
except Exception:
    pass
layout["Reggaeton__"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 47, 58)
    canvas.paste(_c22, (250, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [250, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 53)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/24_icon_7.23.png
try:
    _c24 = get_crop(24, 58, 59)
    canvas.paste(_c24, (115, 3), _c24)
except Exception:
    pass
layout["7.23"] = [115, 3, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/25_icon_8_29_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 886), _c25)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/26_icon_59_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 490), _c26)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 57, 58)
    canvas.paste(_c27, (1213, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [1213, 4, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/28_icon_Free.png
try:
    _c28 = get_crop(28, 125, 73)
    canvas.paste(_c28, (248, 561), _c28)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 41, 55)
    canvas.paste(_c29, (1272, 6), _c29)
except Exception:
    pass
layout["icon_29"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/30_icon_8_100_creator_followers.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1678), _c30)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/31_icon_Salsa.png
try:
    _c31 = get_crop(31, 1344, 346)
    canvas.paste(_c31, (48, 2470), _c31)
except Exception:
    pass
layout["Salsa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/32_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 43, 55)
    canvas.paste(_c33, (385, 7), _c33)
except Exception:
    pass
layout["icon_33"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/34_icon_8_00_AM_PDT.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["8:00_AM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/35_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 886), _c35)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/36_icon_Yggae.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Yggae"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/37_icon_Yggae.png
try:
    _c37 = get_crop(37, 151, 68)
    canvas.paste(_c37, (933, 2644), _c37)
except Exception:
    pass
layout["Yggae"] = [933, 2644, 1084, 2712]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/38_icon_8_29_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 886), _c38)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/39_text_7.23.png
try:
    _c39 = get_crop(39, 91, 45)
    canvas.paste(_c39, (20, 15), _c39)
except Exception:
    pass
layout["7.23"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/41_text_Aanonananal.png
try:
    _c41 = get_crop(41, 194, 14)
    canvas.paste(_c41, (98, 2542), _c41)
except Exception:
    pass
layout["Aanonananal"] = [98, 2542, 292, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/42_text_Sat_May_4_._10_00_AM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["Sat,_May_4_._10:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/43_text_hellaGood.png
try:
    _c43 = get_crop(43, 186, 41)
    canvas.paste(_c43, (101, 2556), _c43)
except Exception:
    pass
layout["hellaGood"] = [101, 2556, 287, 2597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/44_text_ssan.png
try:
    _c44 = get_crop(44, 25, 9)
    canvas.paste(_c44, (252, 2636), _c44)
except Exception:
    pass
layout["ssan"] = [252, 2636, 277, 2645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/45_text_featuring-.png
try:
    _c45 = get_crop(45, 43, 15)
    canvas.paste(_c45, (215, 2650), _c45)
except Exception:
    pass
layout["'featuring-"] = [215, 2650, 258, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/46_text_Jah_Wafeidk_SHELTER.png
try:
    _c46 = get_crop(46, 129, 13)
    canvas.paste(_c46, (142, 2702), _c46)
except Exception:
    pass
layout["Jah_Wafeidk_SHELTER"] = [142, 2702, 271, 2715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/47_text_DJGREENB_DJAGANA_DJMALIIGZ.png
try:
    _c47 = get_crop(47, 215, 18)
    canvas.paste(_c47, (91, 2718), _c47)
except Exception:
    pass
layout["DJGREENB_DJAGANA_DJMALIIG"] = [91, 2718, 306, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/48_text_Log_AETa.png
try:
    _c48 = get_crop(48, 41, 6)
    canvas.paste(_c48, (111, 2738), _c48)
except Exception:
    pass
layout["Log__AETa"] = [111, 2738, 152, 2744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/49_text_atrobcats.png
try:
    _c49 = get_crop(49, 43, 13)
    canvas.paste(_c49, (156, 2746), _c49)
except Exception:
    pass
layout["atrobcats"] = [156, 2746, 199, 2759]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/50_text_nalceuani.png
try:
    _c50 = get_crop(50, 37, 7)
    canvas.paste(_c50, (212, 2742), _c50)
except Exception:
    pass
layout["nalceuani"] = [212, 2742, 249, 2749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/51_text_Lrocaa_Rnrae.png
try:
    _c51 = get_crop(51, 53, 9)
    canvas.paste(_c51, (240, 2763), _c51)
except Exception:
    pass
layout["Lrocaa_Rnrae"] = [240, 2763, 293, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/52_text_SatvaonG.png
try:
    _c52 = get_crop(52, 60, 29)
    canvas.paste(_c52, (92, 2761), _c52)
except Exception:
    pass
layout["SatvaonG"] = [92, 2761, 152, 2790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/53_text_t0ph.png
try:
    _c53 = get_crop(53, 32, 15)
    canvas.paste(_c53, (158, 2767), _c53)
except Exception:
    pass
layout["t0ph"] = [158, 2767, 190, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/54_text_Z44.png
try:
    _c54 = get_crop(54, 23, 15)
    canvas.paste(_c54, (197, 2767), _c54)
except Exception:
    pass
layout["Z44"] = [197, 2767, 220, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/55_text_71J_Nissiom_St.st.png
try:
    _c55 = get_crop(55, 74, 13)
    canvas.paste(_c55, (232, 2774), _c55)
except Exception:
    pass
layout["{71J_Nissiom_St.st"] = [232, 2774, 306, 2787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/56_clickable_Favorites.png
try:
    _c56 = get_crop(56, 288, 156)
    canvas.paste(_c56, (576, 2804), _c56)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_01_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-3/57_clickable_More.png
try:
    _c57 = get_crop(57, 288, 156)
    canvas.paste(_c57, (1152, 2804), _c57)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
