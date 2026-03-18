# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_06
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8.png
# step_index: 6/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# 1) Overall page background (subtle warm off-white)
draw.rectangle([(0, 0), (W, H)], fill="#FBFAFC")

# 2) Status bar at top (~50px). Use slightly darker grey band.
status_h = 68
draw.rectangle([(0, 0), (W, status_h)], fill="#BDBDBD")

# Add a subtle inner highlight at bottom of status bar
draw.line([(0, status_h), (W, status_h)], fill="#D6D6D6", width=1)

# 3) Header / toolbar area background under status bar.
# Keep it light (white) with a soft divider below.
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill="#FFFFFF")

# Subtle bottom divider under header
divider_y = header_top + header_h
draw.line([(24, divider_y), (W - 24, divider_y)], fill="#EFEFF0", width=1)

# 4) Main content area - a very light subtle column to hold cards (centered padding)
content_x = 48
content_w = W - content_x * 2

# Draw rounded "card" backgrounds behind each listed event group.
# Use positions inferred from UI: y positions for the list items.
item_ys = [490, 886, 1282, 1678, 2074, 2470]
item_hs = [396, 396, 396, 396, 396, 346]  # heights from detection
card_radius = 12

for y, h in zip(item_ys, item_hs):
    x = content_x
    right = x + content_w
    bottom = y + h

    # Slight drop-shadow band (subtle thin line) above and below to separate from background.
    # Top shadow line
    draw.line([(x + 2, y), (right - 2, y)], fill="#F6F5F7", width=1)
    # Card background (white) and soft border
    draw.rounded_rectangle(
        [(x, y), (right, bottom)],
        radius=card_radius,
        fill="#FFFFFF",
        outline="#F0EEF2"
    )
    # Bottom separator line to enhance separation without drawing content
    draw.line([(x + 8, bottom), (right - 8, bottom)], fill="#F4F3F6", width=1)

    # Left thumbnail placeholder as a neutral block behind actual images (subtle, non-distracting).
    thumb_size = min(160, int(h * 0.8))
    thumb_x = x + 0
    thumb_y = y + (h - thumb_size) // 2
    draw.rounded_rectangle(
        [(thumb_x, thumb_y), (thumb_x + thumb_size, thumb_y + thumb_size)],
        radius=8,
        fill="#F7F7F9",
        outline="#EFEFF1"
    )

# 5) Section header area (big heading region). Draw its background accent (no text).
title_region_top = 420
title_region_h = 80
draw.rectangle([(content_x, title_region_top), (content_x + content_w, title_region_top + title_region_h)], fill="#FBFAFC")
# subtle underline under the title region
draw.line([(content_x + 2, title_region_top + title_region_h), (content_x + content_w - 2, title_region_top + title_region_h)], fill="#F0EFF2", width=1)

# 6) Floating separators for visual rhythm between cards (vertical spacing guides)
for y, h in zip(item_ys, item_hs):
    # light vertical guide at the right edge of thumbnails (not a UI element, just structure)
    thumb_right_x = content_x + 160 + 12
    draw.line([(thumb_right_x, y + 12), (thumb_right_x, y + h - 12)], fill="#FAF9FB", width=1)

# 7) Bottom navigation bar area
nav_h = 120
nav_top = H - nav_h
draw.rectangle([(0, nav_top), (W, H)], fill="#FFFFFF")
# top divider for nav
draw.line([(16, nav_top), (W - 16, nav_top)], fill="#EDECF0", width=1)
# slight shadow under the nav bar to lift it
draw.line([(16, nav_top + 1), (W - 16, nav_top + 1)], fill="#F6F5F7", width=1)

# 8) Add subtle active accent spot on left side of bottom nav (structural only, no icon duplicates).
accent_w = 56
accent_h = 6
accent_x = 56
accent_y = nav_top + 12
draw.rounded_rectangle(
    [(accent_x, accent_y), (accent_x + accent_w, accent_y + accent_h)],
    radius=6,
    fill="#FFF2EC"
)

# 9) Minor global separators (to echo the original layout's subtle rules)
# Gentle vertical edge margins
draw.line([(content_x, header_top + 8), (content_x, H - nav_h - 8)], fill="#FBF9FB", width=1)
draw.line([(W - content_x, header_top + 8), (W - content_x, H - nav_h - 8)], fill="#FBF9FB", width=1)

# Done drawing structural/background elements.
# Note: All textual labels, icons, and interactive buttons will be pasted on top at their detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/02_icon_San_Francisco.png
try:
    _c2 = get_crop(2, 495, 117)
    canvas.paste(_c2, (473, 2651), _c2)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/04_icon_QUEEN.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 2074), _c4)
except Exception:
    pass
layout["QUEEN"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/05_icon_Sat.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 65)
    canvas.paste(_c6, (1154, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1154, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 747), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/08_icon_3600.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (288, 2804), _c8)
except Exception:
    pass
layout["3600"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/09_icon_City.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1539), _c9)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 747), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/11_icon_City.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1539), _c11)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1140, 1951), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/13_icon_7.47.png
try:
    _c13 = get_crop(13, 108, 102)
    canvas.paste(_c13, (38, 121), _c13)
except Exception:
    pass
layout["7.47"] = [38, 121, 146, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1951), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/15_icon_Spring-Zing_Happy.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 2331), _c15)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1143), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1284, 2331), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/18_icon_City.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 1143), _c18)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/19_icon_7.47.png
try:
    _c19 = get_crop(19, 54, 60)
    canvas.paste(_c19, (184, 2), _c19)
except Exception:
    pass
layout["7.47"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/20_icon_PDO_Thread_Training.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1282), _c20)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/21_icon_RIEF_MEDICIN.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1282), _c21)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 59, 58)
    canvas.paste(_c22, (313, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [313, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 97, 60)
    canvas.paste(_c23, (1216, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [1216, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 58)
    canvas.paste(_c24, (248, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 3, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 48, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/26_icon_7.47.png
try:
    _c26 = get_crop(26, 58, 59)
    canvas.paste(_c26, (114, 3), _c26)
except Exception:
    pass
layout["7.47"] = [114, 3, 172, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/27_icon_Register_Nowl.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Register_Nowl"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/28_icon_8_29_creator_followers.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 886), _c28)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/29_icon_59_creator_followers.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 490), _c29)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/30_icon_Spring-Zing_Happy_Hour.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/31_icon_Free.png
try:
    _c31 = get_crop(31, 125, 73)
    canvas.paste(_c31, (248, 561), _c31)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/32_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 44, 56)
    canvas.paste(_c33, (385, 6), _c33)
except Exception:
    pass
layout["icon_33"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/34_icon_Queen_of_Indies_2024.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["Queen_of_Indies_2024"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/35_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 886), _c35)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/37_icon_7_00_PM_PDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["7:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/38_icon_7.47.png
try:
    _c38 = get_crop(38, 89, 58)
    canvas.paste(_c38, (17, 5), _c38)
except Exception:
    pass
layout["7.47"] = [17, 5, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/40_text_1_Buchanan_St.png
try:
    _c40 = get_crop(40, 235, 40)
    canvas.paste(_c40, (397, 1930), _c40)
except Exception:
    pass
layout["1_Buchanan_St"] = [397, 1930, 632, 1970]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/41_text_8_100_creator_followers.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 1678), _c41)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/42_text_Mon.png
try:
    _c42 = get_crop(42, 92, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,"] = [393, 2525, 485, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/43_text_13_S_00_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["13_+_S:00_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/44_text_Out_in_Tech_SF_Bay_Area.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["Out_in_Tech_SF_Bay_Area"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/45_clickable_Favorites.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_06_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-8/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
