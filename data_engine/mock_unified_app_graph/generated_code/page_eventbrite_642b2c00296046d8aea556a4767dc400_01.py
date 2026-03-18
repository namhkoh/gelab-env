# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_01
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3.png
# step_index: 1/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle([(0, 0), (1440, 2960)], fill=(249, 249, 249))

# Status bar area (top ~56px)
draw.rectangle([(0, 0), (1440, 56)], fill=(160, 160, 160))
# subtle divider under status/search area
draw.line([(0, 56), (1440, 56)], fill=(220, 220, 222), width=1)

# Header/Search area background band (keeps contrast but doesn't draw the search control itself)
draw.rectangle([(0, 56), (1440, 180)], fill=(249, 249, 249))
draw.line([(48, 180), (1392, 180)], fill=(235, 235, 238), width=1)

# Event list card rows (rounded white cards with soft shadow)
rows = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)
]

for (x, y, w, h) in rows:
    x2 = x + w
    y2 = y + h
    radius = 14

    # shadow (subtle)
    shadow_offset = 6
    draw.rounded_rectangle(
        [(x, y + shadow_offset), (x2, y2 + shadow_offset)],
        radius=radius,
        fill=(242, 242, 245),
        outline=None
    )

    # card background
    draw.rounded_rectangle(
        [(x, y), (x2, y2)],
        radius=radius,
        fill=(255, 255, 255),
        outline=(230, 230, 235)
    )

    # inner subtle separator under each card (for spacing between list items)
    sep_y = y2 + 24
    draw.line([(x + 12, sep_y), (x2 - 12, sep_y)], fill=(245, 245, 246), width=1)

# Large left thumbnail placeholder backgrounds (behind thumbnail images that will be pasted)
# Keep these as subtle colored blocks so the pasted thumbnails sit on consistent background
thumbnail_w = 224
thumb_x = 48
for (x, y, w, h) in rows:
    thumb_y = y + 24
    draw.rounded_rectangle(
        [(thumb_x, thumb_y), (thumb_x + thumbnail_w, thumb_y + thumbnail_w)],
        radius=8,
        fill=(245, 245, 247),
        outline=(235, 235, 238)
    )

# Divider lines between the header/title region and the list
draw.line([(48, 240), (1392, 240)], fill=(230, 230, 234), width=1)
draw.line([(48, 320), (1392, 320)], fill=(247, 247, 248), width=1)

# Bottom navigation bar background (keeps area clean for icons which will be pasted)
bottom_bar_top = 2800
draw.rectangle([(0, bottom_bar_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill=(230, 230, 234), width=1)

# Floating location pill background area placeholder (do not draw the pill content)
# Draw only a very subtle shadow behind where the pill will appear so pasted pill sits naturally.
pill_shadow_box = (420, 2580, 1020, 2700)
draw.rounded_rectangle([ (pill_shadow_box[0], pill_shadow_box[1]+6), (pill_shadow_box[2], pill_shadow_box[3]+6) ],
                       radius=32, fill=(240,240,243))

# subtle overall vignette line separators for visual hierarchy (do not draw any icons/text)
for y in [460, 844, 1240, 1636, 2032, 2430]:
    draw.line([(48, y), (1392, y)], fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/00_icon_ORK.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["'ORK"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/05_icon_The_DL.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 1951), _c5)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/06_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1282), _c6)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/07_icon_The_DL.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1539), _c7)
except Exception:
    pass
layout["The_DL"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/08_icon_Dtlaict.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 2074), _c8)
except Exception:
    pass
layout["Dtlaict"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/09_icon_The_DL_Rooftop.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 2347), _c9)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 763), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/11_icon_The_DL.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/13_icon_The_DL_Rooftop.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 2347), _c13)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1159), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/15_icon_The_DL.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["The_DL"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1140, 1159), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1284, 763), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/19_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/20_icon_9.09.png
try:
    _c20 = get_crop(20, 52, 58)
    canvas.paste(_c20, (183, 3), _c20)
except Exception:
    pass
layout["9.09"] = [183, 3, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/21_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1678), _c21)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/22_icon_Ary.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 52)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/24_icon_New_York.png
try:
    _c24 = get_crop(24, 405, 117)
    canvas.paste(_c24, (518, 2651), _c24)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/25_icon_9.09.png
try:
    _c25 = get_crop(25, 94, 102)
    canvas.paste(_c25, (46, 119), _c25)
except Exception:
    pass
layout["9.09"] = [46, 119, 140, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 65, 59)
    canvas.paste(_c26, (1211, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 4, 1276, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/29_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 42, 56)
    canvas.paste(_c30, (1272, 5), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/31_icon_TUmU_5i0.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 31, 48)
    canvas.paste(_c32, (913, 2687), _c32)
except Exception:
    pass
layout["icon_32"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/33_icon_Fireworks_July_4th_Rooftop_Party.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["Fireworks_July_4th_Roofto"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/34_icon_Free.png
try:
    _c34 = get_crop(34, 128, 75)
    canvas.paste(_c34, (245, 1352), _c34)
except Exception:
    pass
layout["Free"] = [245, 1352, 373, 1427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/35_text_9.09.png
try:
    _c35 = get_crop(35, 91, 43)
    canvas.paste(_c35, (20, 17), _c35)
except Exception:
    pass
layout["9.09"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/37_text_Sun_Jun_23.png
try:
    _c37 = get_crop(37, 205, 49)
    canvas.paste(_c37, (388, 2554), _c37)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/38_text_3_00_PM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/39_text_The_DL_Rooftop.png
try:
    _c39 = get_crop(39, 144, 123)
    canvas.paste(_c39, (1140, 2347), _c39)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/40_text_Ary.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/41_text_The_DL.png
try:
    _c41 = get_crop(41, 115, 38)
    canvas.paste(_c41, (394, 2693), _c41)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/42_text_TUmU_5i0.png
try:
    _c42 = get_crop(42, 405, 117)
    canvas.paste(_c42, (518, 2651), _c42)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/43_clickable_Tickets.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (864, 2804), _c43)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_01_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-3/44_clickable_More.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (1152, 2804), _c44)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
