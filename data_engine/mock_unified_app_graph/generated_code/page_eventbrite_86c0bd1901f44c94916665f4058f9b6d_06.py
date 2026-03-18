# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_06
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8.png
# step_index: 6/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall subtle off-white background
draw.rectangle([(0, 0), canvas.size], fill="#f7f7f9")

# Status bar (top system bar)
status_bar_h = 56
draw.rectangle([(0, 0), (1440, status_bar_h)], fill="#bdbdbd")

# Header / toolbar area under status bar
header_top = status_bar_h
header_bottom = 170
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# header bottom divider
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#e6e6ea", width=1)

# Content area main background (slightly warmer white to separate from header)
content_top = header_bottom + 24
draw.rectangle([(0, content_top), (1440, 2960)], fill="#ffffff")

# Section title background band (behind "More events you'll love" area)
section_band_y = 430
draw.rectangle([(0, section_band_y), (1440, section_band_y + 80)], fill="#ffffff")

# Define event row card specs from detected positions (do NOT draw text/icons)
rows = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)
]

# Draw subtle card backgrounds and separators for each event row
for (x, y, w, h) in rows:
    rect_coords = (x, y, x + w, y + h)
    # Slight drop shadow band (very subtle)
    shadow_y1 = y + h + 6
    shadow_y2 = shadow_y1 + 2
    draw.rectangle([(x + 8, shadow_y1), (x + w - 8, shadow_y2)], fill="#f1f1f3")
    # Card background with soft border
    draw.rounded_rectangle(rect_coords, radius=18, fill="#ffffff", outline="#f0eff4", width=1)

    # Thin separator line above each card for visual separation (except first)
    sep_y = y - 18
    if sep_y > content_top:
        draw.line([(x + 8, sep_y), (x + w - 8, sep_y)], fill="#f3f3f5", width=1)

# Large divider lines between groups (every 2 cards) to mimic subtle grouping
group_dividers = [ (48, 1220), (48, 2008) ]
for gx, gy in group_dividers:
    draw.line([(gx, gy), (gx + 1344, gy)], fill="#eeeeee", width=1)

# Floating-ish background for the city selector area (do NOT draw the actual pill/label)
# Draw an ambient shadow where the floating pill will appear so pasted pill appears integrated
pill_shadow_box = (420, 2628, 1020, 2710)  # shadow area under the floating control
draw.rectangle([(pill_shadow_box[0], pill_shadow_box[1]), (pill_shadow_box[2], pill_shadow_box[3])], fill="#fbfbfc")

# Bottom navigation bar background and top divider/shadow
nav_h = 156
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# soft top divider
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#e9e9ec", width=1)
# faint nav shadow above the divider
draw.line([(24, nav_top+1), (1440-24, nav_top+1)], fill="#f6f6f8", width=1)

# Subtle left edge and right edge padding guidelines (visual rails) - very faint
draw.line([(48, 0), (48, 2960)], fill="#fbfbfb", width=1)
draw.line([(1440-48, 0), (1440-48, 2960)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/00_icon_NDIE_DANCEPA.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/01_icon_Ibaigktsinel.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1282), _c1)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/03_icon_FRIDAY.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 2074), _c3)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/05_icon_NDIE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1555), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 65)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/09_icon_Afliccion_Perdida_y.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 2347), _c9)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 1555), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1935), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/13_icon_Sylmai.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/14_icon_Club_Decades.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1143), _c14)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1143), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/16_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 490), _c16)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 763), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/18_icon_59_creator_followers.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1678), _c18)
except Exception:
    pass
layout["59_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/19_icon_8_4717_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 97, 60)
    canvas.paste(_c20, (1216, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [1216, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 59)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/23_icon_Overflow_menu_button.png
try:
    _c23 = get_crop(23, 144, 123)
    canvas.paste(_c23, (1284, 763), _c23)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/24_icon_7.13.png
try:
    _c24 = get_crop(24, 57, 61)
    canvas.paste(_c24, (182, 2), _c24)
except Exception:
    pass
layout["7.13"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 52, 60)
    canvas.paste(_c25, (247, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/26_icon_7.13.png
try:
    _c26 = get_crop(26, 100, 97)
    canvas.paste(_c26, (42, 123), _c26)
except Exception:
    pass
layout["7.13"] = [42, 123, 142, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 53)
    canvas.paste(_c27, (1321, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/28_icon_8_21119_creator_followers.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1282), _c28)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/29_icon_Public_House_Los_Angeles_CA.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 490), _c29)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/30_icon_7.13.png
try:
    _c30 = get_crop(30, 58, 62)
    canvas.paste(_c30, (115, 1), _c30)
except Exception:
    pass
layout["7.13"] = [115, 1, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/32_icon_Free.png
try:
    _c32 = get_crop(32, 1344, 346)
    canvas.paste(_c32, (48, 2470), _c32)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/33_icon_9.30_PM_PDT.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 886), _c33)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/34_icon_Free.png
try:
    _c34 = get_crop(34, 127, 73)
    canvas.paste(_c34, (247, 1749), _c34)
except Exception:
    pass
layout["Free"] = [247, 1749, 374, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/35_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/37_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/38_icon_5.30_PM_PDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/39_icon_31_creator_followers.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (576, 2804), _c39)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/40_text_7.13.png
try:
    _c40 = get_crop(40, 91, 43)
    canvas.paste(_c40, (20, 15), _c40)
except Exception:
    pass
layout["7.13"] = [20, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/42_text_Mon_May_13.png
try:
    _c42 = get_crop(42, 222, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/43_text_5.30_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_06_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-8/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
