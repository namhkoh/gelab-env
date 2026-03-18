# page_id: page_eventbrite_92c22920a83749c994864397a370a984_01
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-3.png
# step_index: 1/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for Eventbrite-like UI
# Uses provided canvas (1440x2960) and draw (ImageDraw)

# Base background (dominant color: white)
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# Top status bar (approx ~56px high) - light gray like Android status bar
status_h = 56
draw.rectangle([0, 0, 1440, status_h], fill=(200, 200, 200))
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(185, 185, 185), width=1)

# Header/toolbar area (below status bar)
toolbar_top = status_h
toolbar_bottom = 170
draw.rectangle([0, toolbar_top, 1440, toolbar_bottom], fill=(255, 255, 255))
# faint shadow under toolbar to separate from content
draw.line([(0, toolbar_bottom), (1440, toolbar_bottom)], fill=(235, 235, 235), width=2)

# Slight inner highlight band to mimic subtle toolbar elevation
draw.rectangle([0, toolbar_bottom-6, 1440, toolbar_bottom], fill=(247, 247, 248))

# Section area background (the main scrolling area stays white, but we add subtle card backgrounds)
# Event row card backgrounds: using positions derived from detected elements (left margin 48, width 1344)
card_fill = (250, 250, 252)   # very slight off-white to distinguish rows
card_outline = (235, 235, 238) # subtle outline/shadow

rows = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346),
    (48, 2804, 48 + 1344, 2804 + 156)
]

for (x1, y1, x2, y2) in rows:
    # Draw rounded card background for each event row
    try:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=card_fill, outline=card_outline, width=1)
    except Exception:
        # Fallback if rounded_rectangle unavailable
        draw.rectangle([x1, y1, x2, y2], fill=card_fill, outline=card_outline)

    # subtle drop shadow line under each card to separate rows from background
    shadow_y = y2 + 6
    if shadow_y < 2960:
        draw.line([(x1 + 8, shadow_y), (x2 - 8, shadow_y)], fill=(242, 242, 244), width=2)

    # light separator between rows (closer to content edge)
    sep_y = y2 + 18
    if sep_y < 2960:
        draw.line([(48, sep_y), (1440 - 48, sep_y)], fill=(245, 245, 246), width=1)

# Additional subtle horizontal separators for the content flow (between major blocks)
separator_positions = [430, 800, 1180, 1580, 1980, 2380]
for sy in separator_positions:
    draw.line([(48, sy), (1440 - 48, sy)], fill=(250, 250, 251), width=1)

# Floating location / filter pill background area (structural backdrop)
# Draw a faint translucent rounded area where the "Los Angeles" pill will appear (so the pasted pill sits on a matching background)
pill_x1, pill_y1 = 432, 2616
pill_x2, pill_y2 = 1008, 2688
try:
    draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=28, fill=(255, 255, 255), outline=(225,225,230), width=1)
except Exception:
    draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=(255,255,255), outline=(225,225,230))

# Bottom navigation bar background (structural only)
nav_top = 2800
nav_bottom = 2960
draw.rectangle([0, nav_top, 1440, nav_bottom], fill=(255, 255, 255))
# top border of nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 225), width=2)

# small elevation hint under nav (very subtle)
draw.line([(0, nav_top + 3), (1440, nav_top + 3)], fill=(245, 245, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/00_icon_NDIE_DANCEPA.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/01_icon_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/02_icon_Ibaigktsinel.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1282), _c2)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/03_icon_FRIDAY.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 2074), _c3)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 139)
    canvas.paste(_c4, (1140, 1935), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 1555), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/06_icon_Los_Angeles.png
try:
    _c6 = get_crop(6, 456, 117)
    canvas.paste(_c6, (492, 2651), _c6)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/07_icon_NDIE.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 886), _c7)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 2347), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1284, 1555), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 2347), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/12_icon_Club_Decades.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1140, 1143), _c12)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/13_icon_Grand.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Grand"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1143), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/15_icon_4721_creator_followers.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 886), _c15)
except Exception:
    pass
layout["4721_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1140, 763), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/17_icon_8_60_creator_followers.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1678), _c17)
except Exception:
    pass
layout["8_60_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 60, 59)
    canvas.paste(_c18, (312, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/19_icon_4.59.png
try:
    _c19 = get_crop(19, 57, 61)
    canvas.paste(_c19, (182, 2), _c19)
except Exception:
    pass
layout["4.59"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/20_icon_Lmy_Danse_Gala_wl_di_pel.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Lmy_Danse_Gala_wl_di_pel"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 52, 60)
    canvas.paste(_c22, (247, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/23_icon_4.59.png
try:
    _c23 = get_crop(23, 102, 99)
    canvas.paste(_c23, (41, 122), _c23)
except Exception:
    pass
layout["4.59"] = [41, 122, 143, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 52)
    canvas.paste(_c24, (1321, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/25_icon_8_21125_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1282), _c25)
except Exception:
    pass
layout["8_21125_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/26_icon_Public_House_Los_Angeles_CA.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 490), _c26)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/27_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 85, 57)
    canvas.paste(_c28, (1212, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 5, 1297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/29_icon_4.59.png
try:
    _c29 = get_crop(29, 60, 62)
    canvas.paste(_c29, (114, 1), _c29)
except Exception:
    pass
layout["4.59"] = [114, 1, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/30_icon_Bar_Franca.png
try:
    _c30 = get_crop(30, 177, 54)
    canvas.paste(_c30, (390, 2318), _c30)
except Exception:
    pass
layout["Bar_Franca"] = [390, 2318, 567, 2372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/32_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/33_icon_9.30_PM_PDT.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 886), _c33)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/34_icon_Free.png
try:
    _c34 = get_crop(34, 127, 73)
    canvas.paste(_c34, (247, 1749), _c34)
except Exception:
    pass
layout["Free"] = [247, 1749, 374, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 41, 55)
    canvas.paste(_c35, (1272, 6), _c35)
except Exception:
    pass
layout["icon_35"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/36_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/37_icon_Danse.png
try:
    _c37 = get_crop(37, 163, 65)
    canvas.paste(_c37, (911, 2643), _c37)
except Exception:
    pass
layout["Danse"] = [911, 2643, 1074, 2708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/38_icon_BIZARRE_LOVE_TRIANGLE_New_Wave_Post.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2074), _c38)
except Exception:
    pass
layout["BIZARRE_LOVE_TRIANGLE:_Ne"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/39_text_4.59.png
try:
    _c39 = get_crop(39, 89, 43)
    canvas.paste(_c39, (22, 17), _c39)
except Exception:
    pass
layout["4.59"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/41_text_Sat_Apr_27.png
try:
    _c41 = get_crop(41, 195, 43)
    canvas.paste(_c41, (390, 2525), _c41)
except Exception:
    pass
layout["Sat,_Apr_27"] = [390, 2525, 585, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/42_text_9_30_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["9:30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/43_text_le_for_1_FREE_PAIR_of_ticke.png
try:
    _c43 = get_crop(43, 307, 27)
    canvas.paste(_c43, (46, 2599), _c43)
except Exception:
    pass
layout["le_for_1_FREE_PAIR_of_tic"] = [46, 2599, 353, 2626]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/44_text_arrive_before_10.30_to_ENT.png
try:
    _c44 = get_crop(44, 290, 27)
    canvas.paste(_c44, (58, 2645), _c44)
except Exception:
    pass
layout["arrive_before_10.30_to_EN"] = [58, 2645, 348, 2672]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/45_text_wove.png
try:
    _c45 = get_crop(45, 76, 18)
    canvas.paste(_c45, (61, 2679), _c45)
except Exception:
    pass
layout["wove"] = [61, 2679, 137, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/46_text_Passio.png
try:
    _c46 = get_crop(46, 96, 29)
    canvas.paste(_c46, (245, 2694), _c46)
except Exception:
    pass
layout["Passio"] = [245, 2694, 341, 2723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/47_text_Lmy_Danse_Gala_wl_di_pel.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (0, 2804), _c47)
except Exception:
    pass
layout["Lmy_Danse_Gala_wl_di_pel"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/48_text_9_666_creator_followers.png
try:
    _c48 = get_crop(48, 1344, 346)
    canvas.paste(_c48, (48, 2470), _c48)
except Exception:
    pass
layout["9_666_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/49_clickable_Favorites.png
try:
    _c49 = get_crop(49, 288, 156)
    canvas.paste(_c49, (576, 2804), _c49)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/50_clickable_Tickets.png
try:
    _c50 = get_crop(50, 288, 156)
    canvas.paste(_c50, (864, 2804), _c50)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_01_2024_4_24_16_59_92c22920a83749c994864397a370a984-3/51_clickable_More.png
try:
    _c51 = get_crop(51, 288, 156)
    canvas.paste(_c51, (1152, 2804), _c51)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
