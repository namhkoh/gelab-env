# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_01
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3.png
# step_index: 1/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for a 1440x2960 canvas.
# Assumes: canvas (PIL.Image 1440x2960 RGB), draw (PIL.ImageDraw.Draw), font_* variables available.

# Overall page background (very light warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 249, 246))

# Status bar area at top (~80px) - muted grey
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=(205, 205, 205))
# subtle bottom edge for status bar
draw.line([(0, status_h), (1440, status_h)], fill=(190, 190, 190), width=1)

# Header / toolbar area beneath status bar (space for app bar and search background)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# toolbar bottom divider (soft)
draw.line([(0, header_bottom), (1440, header_bottom)], fill=(235, 235, 235), width=2)

# Content area background (slightly different tone behind the list to separate from header)
content_top = header_bottom
content_bottom = 2800  # leave room for bottom nav bar
draw.rectangle([(0, content_top), (1440, content_bottom)], fill=(250, 249, 246))

# Card/list item container positions (left padding 48, width 1344)
card_left = 48
card_right = card_left + 1344
card_positions = [490, 886, 1282, 1678, 2074, 2470]
card_heights = [396, 396, 396, 396, 396, 346]
card_radius = 14

# Draw each card background and subtle separators/shadows
for top_y, h in zip(card_positions, card_heights):
    bottom_y = top_y + h
    # subtle drop shadow (very faint)
    shadow_rect = (card_left + 4, bottom_y - 2, card_right + 4, bottom_y + 6)
    draw.rectangle(shadow_rect, fill=(245, 244, 243))
    # rounded card background (keeps cards visually separate from page)
    draw.rounded_rectangle([(card_left, top_y), (card_right, bottom_y)],
                           radius=card_radius, fill=(255, 255, 255), outline=None)
    # thin divider line at bottom of card
    draw.line([(card_left + 12, bottom_y), (card_right - 12, bottom_y)],
              fill=(240, 240, 240), width=1)

# Section separators between groups (extra subtle lines)
sep_x1 = card_left
sep_x2 = card_right
for y in [card_positions[0] - 34, card_positions[2] - 34, card_positions[4] - 34]:
    # only draw within content area
    if content_top < y < content_bottom:
        draw.line([(sep_x1, y), (sep_x2, y)], fill=(245, 245, 245), width=1)

# Bottom navigation bar background and top divider (approx 156px high)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 230), width=2)

# Subtle safe-area padding shadow above nav to separate from content
draw.rectangle([(0, nav_top - 6), (1440, nav_top)], fill=(248, 248, 248))

# Small left edge guideline for content column (visual structural accent)
accent_x = card_left - 8
draw.line([(accent_x, header_bottom + 12), (accent_x, nav_top - 12)], fill=(247, 246, 245), width=2)

# Finished structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/00_icon_FRIDAY.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/01_icon_NDIE_DANCEPA.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/03_icon_NDIE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["NDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/04_icon_REoPUNKSFRE.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 886), _c4)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/05_icon_Los_Angeles.png
try:
    _c5 = get_crop(5, 456, 117)
    canvas.paste(_c5, (492, 2651), _c5)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1539), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 763), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/09_icon_Afliccion_Perdida_y.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 2347), _c9)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/11_icon_Public_House_Los_Angeles_CA.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 490), _c11)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1159), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/14_icon_8_4717_creator_followers.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1282), _c14)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/15_icon_Sylmai.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 763), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1284, 1539), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1159), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/19_icon_8_21119_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/20_icon_Home.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 59)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/22_icon_7.46.png
try:
    _c22 = get_crop(22, 57, 61)
    canvas.paste(_c22, (182, 2), _c22)
except Exception:
    pass
layout["7.46"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/23_icon_7.46.png
try:
    _c23 = get_crop(23, 102, 98)
    canvas.paste(_c23, (41, 122), _c23)
except Exception:
    pass
layout["7.46"] = [41, 122, 143, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 52, 60)
    canvas.paste(_c24, (247, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/25_icon_59_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1678), _c25)
except Exception:
    pass
layout["59_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 52)
    canvas.paste(_c26, (1321, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 83, 57)
    canvas.paste(_c27, (1212, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [1212, 5, 1295, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/28_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 490), _c28)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/29_icon_7.46.png
try:
    _c29 = get_crop(29, 59, 62)
    canvas.paste(_c29, (115, 1), _c29)
except Exception:
    pass
layout["7.46"] = [115, 1, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/32_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1282), _c32)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/33_icon_Free.png
try:
    _c33 = get_crop(33, 125, 74)
    canvas.paste(_c33, (248, 1749), _c33)
except Exception:
    pass
layout["Free"] = [248, 1749, 373, 1823]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 41, 55)
    canvas.paste(_c34, (1272, 6), _c34)
except Exception:
    pass
layout["icon_34"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/36_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/37_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/38_icon_5.30_PM_PDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/39_icon_31_creator_followers.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (576, 2804), _c39)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/40_text_7.46.png
try:
    _c40 = get_crop(40, 89, 43)
    canvas.paste(_c40, (22, 15), _c40)
except Exception:
    pass
layout["7.46"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/42_text_Mon_May_13.png
try:
    _c42 = get_crop(42, 222, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/43_text_5.30_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_01_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
