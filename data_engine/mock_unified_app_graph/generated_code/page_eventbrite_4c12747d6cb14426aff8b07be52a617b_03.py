# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_03
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5.png
# step_index: 3/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(225, 225, 225))

# Header / toolbar area below status bar
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Blue accent underline for the search/header (thin)
underline_left = 48
underline_right = 1392
underline_y = 148
draw.rectangle((underline_left, underline_y, underline_right, underline_y + 4), fill=(30, 90, 255))

# Light divider under header
draw.line((0, header_bottom, 1440, header_bottom), fill=(235, 235, 235), width=1)

# Section separators (light horizontal rules)
sep_positions = [440, 1020, 1400]
for y in sep_positions:
    draw.line((48, y, 1392, y), fill=(240, 240, 240), width=2)

# Event list: draw subtle card backgrounds (rounded rects) behind event groups
card_x0 = 48
card_x1 = 1392
card_top_start = 1480
card_height = 360
card_gap = 20
card_fill_light = (255, 255, 255)
card_fill_alt = (250, 250, 252)
card_outline = (235, 235, 236)

# Draw 4 cards down the list area
for i in range(4):
    top = card_top_start + i * (card_height + card_gap)
    bottom = top + card_height
    fill = card_fill_alt if (i % 2) else card_fill_light
    # rounded_rectangle is available on ImageDraw
    try:
        draw.rounded_rectangle((card_x0, top, card_x1, bottom), radius=10, fill=fill, outline=card_outline, width=1)
    except Exception:
        # fallback for older PIL: draw rectangle and small corner circles
        draw.rectangle((card_x0, top, card_x1, bottom), fill=fill, outline=card_outline, width=1)

    # subtle inner divider separating thumbnail area from text area
    thumb_w = 200
    thumb_margin = 12
    thumb_x0 = card_x0 + thumb_margin
    thumb_x1 = thumb_x0 + thumb_w
    # draw a pale image placeholder background (will be covered by actual thumbnails)
    draw.rectangle((thumb_x0, top + 12, thumb_x1, bottom - 12), fill=(245, 245, 245))

    # vertical separation line between thumbnail and content (very subtle)
    draw.line((thumb_x1 + 8, top + 16, thumb_x1 + 8, bottom - 16), fill=(245, 245, 245), width=1)

    # bottom separator for the card
    draw.line((card_x0 + 8, bottom + 6, card_x1 - 8, bottom + 6), fill=(245, 245, 245), width=1)

# Large banner / content background near top (e.g., 'Popular' area accent)
banner_top = 300
banner_bottom = 360
draw.rectangle((48, banner_top, 1392, banner_bottom), fill=(255, 255, 255))
draw.line((48, banner_bottom, 1392, banner_bottom), fill=(240, 240, 240), width=1)

# Bottom navigation bar background and divider
bottom_nav_top = 2804
draw.rectangle((0, bottom_nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill=(230, 230, 230), width=2)
draw.rectangle((0, bottom_nav_top - 6, 1440, bottom_nav_top), fill=(248, 248, 248))

# Subtle left margin vertical guide (not content)
draw.line((48, header_bottom + 8, 48, bottom_nav_top - 8), fill=(250, 250, 250), width=1)

# Final subtle overall shading lines to match screenshot feel
draw.line((0, status_h, 1440, status_h), fill=(220, 220, 220), width=1)
draw.line((0, 2960 - 1, 1440, 2960 - 1), fill=(230, 230, 230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/00_icon_7.52.png
try:
    _c0 = get_crop(0, 55, 62)
    canvas.paste(_c0, (116, 2), _c0)
except Exception:
    pass
layout["7.52"] = [116, 2, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 54, 59)
    canvas.paste(_c1, (314, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [314, 3, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/02_icon_7.52.png
try:
    _c2 = get_crop(2, 53, 60)
    canvas.paste(_c2, (183, 2), _c2)
except Exception:
    pass
layout["7.52"] = [183, 2, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/03_icon_Swchez_ML_CesMfbetd_CebkUx_Z02Aldl_Cart_.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1909), _c3)
except Exception:
    pass
layout["Swchez_ML_CesMfbetd_CebkU"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 41, 54)
    canvas.paste(_c4, (254, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [254, 6, 295, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/05_icon_Fri_May_3_6.30_PM_PDT.png
try:
    _c5 = get_crop(5, 288, 156)
    canvas.paste(_c5, (288, 2804), _c5)
except Exception:
    pass
layout["Fri,_May_3_*_6.30_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/06_icon_art_fairs.png
try:
    _c6 = get_crop(6, 1344, 120)
    canvas.paste(_c6, (48, 738), _c6)
except Exception:
    pass
layout["art_fairs"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (864, 2804), _c7)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/08_icon_Sanchez_Art_Center_programs.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1909), _c8)
except Exception:
    pass
layout["Sanchez_Art_Center_progra"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/09_icon_7.52.png
try:
    _c9 = get_crop(9, 118, 104)
    canvas.paste(_c9, (58, 119), _c9)
except Exception:
    pass
layout["7.52"] = [58, 119, 176, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/10_icon_Events.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1117), _c10)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/11_icon_Fri_May_3_6.30_PM_PDT.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["Fri,_May_3_*_6.30_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/12_icon_8_21268_creator_followers.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1117), _c12)
except Exception:
    pass
layout["8_21268_creator_followers"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/13_icon_Sat.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 2305), _c13)
except Exception:
    pass
layout["Sat,"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 47, 60)
    canvas.paste(_c14, (1322, 2), _c14)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/15_icon_More.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (1152, 2804), _c15)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 74, 63)
    canvas.paste(_c16, (1216, 0), _c16)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1290, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/17_icon_The_Art_of_Kokedama.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1513), _c17)
except Exception:
    pass
layout["The_Art_of_Kokedama"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 41, 61)
    canvas.paste(_c18, (1272, 2), _c18)
except Exception:
    pass
layout["Cancel"] = [1272, 2, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1099, 96), _c19)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/20_icon_7.52.png
try:
    _c20 = get_crop(20, 92, 60)
    canvas.paste(_c20, (16, 3), _c20)
except Exception:
    pass
layout["7.52"] = [16, 3, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/21_icon_Classpoplm.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1513), _c21)
except Exception:
    pass
layout["Classpoplm"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 149, 144)
    canvas.paste(_c22, (1243, 97), _c22)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/23_icon_21268_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 2305), _c23)
except Exception:
    pass
layout["21268_creator_followers"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/24_icon_Sanchez_Art_Center.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1909), _c24)
except Exception:
    pass
layout["Sanchez_Art_Center"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/25_icon_Home.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/26_icon_Art.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1909), _c26)
except Exception:
    pass
layout["Art,"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/27_icon_Luludi_Living_Art.png
try:
    _c27 = get_crop(27, 262, 55)
    canvas.paste(_c27, (390, 2539), _c27)
except Exception:
    pass
layout["Luludi_Living_Art"] = [390, 2539, 652, 2594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/28_text_Art.png
try:
    _c28 = get_crop(28, 123, 73)
    canvas.paste(_c28, (203, 135), _c28)
except Exception:
    pass
layout["Art"] = [203, 135, 326, 208]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/29_text_Popular.png
try:
    _c29 = get_crop(29, 221, 78)
    canvas.paste(_c29, (44, 298), _c29)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/30_text_artificial_intelligence.png
try:
    _c30 = get_crop(30, 1344, 120)
    canvas.paste(_c30, (48, 378), _c30)
except Exception:
    pass
layout["artificial_intelligence"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/31_text_art.png
try:
    _c31 = get_crop(31, 64, 39)
    canvas.paste(_c31, (163, 556), _c31)
except Exception:
    pass
layout["art"] = [163, 556, 227, 595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/32_text_art_openings.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 618), _c32)
except Exception:
    pass
layout["art_openings"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/33_text_martial_arts.png
try:
    _c33 = get_crop(33, 218, 43)
    canvas.paste(_c33, (168, 912), _c33)
except Exception:
    pass
layout["martial_arts"] = [168, 912, 386, 955]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/34_text_Events.png
try:
    _c34 = get_crop(34, 188, 61)
    canvas.paste(_c34, (45, 1026), _c34)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/35_text_Fri_May_3_6.30_PM_PDT.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (576, 2804), _c35)
except Exception:
    pass
layout["Fri,_May_3_*_6.30_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/36_clickable_Art.png
try:
    _c36 = get_crop(36, 1344, 191)
    canvas.paste(_c36, (48, 72), _c36)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/37_clickable_art_workshops.png
try:
    _c37 = get_crop(37, 1344, 120)
    canvas.paste(_c37, (48, 498), _c37)
except Exception:
    pass
layout["art_workshops"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_03_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-5/38_clickable_martial_arts.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 858), _c38)
except Exception:
    pass
layout["martial_arts"] = [48, 858, 1392, 1002]
