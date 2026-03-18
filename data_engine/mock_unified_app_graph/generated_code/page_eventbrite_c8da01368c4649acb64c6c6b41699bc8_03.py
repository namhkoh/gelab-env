# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_03
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5.png
# step_index: 3/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_height = 56
status_color = (191, 191, 191)  # light gray for status bar
draw.rectangle([(0, 0), (1440, status_height)], fill=status_color)

# Header area (keeps canvas white, but add subtle background band and underline accent)
header_top = status_height
header_bottom = 152
header_bg_color = (255, 255, 255)
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_bg_color)

# Blue underline below search/header (accent)
underline_y = header_bottom - 2
underline_color = (43, 107, 230)  # vivid blue
draw.rectangle([(48, underline_y), (1392, underline_y + 6)], fill=underline_color)

# subtle divider line just above the blue underline
draw.line([(48, underline_y - 6), (1392, underline_y - 6)], fill=(230, 230, 230), width=1)

# "Popular" list separators (rows for suggestions)
popular_rows = [
    (48, 378, 1344, 120),
    (48, 498, 1344, 120),
    (48, 618, 1344, 120),
    (48, 738, 1344, 120),
    (48, 858, 1344, 144),
]
separator_color = (230, 230, 230)
for x, y, w, h in popular_rows:
    # draw a faint bottom separator for each row
    sep_y = y + h - 1
    draw.line([(x + 12, sep_y), (x + w - 12, sep_y)], fill=separator_color, width=1)

# Light section separator above events list
draw.line([(48, 1008), (1392, 1008)], fill=(240, 240, 240), width=1)

# Event cards background (rounded rectangles behind each event row)
event_rows_y = [1117, 1513, 1909, 2305]
card_fill = (251, 251, 251)  # very light off-white for event row background
card_outline = (237, 237, 240)
card_radius = 14
card_x1 = 48
card_w = 1344
card_h = 396
for y in event_rows_y:
    rect = [ (card_x1, y), (card_x1 + card_w, y + card_h) ]
    # subtle top shadow (thin darker line)
    shadow_y = y
    draw.line([(card_x1 + 6, shadow_y), (card_x1 + card_w - 6, shadow_y)], fill=(245,245,246), width=1)
    # rounded card
    draw.rounded_rectangle(rect, radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    # separator under each card
    draw.line([(card_x1 + 12, y + card_h), (card_x1 + card_w - 12, y + card_h)], fill=(236,236,236), width=1)

# Thin separators between major sections (after popular block and events header)
draw.line([(48, 1040), (1392, 1040)], fill=(242, 242, 242), width=1)

# Bottom navigation background and top divider
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill=(230, 230, 230), width=2)

# Final subtle overall left/right margins guides (very faint)
draw.line([(48, 0), (48, 2960)], fill=(250,250,250), width=1)
draw.line([(1392, 0), (1392, 2960)], fill=(250,250,250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/00_icon_top-Motion_Animation.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1909), _c0)
except Exception:
    pass
layout["top-Motion_Animation"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/01_icon_Oam-4pm.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2305), _c1)
except Exception:
    pass
layout["Oam-4pm"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/02_icon_Rush_Tribute_wl_Animation_at.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1513), _c2)
except Exception:
    pass
layout["Rush_Tribute_wl_Animation"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/03_icon_06.02.2024_2-5_PM.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1117), _c3)
except Exception:
    pass
layout["06.02.2024|2-5_PM_"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/04_icon_Aninzon.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1513), _c4)
except Exception:
    pass
layout["Aninzon"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/05_icon_One_Frame_at_a_Time_An_Intro_to.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1909), _c5)
except Exception:
    pass
layout["One_Frame_at_a_Time:_An_I"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/06_icon_Animal.png
try:
    _c6 = get_crop(6, 1344, 191)
    canvas.paste(_c6, (48, 72), _c6)
except Exception:
    pass
layout["Animal"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/07_icon_Animal.png
try:
    _c7 = get_crop(7, 56, 60)
    canvas.paste(_c7, (313, 3), _c7)
except Exception:
    pass
layout["Animal"] = [313, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/08_icon_5.15.png
try:
    _c8 = get_crop(8, 53, 61)
    canvas.paste(_c8, (183, 2), _c8)
except Exception:
    pass
layout["5.15"] = [183, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/09_icon_An7.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["An7"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 43, 56)
    canvas.paste(_c10, (253, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [253, 5, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/11_icon_5.15.png
try:
    _c11 = get_crop(11, 56, 62)
    canvas.paste(_c11, (116, 2), _c11)
except Exception:
    pass
layout["5.15"] = [116, 2, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/12_icon_2_00_PM_CDT.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1117), _c12)
except Exception:
    pass
layout["2:00_PM_CDT"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/13_icon_5.15.png
try:
    _c13 = get_crop(13, 128, 110)
    canvas.paste(_c13, (51, 116), _c13)
except Exception:
    pass
layout["5.15"] = [51, 116, 179, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/14_icon_Stop-Motion_Animation.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1909), _c14)
except Exception:
    pass
layout["Stop-Motion_Animation"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/15_icon_Pierce_County_Fairgrounds.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1117), _c15)
except Exception:
    pass
layout["Pierce_County_Fairgrounds"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/16_icon_animal_yoga.png
try:
    _c16 = get_crop(16, 1344, 120)
    canvas.paste(_c16, (48, 738), _c16)
except Exception:
    pass
layout["animal_yoga"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 95, 63)
    canvas.paste(_c17, (1214, 0), _c17)
except Exception:
    pass
layout["Cancel"] = [1214, 0, 1309, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 50, 62)
    canvas.paste(_c18, (1321, 1), _c18)
except Exception:
    pass
layout["Cancel"] = [1321, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/19_icon_animal_rights.png
try:
    _c19 = get_crop(19, 1344, 120)
    canvas.paste(_c19, (48, 498), _c19)
except Exception:
    pass
layout["animal_rights"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/20_icon_IO_O0AM_CDT.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 2305), _c20)
except Exception:
    pass
layout["IO:O0AM_CDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/21_icon_2_00_PM_CDT.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 858), _c21)
except Exception:
    pass
layout["2:00_PM_CDT"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/23_icon_83468_creator_followers.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["83468_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/24_icon_animals.png
try:
    _c24 = get_crop(24, 1344, 120)
    canvas.paste(_c24, (48, 618), _c24)
except Exception:
    pass
layout["animals"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/25_icon_animal_events.png
try:
    _c25 = get_crop(25, 1344, 120)
    canvas.paste(_c25, (48, 378), _c25)
except Exception:
    pass
layout["animal_events"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/26_icon_Cancel.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1099, 96), _c26)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/27_icon_Favorites.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/28_icon_Cancel.png
try:
    _c28 = get_crop(28, 149, 144)
    canvas.paste(_c28, (1243, 97), _c28)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/29_icon_IO_O0AM_CDT.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2305), _c29)
except Exception:
    pass
layout["IO:O0AM_CDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/30_icon_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/31_icon_Garfield_Park_Conservatory.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2305), _c31)
except Exception:
    pass
layout["Garfield_Park_Conservator"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/32_icon_animal_tales.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 858), _c32)
except Exception:
    pass
layout["animal_tales"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/33_text_5.15.png
try:
    _c33 = get_crop(33, 92, 43)
    canvas.paste(_c33, (22, 17), _c33)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/34_text_Popular.png
try:
    _c34 = get_crop(34, 221, 78)
    canvas.paste(_c34, (44, 298), _c34)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_03_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-5/35_text_An7.png
try:
    _c35 = get_crop(35, 62, 14)
    canvas.paste(_c35, (981, 2794), _c35)
except Exception:
    pass
layout["An7"] = [981, 2794, 1043, 2808]
