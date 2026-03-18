# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_05
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7.png
# step_index: 5/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure only (no icons/text)
# Assumes: canvas (1440x2960 PIL Image) and draw (ImageDraw) are provided.

# Overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")  # very light off-white canvas

# Status bar (top)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#8F9498")  # muted dark status bar
# subtle bottom line under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#7E8387", width=1)

# Header / toolbar area (under status bar)
header_top = status_h
header_bottom = 320
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")  # white toolbar bg
# soft divider below header
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#E1E2E6", width=2)

# Big section divider under filters/search area
filters_div_y = 620
draw.line([(36, filters_div_y), (1404, filters_div_y)], fill="#E6E7EA", width=2)

# Event image card 1 background (behind pasted artwork at pos (48,676) size 1344x1096)
card1_box = (36, 660, 1404, 1788)  # slight padding around detected image rect
draw.rounded_rectangle(card1_box, radius=18, fill="#FFFFFF", outline="#E7E9EC", width=1)
# subtle drop shadow (bottom)
shadow_box1 = (36, 1786, 1404, 1800)
draw.rectangle(shadow_box1, fill="#F3F4F6")

# Thin separator under first card (mimics subtle divider between image and details)
sep1_y = 1788
draw.line([(48, sep1_y), (1392, sep1_y)], fill="#E6E7EA", width=2)

# Event details card area (the area that holds title/date/venue - keep as background block)
details1_box = (36, 1800, 1404, 1988)
draw.rounded_rectangle(details1_box, radius=12, fill="#FFFFFF", outline="#F0F1F3", width=1)
# faint inner divider to separate meta from promoted label area
draw.line([(48, 1924), (1392, 1924)], fill="#F0F1F3", width=1)

# Second event image card background (behind pasted artwork at pos (48,1820) size 1344x996)
card2_box = (36, 1808, 1404, 2828)  # padded area for second large poster
draw.rounded_rectangle(card2_box, radius=18, fill="#FFFFFF", outline="#E7E9EC", width=1)
# drop shadow under second card
shadow_box2 = (36, 2826, 1404, 2840)
draw.rectangle(shadow_box2, fill="#F3F4F6")

# Separator between event listings (between second big card and subsequent content)
sep2_y = 2828
draw.line([(48, sep2_y), (1392, sep2_y)], fill="#E2E3E6", width=2)

# Bottom navigation bar background
nav_h = 144
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
# top border for nav
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E7EA", width=2)

# Small horizontal separators for list sections (additional subtle guides)
for y in (520, 880, 1360, 1660, 2200):
    draw.line([(48, y), (1392, y)], fill="#F1F2F4", width=1)

# Accent left margin column (subtle vertical guideline matching screenshot layout)
draw.rectangle([(0, 260), (48, 2620)], fill="#FBFBFD")  # keep margin area consistent (no content)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/04_icon_Fo.png
try:
    _c4 = get_crop(4, 142, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1437, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2336), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2336), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/07_icon_9.12.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.12"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 62, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/10_icon_9.12.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.12"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 92, 60)
    canvas.paste(_c11, (1207, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1207, 0, 1299, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 65, 59)
    canvas.paste(_c12, (1315, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1315, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/13_icon_9.12.png
try:
    _c13 = get_crop(13, 57, 64)
    canvas.paste(_c13, (115, 0), _c13)
except Exception:
    pass
layout["9.12"] = [115, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1236, 1192), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/16_icon_Fri_Mar_22_._6_00_PM_EDT.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["Fri,_Mar_22_._6:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/17_icon_New_York.png
try:
    _c17 = get_crop(17, 434, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 51, 61)
    canvas.paste(_c18, (383, 2), _c18)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/19_icon_Creative_Chai_Chronicles_Belonging_as_a.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Creative_Chai_Chronicles:"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/20_icon_22_Haach_80z4_I_PIcA_07_Chclsca.png
try:
    _c20 = get_crop(20, 1344, 996)
    canvas.paste(_c20, (48, 1820), _c20)
except Exception:
    pass
layout["22_Haach_80z4_I_PIcA_07,_"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/21_icon_Tequila_Artistic_Transformation.png
try:
    _c21 = get_crop(21, 1344, 1096)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/23_icon_Creative_Chai_Chronicles_Belonging_as_a.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Creative_Chai_Chronicles:"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/24_icon_Fri_Mar_22_._6_00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Fri,_Mar_22_._6:00_PM_EDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/25_icon_10_000_events.png
try:
    _c25 = get_crop(25, 215, 290)
    canvas.paste(_c25, (215, 669), _c25)
except Exception:
    pass
layout["10,000_events"] = [215, 669, 430, 959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 247, 67)
    canvas.paste(_c26, (84, 1664), _c26)
except Exception:
    pass
layout["Promoted"] = [84, 1664, 331, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/27_icon_Favorite_button.png
try:
    _c27 = get_crop(27, 144, 144)
    canvas.paste(_c27, (1092, 1192), _c27)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 40, 61)
    canvas.paste(_c28, (1274, 0), _c28)
except Exception:
    pass
layout["icon_28"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/29_text_9.12.png
try:
    _c29 = get_crop(29, 91, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_05_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-7/31_text_B.GLALl.png
try:
    _c31 = get_crop(31, 241, 126)
    canvas.paste(_c31, (532, 1853), _c31)
except Exception:
    pass
layout["B.GLALl"] = [532, 1853, 773, 1979]
