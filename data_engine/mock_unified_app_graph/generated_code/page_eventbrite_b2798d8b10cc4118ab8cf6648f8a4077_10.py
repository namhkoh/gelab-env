# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_10
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12.png
# step_index: 10/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 56)], fill=(154, 154, 154))

# Header area (white) with bottom divider
draw.rectangle([(0, 56), (1440, 144)], fill=(255, 255, 255))
draw.line([(48, 144), (1392, 144)], fill=(230, 230, 230), width=2)

# Search field background (rounded) - leave content area empty for paste-on elements
search_box = (48, 72, 1392, 263)  # matches detected search area dimensions
draw.rounded_rectangle(search_box, radius=26, fill=(249, 249, 249), outline=(224, 224, 224), width=1)

# Subtle divider under filters area
draw.line([(48, 360), (1392, 360)], fill=(240, 240, 240), width=2)

# Section heading divider (above first event)
draw.line([(48, 520), (1392, 520)], fill=(245, 245, 245), width=2)

# First event card background (rounded rectangle + subtle border)
card1_outer = (40, 640, 1400, 1768)
# light shadow/backdrop
draw.rounded_rectangle((card1_outer[0]+2, card1_outer[1]+6, card1_outer[2]+2, card1_outer[3]+6),
                       radius=22, fill=(250, 250, 250), outline=None)
# card itself
draw.rounded_rectangle(card1_outer, radius=20, fill=(255, 255, 255), outline=(235, 235, 235), width=1)

# Divider between image area and text area within first card (subtle)
# Place near bottom part of the image region (roughly)
draw.line([(card1_outer[0]+24, card1_outer[1]+520), (card1_outer[2]-24, card1_outer[1]+520)],
          fill=(245, 245, 245), width=1)

# Second event card background (rounded rectangle + subtle border)
card2_outer = (40, 1840, 1400, 2628)
# shadow/backdrop
draw.rounded_rectangle((card2_outer[0]+2, card2_outer[1]+6, card2_outer[2]+2, card2_outer[3]+6),
                       radius=22, fill=(250, 250, 250), outline=None)
# card itself
draw.rounded_rectangle(card2_outer, radius=20, fill=(255, 255, 255), outline=(235, 235, 235), width=1)

# Divider lines to separate event items/metadata areas
draw.line([(48, 1760), (1392, 1760)], fill=(245, 245, 245), width=2)
draw.line([(48, 2568), (1392, 2568)], fill=(245, 245, 245), width=2)

# Large content background band behind event listings (very subtle off-white)
draw.rectangle([(0, 520), (1440, 2800)], fill=(255, 255, 255))

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(24, nav_top), (1416, nav_top)], fill=(230, 230, 230), width=2)

# Small horizontal separators for spacing near top and mid content
draw.line([(48, 320), (1392, 320)], fill=(238, 238, 238), width=1)
draw.line([(48, 440), (1392, 440)], fill=(245, 245, 245), width=1)

# Rounded corner highlight on page edges to mimic subtle card spacing (very light)
draw.rounded_rectangle((28, 628, 1412, 2640), radius=24, outline=(248, 248, 248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (850, 390), _c1)
except Exception:
    pass
layout["Music"] = [850, 390, 1037, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/03_icon_6_J01.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout[";6_J01"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/04_icon_6_J01.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout[";6_J01"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/05_icon_SABOR.png
try:
    _c5 = get_crop(5, 1344, 1175)
    canvas.paste(_c5, (48, 676), _c5)
except Exception:
    pass
layout["SABOR"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/06_icon_9.20.png
try:
    _c6 = get_crop(6, 122, 113)
    canvas.paste(_c6, (56, 114), _c6)
except Exception:
    pass
layout["9.20"] = [56, 114, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/07_icon_RREE.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["RREE"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 65)
    canvas.paste(_c8, (1152, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1152, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 55, 61)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/10_icon_RREE.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1092, 2415), _c10)
except Exception:
    pass
layout["RREE"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/11_icon_9.20.png
try:
    _c11 = get_crop(11, 55, 62)
    canvas.paste(_c11, (182, 0), _c11)
except Exception:
    pass
layout["9.20"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 69, 62)
    canvas.paste(_c12, (1212, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1212, 0, 1281, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 59, 62)
    canvas.paste(_c13, (312, 1), _c13)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 58, 59)
    canvas.paste(_c14, (1318, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/16_icon_9.20.png
try:
    _c16 = get_crop(16, 57, 64)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["9.20"] = [114, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 50, 62)
    canvas.paste(_c17, (383, 1), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/18_icon_Promoted.png
try:
    _c18 = get_crop(18, 255, 66)
    canvas.paste(_c18, (73, 1743), _c18)
except Exception:
    pass
layout["Promoted"] = [73, 1743, 328, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/19_icon_DiA_rnt.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["DiA_rnt"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/20_icon_Zona_Rosa_Saturdays_at_Amadeus_New_York.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["Zona_Rosa_Saturdays_at_Am"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/21_icon_Ticket_sales_end_soon.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/22_icon_Zona_Rosa_Saturdays_at_Amadeus_New_York.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Zona_Rosa_Saturdays_at_Am"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/23_icon_New_York.png
try:
    _c23 = get_crop(23, 434, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 42, 61)
    canvas.paste(_c24, (1273, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1273, 0, 1315, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/26_icon_R_A.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["R_A"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/27_icon_Reggaeton.png
try:
    _c27 = get_crop(27, 1344, 1175)
    canvas.paste(_c27, (48, 676), _c27)
except Exception:
    pass
layout["Reggaeton"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/28_text_9.20.png
try:
    _c28 = get_crop(28, 91, 43)
    canvas.paste(_c28, (20, 17), _c28)
except Exception:
    pass
layout["9.20"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/29_text_4_625_events.png
try:
    _c29 = get_crop(29, 372, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["4,625_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/30_text_Free.png
try:
    _c30 = get_crop(30, 80, 38)
    canvas.paste(_c30, (117, 1391), _c30)
except Exception:
    pass
layout["Free"] = [117, 1391, 197, 1429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/31_text_0_NA.png
try:
    _c31 = get_crop(31, 186, 61)
    canvas.paste(_c31, (214, 1928), _c31)
except Exception:
    pass
layout["0_NA"] = [214, 1928, 400, 1989]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/32_text_SATURDAY.png
try:
    _c32 = get_crop(32, 161, 41)
    canvas.paste(_c32, (1058, 1932), _c32)
except Exception:
    pass
layout["SATURDAY"] = [1058, 1932, 1219, 1973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/33_text_AMAPEUS.png
try:
    _c33 = get_crop(33, 222, 88)
    canvas.paste(_c33, (171, 2379), _c33)
except Exception:
    pass
layout["AMAPEUS"] = [171, 2379, 393, 2467]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/34_text_NEW_ERA.png
try:
    _c34 = get_crop(34, 1344, 917)
    canvas.paste(_c34, (48, 1899), _c34)
except Exception:
    pass
layout["NEW_ERA"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/35_text_7i0_u7o.png
try:
    _c35 = get_crop(35, 136, 25)
    canvas.paste(_c35, (99, 2488), _c35)
except Exception:
    pass
layout["7i0__,u7o"] = [99, 2488, 235, 2513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/36_text_CLUDA_ADLUS_Cov.png
try:
    _c36 = get_crop(36, 208, 25)
    canvas.paste(_c36, (254, 2488), _c36)
except Exception:
    pass
layout["CLUDA__ADLUS_Cov"] = [254, 2488, 462, 2513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/37_text_Jgmnonaunul_AnUaI.png
try:
    _c37 = get_crop(37, 266, 18)
    canvas.paste(_c37, (135, 2519), _c37)
except Exception:
    pass
layout["Jgmnonaunul_(AnUaI"] = [135, 2519, 401, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/38_text_TOUIE_MINAYA.png
try:
    _c38 = get_crop(38, 1344, 917)
    canvas.paste(_c38, (48, 1899), _c38)
except Exception:
    pass
layout["TOUIE_MINAYA"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/39_text_Zona_Rosa_Saturdays_at_Amadeus_New_York.png
try:
    _c39 = get_crop(39, 1344, 917)
    canvas.paste(_c39, (48, 1899), _c39)
except Exception:
    pass
layout["Zona_Rosa_Saturdays_at_Am"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/40_text_R_A.png
try:
    _c40 = get_crop(40, 39, 18)
    canvas.paste(_c40, (181, 2790), _c40)
except Exception:
    pass
layout["R_A"] = [181, 2790, 220, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/41_text_00.png
try:
    _c41 = get_crop(41, 44, 18)
    canvas.paste(_c41, (269, 2790), _c41)
except Exception:
    pass
layout["00"] = [269, 2790, 313, 2808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_10_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-12/42_text_DiA_rnt.png
try:
    _c42 = get_crop(42, 145, 25)
    canvas.paste(_c42, (467, 2784), _c42)
except Exception:
    pass
layout["DiA_rnt"] = [467, 2784, 612, 2809]
