# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_05
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7.png
# step_index: 5/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base fill
bg_color = (250, 250, 252)  # subtle off-white dominant background
draw.rectangle((0, 0, 1440, 2960), fill=bg_color)

# Status bar area (top ~56px) - dark strip
status_h = 56
status_color = (115, 115, 115)
draw.rectangle((0, 0, 1440, status_h), fill=status_color)

# Header / search area background (under status bar)
header_top = status_h
header_bottom = 160
header_bg = (250, 250, 250)
draw.rectangle((0, header_top, 1440, header_bottom), fill=header_bg)

# Subtle divider under the search/header area
divider_y = 140
draw.line((48, divider_y, 1392, divider_y), fill=(210, 209, 214), width=2)

# Rounded search field background (behind detected search icons/text)
search_left = 48
search_right = 1392
search_top = 72
search_bottom = 132
search_bg = (244, 247, 250)  # very light bluish-gray
draw.rounded_rectangle((search_left, search_top, search_right, search_bottom), radius=28, fill=search_bg, outline=None)

# Light location row background (under search, behind filter pills)
loc_row_top = 152
loc_row_bottom = 230
draw.rectangle((48, loc_row_top, 1392, loc_row_bottom), fill=header_bg)

# First event card shadow + card background
card1_x0 = 44
card1_y0 = 670
card1_x1 = 1396
card1_y1 = 1856
shadow_color = (236, 237, 240)
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1), radius=28, fill=shadow_color)

card1_inset = 6
card1_fill = (255, 255, 255)
draw.rounded_rectangle((card1_x0 + card1_inset, card1_y0 + card1_inset, card1_x1 - card1_inset, card1_y1 - card1_inset), radius=24, fill=card1_fill, outline=(230,230,235))

# Small thin separator under first card to visually separate from following content
sep_y = card1_y1 + 12
draw.line((48, sep_y, 1392, sep_y), fill=(240,240,243), width=1)

# Second event card shadow + card background
card2_x0 = 44
card2_y0 = 1888
card2_x1 = 1396
card2_y1 = 2818
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=28, fill=shadow_color)

card2_inset = 6
draw.rounded_rectangle((card2_x0 + card2_inset, card2_y0 + card2_inset, card2_x1 - card2_inset, card2_y1 - card2_inset), radius=24, fill=card1_fill, outline=(230,230,235))

# Divider lines between major sections (light)
draw.line((48, 620, 1392, 620), fill=(242,242,245), width=1)  # above first card area
draw.line((48, 1860, 1392, 1860), fill=(242,242,245), width=1)  # between first card content and meta

# Bottom navigation bar background and top divider
nav_top = 2840
draw.line((0, nav_top, 1440, nav_top), fill=(220,220,224), width=2)
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))

# Slight elevation highlight for nav area
draw.line((0, nav_top + 2, 1440, nav_top + 2), fill=(250,250,251), width=1)

# Optional subtle left/right page padding guides (very light, non-intrusive)
pad_color = (255, 255, 255, )
draw.rectangle((0, 0, 48, 2960), fill=bg_color)
draw.rectangle((1392, 0, 1440, 2960), fill=bg_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 135)
    canvas.paste(_c1, (850, 390), _c1)
except Exception:
    pass
layout["Music"] = [850, 390, 1037, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/03_icon_Jpcio.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Jpcio"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/04_icon_Jpcio.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Jpcio"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 53, 59)
    canvas.paste(_c5, (248, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 2, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/06_icon_EcOMMER.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["EcOMMER"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/07_icon_9.19.png
try:
    _c7 = get_crop(7, 123, 115)
    canvas.paste(_c7, (55, 113), _c7)
except Exception:
    pass
layout["9.19"] = [55, 113, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/08_icon_EcOMMER.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["EcOMMER"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 57, 62)
    canvas.paste(_c9, (313, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [313, 1, 370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/10_icon_9.19.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.19"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 58, 59)
    canvas.paste(_c11, (1319, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1319, 0, 1377, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 63, 62)
    canvas.paste(_c12, (1210, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1210, 0, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 41, 61)
    canvas.paste(_c13, (1273, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/14_icon_9.19.png
try:
    _c14 = get_crop(14, 56, 64)
    canvas.paste(_c14, (115, 0), _c14)
except Exception:
    pass
layout["9.19"] = [115, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/15_icon_deepcowl.png
try:
    _c15 = get_crop(15, 1344, 1175)
    canvas.paste(_c15, (48, 676), _c15)
except Exception:
    pass
layout["deepcowl"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 50, 60)
    canvas.paste(_c16, (383, 2), _c16)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/18_icon_Promoted.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (234, 1704), _c18)
except Exception:
    pass
layout["Promoted"] = [234, 1704, 378, 1848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/19_icon_2024.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["2024"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/20_icon_Online.png
try:
    _c20 = get_crop(20, 377, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/21_icon_Free.png
try:
    _c21 = get_crop(21, 124, 77)
    canvas.paste(_c21, (91, 2592), _c21)
except Exception:
    pass
layout["Free"] = [91, 2592, 215, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/22_icon_Free.png
try:
    _c22 = get_crop(22, 125, 78)
    canvas.paste(_c22, (90, 1368), _c22)
except Exception:
    pass
layout["Free"] = [90, 1368, 215, 1446]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/24_icon_2024.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["2024"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/25_icon_Building_a_7_Figure_Ecommerce_Business_i.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Building_a_7_Figure_Ecomm"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/26_text_9.19.png
try:
    _c26 = get_crop(26, 91, 43)
    canvas.paste(_c26, (20, 17), _c26)
except Exception:
    pass
layout["9.19"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/27_text_416_events.png
try:
    _c27 = get_crop(27, 372, 135)
    canvas.paste(_c27, (54, 390), _c27)
except Exception:
    pass
layout["416_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/28_text_Tow.png
try:
    _c28 = get_crop(28, 69, 37)
    canvas.paste(_c28, (106, 698), _c28)
except Exception:
    pass
layout["Tow"] = [106, 698, 175, 735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/29_text_SEo_TRACK.png
try:
    _c29 = get_crop(29, 152, 36)
    canvas.paste(_c29, (186, 700), _c29)
except Exception:
    pass
layout["SEo_TRACK"] = [186, 700, 338, 736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/30_text_deepaow.png
try:
    _c30 = get_crop(30, 149, 38)
    canvas.paste(_c30, (406, 700), _c30)
except Exception:
    pass
layout["deepaow"] = [406, 700, 555, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/31_text_Slo_TRaCK.png
try:
    _c31 = get_crop(31, 149, 36)
    canvas.paste(_c31, (584, 700), _c31)
except Exception:
    pass
layout["Slo_TRaCK"] = [584, 700, 733, 736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/32_text_deepciam.png
try:
    _c32 = get_crop(32, 147, 36)
    canvas.paste(_c32, (811, 700), _c32)
except Exception:
    pass
layout["deepciam"] = [811, 700, 958, 736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/33_text_seo_TracK.png
try:
    _c33 = get_crop(33, 150, 32)
    canvas.paste(_c33, (992, 703), _c33)
except Exception:
    pass
layout["seo_TracK"] = [992, 703, 1142, 735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/34_text_deepao.png
try:
    _c34 = get_crop(34, 121, 41)
    canvas.paste(_c34, (1212, 699), _c34)
except Exception:
    pass
layout["deepao"] = [1212, 699, 1333, 740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/35_text_SEO_MASTERCLASS_HOW_TO_RANKANY.png
try:
    _c35 = get_crop(35, 1344, 1175)
    canvas.paste(_c35, (48, 676), _c35)
except Exception:
    pass
layout["SEO_MASTERCLASS:_HOW_TO_R"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/36_text_WEBSITE.png
try:
    _c36 = get_crop(36, 255, 60)
    canvas.paste(_c36, (94, 1535), _c36)
except Exception:
    pass
layout["WEBSITE"] = [94, 1535, 349, 1595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/37_text_Wed_Mar_20_._5_00_PM_GMT.png
try:
    _c37 = get_crop(37, 144, 144)
    canvas.paste(_c37, (234, 1704), _c37)
except Exception:
    pass
layout["Wed,_Mar_20_._5:00_PM_GMT"] = [234, 1704, 378, 1848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/38_text_Online.png
try:
    _c38 = get_crop(38, 129, 45)
    canvas.paste(_c38, (91, 1687), _c38)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/39_text_IERCE_TraCKJUNI.png
try:
    _c39 = get_crop(39, 144, 144)
    canvas.paste(_c39, (234, 1704), _c39)
except Exception:
    pass
layout["IERCE_TraCKJUNI"] = [234, 1704, 378, 1848]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/40_text_Ecommerce_TracK_JUNI.png
try:
    _c40 = get_crop(40, 397, 39)
    canvas.paste(_c40, (436, 1927), _c40)
except Exception:
    pass
layout["Ecommerce_TracK_JUNI"] = [436, 1927, 833, 1966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/41_text_Ecommerce_TracK_JUNI.png
try:
    _c41 = get_crop(41, 399, 39)
    canvas.paste(_c41, (848, 1927), _c41)
except Exception:
    pass
layout["Ecommerce_TracK__JUNI"] = [848, 1927, 1247, 1966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/42_text_Eco.png
try:
    _c42 = get_crop(42, 64, 29)
    canvas.paste(_c42, (1260, 1931), _c42)
except Exception:
    pass
layout["Eco"] = [1260, 1931, 1324, 1960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/43_text_EZRA.png
try:
    _c43 = get_crop(43, 268, 97)
    canvas.paste(_c43, (690, 2089), _c43)
except Exception:
    pass
layout["EZRA"] = [690, 2089, 958, 2186]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/44_text_CEo_BoOmIby_CINdy_Joseph.png
try:
    _c44 = get_crop(44, 1344, 917)
    canvas.paste(_c44, (48, 1899), _c44)
except Exception:
    pass
layout["CEo,_BoOmIby_CINdy_Joseph"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/45_text_Zipify_Apps.png
try:
    _c45 = get_crop(45, 1344, 917)
    canvas.paste(_c45, (48, 1899), _c45)
except Exception:
    pass
layout["{_Zipify_Apps"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/46_text_ECOMMERCE_TracK_JUNI.png
try:
    _c46 = get_crop(46, 397, 38)
    canvas.paste(_c46, (154, 2501), _c46)
except Exception:
    pass
layout["ECOMMERCE_TracK_JUNI"] = [154, 2501, 551, 2539]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/47_text_EcoMMERCE_TracK.png
try:
    _c47 = get_crop(47, 1344, 917)
    canvas.paste(_c47, (48, 1899), _c47)
except Exception:
    pass
layout["EcoMMERCE_TracK"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/48_text_JUNI.png
try:
    _c48 = get_crop(48, 119, 38)
    canvas.paste(_c48, (843, 2501), _c48)
except Exception:
    pass
layout["JUNI"] = [843, 2501, 962, 2539]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/49_text_Building_a_7_Figure_Ecommerce_Business_i.png
try:
    _c49 = get_crop(49, 1344, 917)
    canvas.paste(_c49, (48, 1899), _c49)
except Exception:
    pass
layout["Building_a_7_Figure_Ecomm"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/50_text_2024.png
try:
    _c50 = get_crop(50, 288, 156)
    canvas.paste(_c50, (0, 2804), _c50)
except Exception:
    pass
layout["2024"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_05_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-7/51_clickable_Tickets.png
try:
    _c51 = get_crop(51, 288, 156)
    canvas.paste(_c51, (864, 2804), _c51)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]
