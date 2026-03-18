# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_08
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10.png
# step_index: 8/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#e6e7e8")

# Top banner / hero background (dark purple gradient)
banner_top = status_h
banner_h = 420
for i in range(banner_h):
    # gradient from deep purple to slightly lighter
    t = i / max(1, banner_h - 1)
    # colors interpolate between #2b0b2d and #4b1c58
    r = int((1 - t) * 0x2b + t * 0x4b)
    g = int((1 - t) * 0x0b + t * 0x1c)
    b = int((1 - t) * 0x2d + t * 0x58)
    draw.line([(0, banner_top + i), (1440, banner_top + i)], fill=(r, g, b))

# Subtle faded edges for banner (blurred vignette look using semi-transparent rectangles)
draw.rectangle([(-40, banner_top), (80, banner_top + banner_h)], fill=(0, 0, 0, 30))
draw.rectangle([(1360, banner_top), (1480, banner_top + banner_h)], fill=(0, 0, 0, 30))

# Large image overlay background (darker center band) behind the poster area
center_band_y0 = banner_top + 20
center_band_y1 = banner_top + banner_h - 20
draw.rectangle([(60, center_band_y0), (1380, center_band_y1)], fill="#371033")

# Main content area (white - canvas already white but add a slight warm white to match screenshot)
draw.rectangle([(0, banner_top + banner_h), (1440, 2960)], fill="#ffffff")

# Organizer card background (rounded) - leave content/icons to be pasted on top
card_x0 = 48
card_x1 = 1392
card_y0 = 1020
card_y1 = 1168
card_radius = 24
# subtle shadow
shadow_rect = [(card_x0 + 6, card_y0 + 8), (card_x1 + 6, card_y1 + 8)]
draw.rounded_rectangle(shadow_rect, radius=card_radius, fill="#e9e9ee")
# card fill
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=card_radius, fill="#f6f7fb")

# Thin divider lines
div_y_1 = 1700  # divider after refund area
draw.line([(48, div_y_1), (1392, div_y_1)], fill="#eef0f2", width=2)

div_y_2 = 2280  # divider after about section
draw.line([(48, div_y_2), (1392, div_y_2)], fill="#f1f2f4", width=2)

# Small subtle horizontal rule under header/title area (below banner)
header_rule_y = banner_top + banner_h - 6
draw.line([(48, header_rule_y), (1392, header_rule_y)], fill="#ffffff", width=2)

# Left vertical guide / margin subtle shading to separate content column
draw.rectangle([(36, banner_top + banner_h + 20), (44, 2600)], fill="#fbfbfc")

# Location section card background (very subtle rounded)
loc_card_y0 = 2380
loc_card_y1 = 2630
draw.rounded_rectangle([(48, loc_card_y0), (1392, loc_card_y1)], radius=12, fill="#ffffff", outline="#f0f1f3")

# Bottom sticky ticket bar background (keep clear area where the actual button will be pasted)
bottom_bar_top = 2680
bottom_bar_bottom = 2960
# draw a light top border/shadow
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill="#e6e6e8", width=3)
# left price area (do not draw over the Get tickets button region which starts at x ~822)
left_price_area = [(0, bottom_bar_top), (812, bottom_bar_bottom)]
draw.rectangle(left_price_area, fill="#ffffff")
# subtle shadow for left area
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_top + 6)], fill="#f7f7f8")

# Rounded white card behind location/address area (to lift it slightly)
addr_card_y0 = 2480
addr_card_y1 = 2650
draw.rounded_rectangle([(48, addr_card_y0), (1392, addr_card_y1)], radius=12, fill="#ffffff", outline="#f3f4f6")

# Accent separator dots / small separators for visual grouping (no icons/text)
sep_x = 48
for y in range(1320, 1620, 80):
    draw.line([(sep_x, y), (1392, y)], fill="#fbfbfb", width=1)

# Decorative faint bottom shadow across the whole canvas base
for i in range(12):
    alpha = int(6 - i * 0.5)
    if alpha > 0:
        draw.line([(0, bottom_bar_bottom - i - 1), (1440, bottom_bar_bottom - i - 1)], fill=(230, 230, 230, alpha))

# Subtle left card badge background (for where organizer avatar sits) - faint circle behind avatar position
avatar_center = (48 + 72, 1020 + 72)
draw.ellipse([(avatar_center[0] - 62, avatar_center[1] - 62), (avatar_center[0] + 62, avatar_center[1] + 62)], fill="#f3f4f7")

# Ensure header area corners are slightly rounded at top of content column
draw.rectangle([(0, banner_top + banner_h), (1440, banner_top + banner_h + 4)], fill="#ffffff")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/00_icon_Following.png
try:
    _c0 = get_crop(0, 398, 144)
    canvas.paste(_c0, (946, 1068), _c0)
except Exception:
    pass
layout["Following"] = [946, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/02_icon_Music.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2150), _c2)
except Exception:
    pass
layout["Music"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/03_icon_BOARDNER_S.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["BOARDNER'S"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/04_icon_5.35.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["5.35"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 69)
    canvas.paste(_c5, (1154, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 1, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 70, 72)
    canvas.paste(_c7, (306, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [306, 0, 376, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 71)
    canvas.paste(_c8, (247, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 0, 303, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 69)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/10_icon_5.35.png
try:
    _c10 = get_crop(10, 64, 71)
    canvas.paste(_c10, (179, 1), _c10)
except Exception:
    pass
layout["5.35"] = [179, 1, 243, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/11_icon_Show_map.png
try:
    _c11 = get_crop(11, 226, 144)
    canvas.paste(_c11, (1166, 2368), _c11)
except Exception:
    pass
layout["Show_map"] = [1166, 2368, 1392, 2512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 44, 65)
    canvas.paste(_c12, (1329, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [1329, 2, 1373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 105, 65)
    canvas.paste(_c13, (1212, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 1, 1317, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/14_icon_SLEAZE.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1116, 108), _c14)
except Exception:
    pass
layout["SLEAZE"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/15_text_5.35.png
try:
    _c15 = get_crop(15, 92, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["5.35"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/16_text_FRIDAY_APR_26TH.png
try:
    _c16 = get_crop(16, 322, 61)
    canvas.paste(_c16, (420, 96), _c16)
except Exception:
    pass
layout["FRIDAY_APR_26TH"] = [420, 96, 742, 157]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/17_text_BOARDNER_S.png
try:
    _c17 = get_crop(17, 234, 57)
    canvas.paste(_c17, (782, 97), _c17)
except Exception:
    pass
layout["BOARDNER'S"] = [782, 97, 1016, 154]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/18_text_Friday_April_26.png
try:
    _c18 = get_crop(18, 383, 78)
    canvas.paste(_c18, (39, 758), _c18)
except Exception:
    pass
layout["Friday;_April_26"] = [39, 758, 422, 836]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/19_text_9_30_PM.png
try:
    _c19 = get_crop(19, 209, 56)
    canvas.paste(_c19, (451, 766), _c19)
except Exception:
    pass
layout["9:30_PM"] = [451, 766, 660, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/20_text_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c20 = get_crop(20, 500, 144)
    canvas.paste(_c20, (288, 1028), _c20)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/21_text_Club_Decades_Presents.png
try:
    _c21 = get_crop(21, 500, 144)
    canvas.paste(_c21, (288, 1028), _c21)
except Exception:
    pass
layout["Club_Decades_Presents"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/22_text_4.7k_Followers.png
try:
    _c22 = get_crop(22, 500, 144)
    canvas.paste(_c22, (288, 1028), _c22)
except Exception:
    pass
layout["4.7k_Followers"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/23_text_Boardner_s_by_La_Belle.png
try:
    _c23 = get_crop(23, 1344, 144)
    canvas.paste(_c23, (48, 1295), _c23)
except Exception:
    pass
layout["Boardner's_by_La_Belle"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/24_text_days_4_hrs_30_mins.png
try:
    _c24 = get_crop(24, 405, 64)
    canvas.paste(_c24, (173, 1449), _c24)
except Exception:
    pass
layout["days_4_hrs_30_mins"] = [173, 1449, 578, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/25_text_Refund_policy.png
try:
    _c25 = get_crop(25, 299, 63)
    canvas.paste(_c25, (138, 1558), _c25)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/26_text_No_refunds.png
try:
    _c26 = get_crop(26, 214, 49)
    canvas.paste(_c26, (139, 1649), _c26)
except Exception:
    pass
layout["No_refunds"] = [139, 1649, 353, 1698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 452, 57)
    canvas.paste(_c27, (46, 1859), _c27)
except Exception:
    pass
layout["About_this_event"] = [46, 1859, 498, 1916]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/28_text_Indie_Sleaze_4_26.png
try:
    _c28 = get_crop(28, 234, 144)
    canvas.paste(_c28, (48, 2150), _c28)
except Exception:
    pass
layout["Indie_Sleaze_4_26"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/29_text_Club_Decades.png
try:
    _c29 = get_crop(29, 268, 45)
    canvas.paste(_c29, (417, 2094), _c29)
except Exception:
    pass
layout["Club_Decades"] = [417, 2094, 685, 2139]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/30_text_Read_more.png
try:
    _c30 = get_crop(30, 234, 144)
    canvas.paste(_c30, (48, 2150), _c30)
except Exception:
    pass
layout["Read_more"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/31_text_Location.png
try:
    _c31 = get_crop(31, 246, 61)
    canvas.paste(_c31, (41, 2413), _c31)
except Exception:
    pass
layout["Location"] = [41, 2413, 287, 2474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/32_text_Boardner_s_by_La_Belle.png
try:
    _c32 = get_crop(32, 486, 61)
    canvas.paste(_c32, (138, 2538), _c32)
except Exception:
    pass
layout["Boardner's_by_La_Belle"] = [138, 2538, 624, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/33_text_Boardner_s_by_La_Belle_1652_North_Cherok.png
try:
    _c33 = get_crop(33, 570, 144)
    canvas.paste(_c33, (822, 2768), _c33)
except Exception:
    pass
layout["Boardner's_by_La_Belle;_1"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/34_text_Anqeles_CA_90028.png
try:
    _c34 = get_crop(34, 420, 56)
    canvas.paste(_c34, (141, 2669), _c34)
except Exception:
    pass
layout["Anqeles,_CA_90028"] = [141, 2669, 561, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/35_text_S15_-_20.png
try:
    _c35 = get_crop(35, 228, 61)
    canvas.paste(_c35, (89, 2811), _c35)
except Exception:
    pass
layout["S15_-_$20"] = [89, 2811, 317, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_08_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-10/36_text_LLub.png
try:
    _c36 = get_crop(36, 144, 144)
    canvas.paste(_c36, (96, 1067), _c36)
except Exception:
    pass
layout["LLub"] = [96, 1067, 240, 1211]
