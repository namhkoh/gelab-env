# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_07
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9.png
# step_index: 7/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Eventbrite-like page.
# Available variables: canvas (PIL Image 1440x2960), draw (PIL.ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# 1) Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill=(232, 232, 232))  # light gray status bar background

# subtle bottom line for status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=(214, 214, 214), width=1)

# 2) Header/banner image area (gradient to simulate poster crop)
banner_top = status_h
banner_bottom = 540  # approximate height of the poster area
b_h = banner_bottom - banner_top
# gradient from dark purple to deep magenta to mimic poster edges
start = (30, 8, 45)    # deep purple
mid = (60, 18, 90)
end = (200, 110, 140)  # warmer tone near bottom
for i in range(b_h):
    t = i / max(b_h - 1, 1)
    # two-stage interpolation for a slightly richer look
    if t < 0.6:
        r = t / 0.6
        rcol = (
            int(start[0] * (1 - r) + mid[0] * r),
            int(start[1] * (1 - r) + mid[1] * r),
            int(start[2] * (1 - r) + mid[2] * r),
        )
    else:
        r = (t - 0.6) / 0.4
        rcol = (
            int(mid[0] * (1 - r) + end[0] * r),
            int(mid[1] * (1 - r) + end[1] * r),
            int(mid[2] * (1 - r) + end[2] * r),
        )
    y = banner_top + i
    draw.line([(0, y), (W, y)], fill=rcol)

# soft dark overlay at top portion of banner to emulate vignette
overlay_height = int(b_h * 0.35)
for i in range(overlay_height):
    alpha = int(30 * (1 - (i / max(overlay_height - 1, 1))))  # subtle darkening
    line_color = (10, 10, 10)
    # simulate alpha by blending against the existing banner color:
    y = banner_top + i
    # sample by drawing a very transparent darker line (approximation)
    draw.line([(0, y), (W, y)], fill=(int(line_color[0]*alpha/255), int(line_color[1]*alpha/255), int(line_color[2]*alpha/255)))

# 3) Main content background (white)
content_top = banner_bottom
draw.rectangle([(0, content_top), (W, H)], fill=(255, 255, 255))

# 4) Organizer card (rounded rectangle) - behind avatar and Follow button (do NOT draw the button or avatar)
card_margin_x = 48
card_top = 980
card_bottom = 1148
card_radius = 24
card_bbox = [card_margin_x, card_top, W - card_margin_x, card_bottom]
# subtle drop shadow (thin)
shadow_bbox = [card_bbox[0], card_bbox[1] + 6, card_bbox[2], card_bbox[3] + 6]
draw.rectangle(shadow_bbox, fill=(235, 235, 238))
# card
draw.rounded_rectangle(card_bbox, radius=card_radius, fill=(246, 247, 250), outline=None)

# 5) Thin separators and subtle section dividers
# divider under the organizer/card area
div_y = card_bottom + 60
draw.line([(card_margin_x, div_y), (W - card_margin_x, div_y)], fill=(235, 235, 238), width=1)

# divider after refund/policy section (approx)
draw.line([(card_margin_x, 1648), (W - card_margin_x, 1648)], fill=(242, 242, 244), width=1)

# divider before Location section
draw.line([(card_margin_x, 2060), (W - card_margin_x, 2060)], fill=(242, 242, 244), width=1)

# 6) "About this event" area background — keep it white but add subtle left accent (not text)
about_top = 1740
about_bottom = 1900
# light background band to imply section grouping
draw.rectangle([(0, about_top), (W, about_bottom)], fill=(255, 255, 255))
# subtle left accent bar
accent_w = 6
draw.rectangle([(card_margin_x, about_top + 18), (card_margin_x + accent_w, about_top + 58)], fill=(88, 30, 120))

# 7) Location section area - small pin icon will be pasted; provide only white background and divider already drawn
location_top = 2060
location_bottom = 2488
draw.rectangle([(0, location_top), (W, location_bottom)], fill=(255, 255, 255))

# 8) Bottom ticket bar background
bottom_bar_h = 260
bottom_bar_top = H - bottom_bar_h
# light neutral bar background spanning full width
draw.rectangle([(0, bottom_bar_top), (W, H)], fill=(250, 248, 249))
# subtle top border line for the bar
draw.line([(0, bottom_bar_top), (W, bottom_bar_top)], fill=(226, 224, 225), width=2)

# 9) Left price area background (behind price text which will be pasted) - keep minimal to avoid duplicating UI controls
price_area_box = [32, bottom_bar_top + 30, 520, H - 32]
# use transparentish fill approximation by slightly different color
draw.rectangle(price_area_box, fill=(250, 248, 249))

# 10) Right side has the orange button region for tickets; draw only the background band (the button itself will be pasted)
tickets_bg_box = [520, bottom_bar_top + 20, W - 32, H - 32]
# draw a subtle backdrop rectangle where the button lives (lighter tone so pasted button stands out)
draw.rectangle(tickets_bg_box, fill=(250, 245, 242))

# 11) Final thin separators across page for additional structure
for y in (880, 1220, 1520, 1820, 2240):
    draw.line([(card_margin_x, y), (W - card_margin_x, y)], fill=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/00_icon_Follow.png
try:
    _c0 = get_crop(0, 331, 144)
    canvas.paste(_c0, (1013, 1068), _c0)
except Exception:
    pass
layout["Follow"] = [1013, 1068, 1344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/02_icon_Music.png
try:
    _c2 = get_crop(2, 234, 144)
    canvas.paste(_c2, (48, 2150), _c2)
except Exception:
    pass
layout["Music"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/03_icon_BOARDNER_S.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1116, 108), _c3)
except Exception:
    pass
layout["BOARDNER'S"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/04_icon_5.35.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["5.35"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 69)
    canvas.paste(_c5, (1154, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 1, 1203, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/06_icon_Share.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1260, 108), _c6)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 56, 71)
    canvas.paste(_c7, (247, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [247, 0, 303, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 70, 71)
    canvas.paste(_c8, (306, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [306, 1, 376, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 68)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/10_icon_5.35.png
try:
    _c10 = get_crop(10, 64, 71)
    canvas.paste(_c10, (179, 1), _c10)
except Exception:
    pass
layout["5.35"] = [179, 1, 243, 72]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 44, 63)
    canvas.paste(_c11, (1329, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1329, 3, 1373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/12_icon_Show_map.png
try:
    _c12 = get_crop(12, 226, 144)
    canvas.paste(_c12, (1166, 2368), _c12)
except Exception:
    pass
layout["Show_map"] = [1166, 2368, 1392, 2512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/13_icon_WDIE.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1116, 108), _c13)
except Exception:
    pass
layout["WDIE"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 101, 64)
    canvas.paste(_c14, (1214, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 1, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/15_text_5.35.png
try:
    _c15 = get_crop(15, 92, 43)
    canvas.paste(_c15, (22, 17), _c15)
except Exception:
    pass
layout["5.35"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/16_text_Friday_April_26.png
try:
    _c16 = get_crop(16, 383, 78)
    canvas.paste(_c16, (39, 758), _c16)
except Exception:
    pass
layout["Friday;_April_26"] = [39, 758, 422, 836]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/17_text_9_30_PM.png
try:
    _c17 = get_crop(17, 209, 56)
    canvas.paste(_c17, (451, 766), _c17)
except Exception:
    pass
layout["9:30_PM"] = [451, 766, 660, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/18_text_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c18 = get_crop(18, 500, 144)
    canvas.paste(_c18, (288, 1028), _c18)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/19_text_Club_Decades_Presents.png
try:
    _c19 = get_crop(19, 500, 144)
    canvas.paste(_c19, (288, 1028), _c19)
except Exception:
    pass
layout["Club_Decades_Presents"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/20_text_4.7k_Followers.png
try:
    _c20 = get_crop(20, 500, 144)
    canvas.paste(_c20, (288, 1028), _c20)
except Exception:
    pass
layout["4.7k_Followers"] = [288, 1028, 788, 1172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/21_text_Boardner_s_by_La_Belle.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 1295), _c21)
except Exception:
    pass
layout["Boardner's_by_La_Belle"] = [48, 1295, 1392, 1439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/22_text_days_4_hrs_30_mins.png
try:
    _c22 = get_crop(22, 405, 64)
    canvas.paste(_c22, (173, 1449), _c22)
except Exception:
    pass
layout["days_4_hrs_30_mins"] = [173, 1449, 578, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/23_text_Refund_policy.png
try:
    _c23 = get_crop(23, 299, 63)
    canvas.paste(_c23, (138, 1558), _c23)
except Exception:
    pass
layout["Refund_policy"] = [138, 1558, 437, 1621]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/24_text_No_refunds.png
try:
    _c24 = get_crop(24, 214, 49)
    canvas.paste(_c24, (139, 1649), _c24)
except Exception:
    pass
layout["No_refunds"] = [139, 1649, 353, 1698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 452, 57)
    canvas.paste(_c25, (46, 1859), _c25)
except Exception:
    pass
layout["About_this_event"] = [46, 1859, 498, 1916]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/26_text_Indie_Sleaze_4_26.png
try:
    _c26 = get_crop(26, 234, 144)
    canvas.paste(_c26, (48, 2150), _c26)
except Exception:
    pass
layout["Indie_Sleaze_4_26"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/27_text_Club_Decades.png
try:
    _c27 = get_crop(27, 268, 45)
    canvas.paste(_c27, (417, 2094), _c27)
except Exception:
    pass
layout["Club_Decades"] = [417, 2094, 685, 2139]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/28_text_Read_more.png
try:
    _c28 = get_crop(28, 234, 144)
    canvas.paste(_c28, (48, 2150), _c28)
except Exception:
    pass
layout["Read_more"] = [48, 2150, 282, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/29_text_Location.png
try:
    _c29 = get_crop(29, 246, 61)
    canvas.paste(_c29, (41, 2413), _c29)
except Exception:
    pass
layout["Location"] = [41, 2413, 287, 2474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/30_text_Boardner_s_by_La_Belle.png
try:
    _c30 = get_crop(30, 486, 61)
    canvas.paste(_c30, (138, 2538), _c30)
except Exception:
    pass
layout["Boardner's_by_La_Belle"] = [138, 2538, 624, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/31_text_Boardner_s_by_La_Belle_1652_North_Cherok.png
try:
    _c31 = get_crop(31, 570, 144)
    canvas.paste(_c31, (822, 2768), _c31)
except Exception:
    pass
layout["Boardner's_by_La_Belle;_1"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/32_text_Anqeles_CA_90028.png
try:
    _c32 = get_crop(32, 420, 56)
    canvas.paste(_c32, (141, 2669), _c32)
except Exception:
    pass
layout["Anqeles,_CA_90028"] = [141, 2669, 561, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/33_text_S15_-_20.png
try:
    _c33 = get_crop(33, 228, 61)
    canvas.paste(_c33, (89, 2811), _c33)
except Exception:
    pass
layout["S15_-_$20"] = [89, 2811, 317, 2872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_07_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-9/34_text_LLub.png
try:
    _c34 = get_crop(34, 144, 144)
    canvas.paste(_c34, (96, 1067), _c34)
except Exception:
    pass
layout["LLub"] = [96, 1067, 240, 1211]
