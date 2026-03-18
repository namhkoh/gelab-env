# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_03
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5.png
# step_index: 3/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall subtle off-white background
draw.rectangle([(0, 0), canvas.size], fill=(250, 250, 252))

# Status bar (top ~56px) - light gray background
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=(190, 190, 190))

# Header underline (search bar accent) - bright blue line under the search area
# Align to the detected horizontal margins (left 48, right 1392)
underline_y = 262
draw.rectangle([(48, underline_y), (1392, underline_y + 6)], fill=(34, 86, 201))

# Subtle divider under header area (thin)
draw.line([(48, underline_y + 12), (1392, underline_y + 12)], fill=(230, 230, 230), width=1)

# Event row card backgrounds (rounded rectangles) - one card per detected row
row_tops = [390, 786, 1182, 1578, 1974, 2370]
card_left = 48
card_right = 1392
card_height = 396
card_radius = 10
card_fill = (254, 254, 255)  # barely off-white so they read as cards against the page
card_outline = (235, 235, 238)

for top in row_tops:
    box = (card_left, top, card_right, top + card_height)
    # card background
    try:
        draw.rounded_rectangle(box, radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    except Exception:
        # fallback if rounded_rectangle not available
        draw.rectangle(box, fill=card_fill, outline=card_outline)
    # subtle bottom separator line for each card (reinforce rows)
    draw.line([(card_left + 8, top + card_height - 1), (card_right - 8, top + card_height - 1)], fill=(240, 240, 242), width=1)
    # placeholder thumbnail background (behind the thumbnail image that will be pasted)
    thumb_x = card_left + 4
    thumb_y = top + 24
    thumb_w = 140
    thumb_h = 104
    thumb_box = (thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h)
    draw.rectangle(thumb_box, fill=(245, 246, 248), outline=(230, 230, 234))

# Light separators across the content area between major groups
for sep_y in [card_t + card_height for card_t in row_tops]:
    draw.line([(24, sep_y + 8), (1416, sep_y + 8)], fill=(242, 242, 243), width=1)

# Bottom navigation bar background (area where icons will be pasted)
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
# top border of nav bar
draw.line([(24, nav_top), (1416, nav_top)], fill=(220, 220, 223), width=2)

# Small top-of-page left margin vertical guide (visual structure only)
draw.line([(48, status_h + 8), (48, 2800)], fill=(248, 248, 249), width=1)

# A faint right-side content margin guide to visually balance layout
draw.line([(1392, status_h + 8), (1392, 2800)], fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/00_icon_Business.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Business"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/01_icon_Business_Seminar.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 786), _c1)
except Exception:
    pass
layout["Business_Seminar"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/02_icon_8_1922_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1182), _c2)
except Exception:
    pass
layout["8_1922_creator_followers"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/03_icon_CENTER.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 786), _c3)
except Exception:
    pass
layout["CENTER"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/04_icon_Igreich_schre.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1974), _c4)
except Exception:
    pass
layout["Igreich_schre"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/05_icon_Business.png
try:
    _c5 = get_crop(5, 54, 63)
    canvas.paste(_c5, (314, 2), _c5)
except Exception:
    pass
layout["Business"] = [314, 2, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/06_icon_5.20.png
try:
    _c6 = get_crop(6, 53, 63)
    canvas.paste(_c6, (183, 2), _c6)
except Exception:
    pass
layout["5.20"] = [183, 2, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/07_icon_5.20.png
try:
    _c7 = get_crop(7, 60, 65)
    canvas.paste(_c7, (113, 1), _c7)
except Exception:
    pass
layout["5.20"] = [113, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/08_icon_Igreich_schre.png
try:
    _c8 = get_crop(8, 1344, 396)
    canvas.paste(_c8, (48, 1578), _c8)
except Exception:
    pass
layout["Igreich_schre"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/09_icon_inc.png
try:
    _c9 = get_crop(9, 102, 309)
    canvas.paste(_c9, (260, 2412), _c9)
except Exception:
    pass
layout["inc"] = [260, 2412, 362, 2721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 45, 59)
    canvas.paste(_c10, (252, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [252, 4, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/11_icon_5.20.png
try:
    _c11 = get_crop(11, 117, 102)
    canvas.paste(_c11, (58, 120), _c11)
except Exception:
    pass
layout["5.20"] = [58, 120, 175, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/12_icon_ani.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["ani"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/13_icon_less.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1182), _c13)
except Exception:
    pass
layout["less"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/14_icon_WEP_Online-Seminar_Businessplan.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1974), _c14)
except Exception:
    pass
layout["WEP_Online-Seminar:_Busin"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 95, 63)
    canvas.paste(_c15, (1215, 0), _c15)
except Exception:
    pass
layout["Cancel"] = [1215, 0, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/16_icon_Business_Seminar.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 390), _c16)
except Exception:
    pass
layout["Business_Seminar"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/17_icon_Online.png
try:
    _c17 = get_crop(17, 112, 53)
    canvas.paste(_c17, (390, 2210), _c17)
except Exception:
    pass
layout["Online"] = [390, 2210, 502, 2263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/18_icon_Online.png
try:
    _c18 = get_crop(18, 112, 51)
    canvas.paste(_c18, (390, 1024), _c18)
except Exception:
    pass
layout["Online"] = [390, 1024, 502, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (864, 2804), _c19)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/20_icon_erfolgreich_schreiben.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1578), _c20)
except Exception:
    pass
layout["erfolgreich_schreiben"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/21_icon_Thryv_Business_Tools_Online_Seminar.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["Thryv_Business_Tools_Onli"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/22_icon_Seminar.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 2370), _c22)
except Exception:
    pass
layout["Seminar"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 51, 62)
    canvas.paste(_c23, (1321, 1), _c23)
except Exception:
    pass
layout["Cancel"] = [1321, 1, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/24_icon_Cancel.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1099, 96), _c24)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/25_icon_Online.png
try:
    _c25 = get_crop(25, 116, 53)
    canvas.paste(_c25, (388, 1814), _c25)
except Exception:
    pass
layout["Online"] = [388, 1814, 504, 1867]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/26_icon_thryv.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["thryv"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/27_icon_Online.png
try:
    _c27 = get_crop(27, 112, 50)
    canvas.paste(_c27, (390, 1420), _c27)
except Exception:
    pass
layout["Online"] = [390, 1420, 502, 1470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/28_icon_Cancel.png
try:
    _c28 = get_crop(28, 149, 144)
    canvas.paste(_c28, (1243, 97), _c28)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/29_icon_Business_Seminar.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 390), _c29)
except Exception:
    pass
layout["Business_Seminar"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/30_icon_erfolgreich_schreiben.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1974), _c30)
except Exception:
    pass
layout["erfolgreich_schreiben"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/31_icon_Business.png
try:
    _c31 = get_crop(31, 46, 63)
    canvas.paste(_c31, (384, 2), _c31)
except Exception:
    pass
layout["Business"] = [384, 2, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/32_icon_thryv.png
try:
    _c32 = get_crop(32, 120, 121)
    canvas.paste(_c32, (97, 2593), _c32)
except Exception:
    pass
layout["thryv"] = [97, 2593, 217, 2714]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/33_text_5.20.png
try:
    _c33 = get_crop(33, 89, 43)
    canvas.paste(_c33, (22, 17), _c33)
except Exception:
    pass
layout["5.20"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/34_text_Events.png
try:
    _c34 = get_crop(34, 186, 56)
    canvas.paste(_c34, (46, 301), _c34)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/35_text_Sun.png
try:
    _c35 = get_crop(35, 89, 45)
    canvas.paste(_c35, (390, 2488), _c35)
except Exception:
    pass
layout["Sun,"] = [390, 2488, 479, 2533]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/36_text_12.png
try:
    _c36 = get_crop(36, 54, 39)
    canvas.paste(_c36, (552, 2489), _c36)
except Exception:
    pass
layout["12"] = [552, 2489, 606, 2528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/37_text_5_00_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2370), _c37)
except Exception:
    pass
layout["5:00_PM_EDT"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/38_text_Thryv_Business_Tools_Online_Seminar.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2370), _c38)
except Exception:
    pass
layout["Thryv_Business_Tools_Onli"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/39_text_Online.png
try:
    _c39 = get_crop(39, 112, 36)
    canvas.paste(_c39, (392, 2614), _c39)
except Exception:
    pass
layout["Online"] = [392, 2614, 504, 2650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_03_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-5/40_clickable_More.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (1152, 2804), _c40)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
