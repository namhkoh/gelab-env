# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_08
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11.png
# step_index: 8/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top background (canvas is pre-created white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar (light gray strip at very top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#F2F2F2")

# Header / toolbar area under status bar
header_top = status_h
header_h = 120
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill="#FFFFFF")

# Header bottom divider
draw.line([(24, header_top + header_h), (1440 - 24, header_top + header_h)], fill="#E6E6E6", width=2)

# Helper to draw a card thumbnail area (rounded rect + subtle shadow)
def draw_card_thumbnail(x, y, w, h, color):
    # We'll draw the thumbnail occupying ~60% of the card height, leaving room for title/text below.
    thumb_h = int(h * 0.60)
    thumb_box = (x, y, x + w, y + thumb_h)
    radius = 26

    # Soft shadow
    shadow_offset = 8
    shadow_box = (thumb_box[0] + shadow_offset, thumb_box[1] + shadow_offset,
                  thumb_box[2] + shadow_offset, thumb_box[3] + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=radius, fill="#E9E9E9")

    # Thumbnail background
    draw.rounded_rectangle(thumb_box, radius=radius, fill=color)

# Draw Sports row thumbnails
sports_cards = [
    (48, 495, 462, 519, "#D7DDE2"),   # left card - light gray
    (546, 495, 462, 519, "#D93A34"),  # center card - red
    (1044, 495, 396, 519, "#1273B9")  # right card - blue
]
for x, y, w, h, col in sports_cards:
    draw_card_thumbnail(x, y, w, h, col)

# Separator under Sports section
sep_y1 = 495 + 519 + 28
draw.line([(24, sep_y1), (1440 - 24, sep_y1)], fill="#ECECEC", width=2)

# Draw Concerts row thumbnails
concerts_cards = [
    (48, 1247, 462, 519, "#2EBF7F"),  # left card - green
    (546, 1247, 462, 519, "#F2C94C"), # center card - yellow/orange
    (1044, 1247, 396, 533, "#9B59B6") # right card - purple
]
for x, y, w, h, col in concerts_cards:
    draw_card_thumbnail(x, y, w, h, col)

# Separator under Concerts section
sep_y2 = 1247 + 519 + 28
draw.line([(24, sep_y2), (1440 - 24, sep_y2)], fill="#ECECEC", width=2)

# Draw Broadway row thumbnails (darker / theatrical)
broadway_cards = [
    (48, 2015, 462, 533, "#0D0D0D"),   # dark image background
    (546, 2015, 462, 519, "#FF2D95"),  # pink-ish
    (1044, 2015, 396, 533, "#000000")  # black
]
for x, y, w, h, col in broadway_cards:
    draw_card_thumbnail(x, y, w, h, col)

# Separator under Broadway section
sep_y3 = 2015 + 533 + 28
draw.line([(24, sep_y3), (1440 - 24, sep_y3)], fill="#ECECEC", width=2)

# Draw faint section separators for top area (below header)
draw.line([(24, header_top + 48), (1440 - 24, header_top + 48)], fill="#F5F5F5", width=1)

# Subtle bottom gradient panel for bottom area (footer feel)
footer_top = 2760
draw.rectangle([(0, footer_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, footer_top), (1440 - 24, footer_top)], fill="#EDEDED", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/00_icon_S84.png
try:
    _c0 = get_crop(0, 396, 533)
    canvas.paste(_c0, (1044, 2015), _c0)
except Exception:
    pass
layout["S84+"] = [1044, 2015, 1440, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/01_icon_7_10_my.png
try:
    _c1 = get_crop(1, 56, 59)
    canvas.paste(_c1, (115, 4), _c1)
except Exception:
    pass
layout["7:10_my"] = [115, 4, 171, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/02_icon_View_all.png
try:
    _c2 = get_crop(2, 264, 183)
    canvas.paste(_c2, (1176, 2600), _c2)
except Exception:
    pass
layout["View_all"] = [1176, 2600, 1440, 2783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 58)
    canvas.paste(_c3, (243, 5), _c3)
except Exception:
    pass
layout["icon_3"] = [243, 5, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/04_icon_7_10_my.png
try:
    _c4 = get_crop(4, 48, 56)
    canvas.paste(_c4, (185, 5), _c4)
except Exception:
    pass
layout["7:10_my"] = [185, 5, 233, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/05_icon_888.png
try:
    _c5 = get_crop(5, 97, 62)
    canvas.paste(_c5, (1216, 2), _c5)
except Exception:
    pass
layout["888"] = [1216, 2, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/06_icon_S133.png
try:
    _c6 = get_crop(6, 462, 533)
    canvas.paste(_c6, (48, 2015), _c6)
except Exception:
    pass
layout["S133+"] = [48, 2015, 510, 2548]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 59)
    canvas.paste(_c7, (1321, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [1321, 4, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/08_icon_7_10_my.png
try:
    _c8 = get_crop(8, 144, 240)
    canvas.paste(_c8, (0, 72), _c8)
except Exception:
    pass
layout["7:10_my"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 58, 60)
    canvas.paste(_c9, (312, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [312, 5, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/10_icon_Sat_Sep_13_2025_7.3..png
try:
    _c10 = get_crop(10, 72, 72)
    canvas.paste(_c10, (906, 2807), _c10)
except Exception:
    pass
layout["Sat,_Sep_13,2025,7.3."] = [906, 2807, 978, 2879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/11_icon_888.png
try:
    _c11 = get_crop(11, 144, 240)
    canvas.paste(_c11, (1260, 72), _c11)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 47, 62)
    canvas.paste(_c12, (1154, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [1154, 3, 1201, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/13_icon_Comedy.png
try:
    _c13 = get_crop(13, 72, 72)
    canvas.paste(_c13, (408, 2807), _c13)
except Exception:
    pass
layout["Comedy"] = [408, 2807, 480, 2879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/14_icon_S22.png
try:
    _c14 = get_crop(14, 396, 519)
    canvas.paste(_c14, (1044, 495), _c14)
except Exception:
    pass
layout["S22+"] = [1044, 495, 1440, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/15_icon_S692.png
try:
    _c15 = get_crop(15, 396, 533)
    canvas.paste(_c15, (1044, 1247), _c15)
except Exception:
    pass
layout["S692+"] = [1044, 1247, 1440, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/16_text_Browse_by_category.png
try:
    _c16 = get_crop(16, 72, 72)
    canvas.paste(_c16, (408, 519), _c16)
except Exception:
    pass
layout["Browse_by_category"] = [408, 519, 480, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/17_text_Sports.png
try:
    _c17 = get_crop(17, 182, 73)
    canvas.paste(_c17, (39, 374), _c17)
except Exception:
    pass
layout["Sports"] = [39, 374, 221, 447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/18_text_View_all.png
try:
    _c18 = get_crop(18, 264, 183)
    canvas.paste(_c18, (1176, 312), _c18)
except Exception:
    pass
layout["View_all"] = [1176, 312, 1440, 495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/19_text_S103.png
try:
    _c19 = get_crop(19, 133, 52)
    canvas.paste(_c19, (95, 722), _c19)
except Exception:
    pass
layout["S103+"] = [95, 722, 228, 774]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/20_text_S114.png
try:
    _c20 = get_crop(20, 126, 52)
    canvas.paste(_c20, (592, 722), _c20)
except Exception:
    pass
layout["S114+"] = [592, 722, 718, 774]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/21_text_Dodgers_at_Padres.png
try:
    _c21 = get_crop(21, 462, 519)
    canvas.paste(_c21, (48, 495), _c21)
except Exception:
    pass
layout["Dodgers_at_Padres"] = [48, 495, 510, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/22_text_Reds_at_Dodgers.png
try:
    _c22 = get_crop(22, 462, 519)
    canvas.paste(_c22, (546, 495), _c22)
except Exception:
    pass
layout["Reds_at_Dodgers"] = [546, 495, 1008, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/23_text_Twins_at_Angels.png
try:
    _c23 = get_crop(23, 396, 519)
    canvas.paste(_c23, (1044, 495), _c23)
except Exception:
    pass
layout["Twins_at_Angels"] = [1044, 495, 1440, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/24_text_Sat.png
try:
    _c24 = get_crop(24, 92, 51)
    canvas.paste(_c24, (43, 917), _c24)
except Exception:
    pass
layout["Sat,"] = [43, 917, 135, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/25_text_11_5.40_PM.png
try:
    _c25 = get_crop(25, 221, 43)
    canvas.paste(_c25, (230, 922), _c25)
except Exception:
    pass
layout["11,5.40_PM"] = [230, 922, 451, 965]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/26_text_Thu.png
try:
    _c26 = get_crop(26, 101, 52)
    canvas.paste(_c26, (539, 917), _c26)
except Exception:
    pass
layout["Thu,"] = [539, 917, 640, 969]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/27_text_16.png
try:
    _c27 = get_crop(27, 56, 48)
    canvas.paste(_c27, (729, 918), _c27)
except Exception:
    pass
layout["16"] = [729, 918, 785, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/28_text_7_10_PM.png
try:
    _c28 = get_crop(28, 151, 43)
    canvas.paste(_c28, (797, 919), _c28)
except Exception:
    pass
layout["7:10_PM"] = [797, 919, 948, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/29_text_Sat_Apr_27_6.38_PM.png
try:
    _c29 = get_crop(29, 396, 519)
    canvas.paste(_c29, (1044, 495), _c29)
except Exception:
    pass
layout["Sat,_Apr_27,6.38_PM"] = [1044, 495, 1440, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/30_text_Concerts.png
try:
    _c30 = get_crop(30, 245, 65)
    canvas.paste(_c30, (42, 1124), _c30)
except Exception:
    pass
layout["Concerts"] = [42, 1124, 287, 1189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/31_text_View_all.png
try:
    _c31 = get_crop(31, 264, 183)
    canvas.paste(_c31, (1176, 1064), _c31)
except Exception:
    pass
layout["View_all"] = [1176, 1064, 1440, 1247]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/32_text_S262.png
try:
    _c32 = get_crop(32, 142, 54)
    canvas.paste(_c32, (95, 1473), _c32)
except Exception:
    pass
layout["S262+"] = [95, 1473, 237, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/33_text_S145.png
try:
    _c33 = get_crop(33, 135, 49)
    canvas.paste(_c33, (592, 1476), _c33)
except Exception:
    pass
layout["S145+"] = [592, 1476, 727, 1525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/34_text_Bruno_Mars.png
try:
    _c34 = get_crop(34, 462, 519)
    canvas.paste(_c34, (48, 1247), _c34)
except Exception:
    pass
layout["Bruno_Mars"] = [48, 1247, 510, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/35_text_Feid.png
try:
    _c35 = get_crop(35, 98, 50)
    canvas.paste(_c35, (541, 1600), _c35)
except Exception:
    pass
layout["Feid"] = [541, 1600, 639, 1650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/36_text_Stagecoach_Count.png
try:
    _c36 = get_crop(36, 396, 533)
    canvas.paste(_c36, (1044, 1247), _c36)
except Exception:
    pass
layout["Stagecoach_Count"] = [1044, 1247, 1440, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/37_text_Thu.png
try:
    _c37 = get_crop(37, 101, 54)
    canvas.paste(_c37, (42, 1667), _c37)
except Exception:
    pass
layout["Thu,"] = [42, 1667, 143, 1721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/38_text_15_8_PM.png
try:
    _c38 = get_crop(38, 168, 45)
    canvas.paste(_c38, (230, 1673), _c38)
except Exception:
    pass
layout["15,_8_PM"] = [230, 1673, 398, 1718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/39_text_Sat.png
try:
    _c39 = get_crop(39, 90, 45)
    canvas.paste(_c39, (542, 1673), _c39)
except Exception:
    pass
layout["Sat,"] = [542, 1673, 632, 1718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/40_text_27_8_PM.png
try:
    _c40 = get_crop(40, 177, 50)
    canvas.paste(_c40, (710, 1667), _c40)
except Exception:
    pass
layout["27,8_PM"] = [710, 1667, 887, 1717]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/41_text_Music_Festival_3_D.png
try:
    _c41 = get_crop(41, 396, 533)
    canvas.paste(_c41, (1044, 1247), _c41)
except Exception:
    pass
layout["Music_Festival_(3_D="] = [1044, 1247, 1440, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/42_text_Fri.png
try:
    _c42 = get_crop(42, 68, 48)
    canvas.paste(_c42, (1041, 1732), _c42)
except Exception:
    pass
layout["Fri,"] = [1041, 1732, 1109, 1780]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/43_text_26_Time_TBC.png
try:
    _c43 = get_crop(43, 264, 183)
    canvas.paste(_c43, (1176, 1832), _c43)
except Exception:
    pass
layout["26,_Time_TBC"] = [1176, 1832, 1440, 2015]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/44_text_Broadway_Shows.png
try:
    _c44 = get_crop(44, 72, 72)
    canvas.paste(_c44, (408, 2039), _c44)
except Exception:
    pass
layout["Broadway_Shows"] = [408, 2039, 480, 2111]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/45_text_View_all.png
try:
    _c45 = get_crop(45, 264, 183)
    canvas.paste(_c45, (1176, 1832), _c45)
except Exception:
    pass
layout["View_all"] = [1176, 1832, 1440, 2015]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/46_text_Comedy.png
try:
    _c46 = get_crop(46, 233, 80)
    canvas.paste(_c46, (40, 2653), _c46)
except Exception:
    pass
layout["Comedy"] = [40, 2653, 273, 2733]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/47_text_View_all.png
try:
    _c47 = get_crop(47, 264, 183)
    canvas.paste(_c47, (1176, 2600), _c47)
except Exception:
    pass
layout["View_all"] = [1176, 2600, 1440, 2783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/48_clickable_Tracking.png
try:
    _c48 = get_crop(48, 72, 72)
    canvas.paste(_c48, (906, 519), _c48)
except Exception:
    pass
layout["Tracking"] = [906, 519, 978, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/49_clickable_Tracking.png
try:
    _c49 = get_crop(49, 462, 519)
    canvas.paste(_c49, (546, 1247), _c49)
except Exception:
    pass
layout["Tracking"] = [546, 1247, 1008, 1766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/50_clickable_Tracking.png
try:
    _c50 = get_crop(50, 72, 72)
    canvas.paste(_c50, (408, 1271), _c50)
except Exception:
    pass
layout["Tracking"] = [408, 1271, 480, 1343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/51_clickable_Tracking.png
try:
    _c51 = get_crop(51, 72, 72)
    canvas.paste(_c51, (906, 1271), _c51)
except Exception:
    pass
layout["Tracking"] = [906, 1271, 978, 1343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/52_clickable_Tracking.png
try:
    _c52 = get_crop(52, 462, 519)
    canvas.paste(_c52, (546, 2015), _c52)
except Exception:
    pass
layout["Tracking"] = [546, 2015, 1008, 2534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_08_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-11/53_clickable_Tracking.png
try:
    _c53 = get_crop(53, 72, 72)
    canvas.paste(_c53, (906, 2039), _c53)
except Exception:
    pass
layout["Tracking"] = [906, 2039, 978, 2111]
