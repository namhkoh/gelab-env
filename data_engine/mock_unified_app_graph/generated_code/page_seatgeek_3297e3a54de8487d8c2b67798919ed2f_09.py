# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_09
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12.png
# step_index: 9/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural chrome for the mobile page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (249, 250, 251)        # very light gray background
status_bar_color = (241, 241, 241)
header_color = (255, 255, 255)
divider_color = (224, 224, 224)
card_color = (255, 255, 255)
banner_black = (10, 10, 10)
banner_blue = (33, 119, 255)      # bright blue banner for second listing background
shadow_color = (235, 235, 235)

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Header / toolbar area (title and filters) - keep white with subtle bottom divider
header_top = status_h
header_bottom = 330
draw.rectangle((0, header_top, w, header_bottom), fill=header_color)
draw.line((24, header_bottom, w-24, header_bottom), fill=divider_color, width=1)

# Subtle shadow under header (soft separation)
draw.line((0, header_bottom+1, w, header_bottom+1), fill=(245,245,245), width=1)

# Primary hero/banner region behind first event (dark background)
# This sits below the filters and behind the large illustration image
banner_top = header_bottom
banner_mid = 1150
draw.rectangle((0, banner_top, w, banner_mid), fill=banner_black)

# First event details card (white rounded rectangle) below the hero/banner
card_margin = 24
card1_top = 1320
card1_bottom = 1520
card1_box = (card_margin, card1_top, w - card_margin, card1_bottom)

# subtle shadow for card
shadow_offset = 8
draw.rectangle((card1_box[0], card1_box[1]+shadow_offset, card1_box[2], card1_box[3]+shadow_offset), fill=shadow_color)
draw.rounded_rectangle(card1_box, radius=12, fill=card_color, outline=None)

# Divider line inside card (separating title/details from actions)
draw.line((card1_box[0]+20, card1_top+96, card1_box[2]-20, card1_top+96), fill=divider_color, width=1)

# Small separator under card (visual spacing)
draw.line((0, card1_bottom + 12, w, card1_bottom + 12), fill=(245,245,245), width=1)

# Second event hero/banner region (blue background behind second image)
banner2_top = card1_bottom + 24
banner2_bottom = 2140
draw.rectangle((0, banner2_top, w, banner2_bottom), fill=banner_blue)

# Second event details card
card2_top = banner2_bottom - 20  # slightly overlapping lower region
card2_bottom = card2_top + 200
card2_box = (card_margin, card2_top, w - card_margin, card2_bottom)
draw.rectangle((card2_box[0], card2_box[1]+shadow_offset, card2_box[2], card2_box[3]+shadow_offset), fill=shadow_color)
draw.rounded_rectangle(card2_box, radius=12, fill=card_color, outline=None)

# Divider line inside second card
draw.line((card2_box[0]+20, card2_top+96, card2_box[2]-20, card2_top+96), fill=divider_color, width=1)

# Separator under second card
draw.line((0, card2_bottom + 12, w, card2_bottom + 12), fill=(245,245,245), width=1)

# Third content/banner area (dark area hint for another image lower on page)
banner3_top = card2_bottom + 40
banner3_bottom = h
# Use very dark neutral, but keep it slightly textured by overlaying a faint gradient-like rectangle
draw.rectangle((0, banner3_top, w, banner3_bottom), fill=(30,30,30))
# Add a faint top divider for the third banner to separate from white card above
draw.line((0, banner3_top, w, banner3_top), fill=divider_color, width=1)

# Horizontal separators between logical sections (subtle)
sep_x0 = 24
sep_x1 = w - 24
separators = [
    header_bottom,         # under header
    banner_mid,            # mid banner edge
    card1_bottom + 12,     # after first card spacing
    banner2_bottom,        # second banner bottom
    card2_bottom + 12      # after second card spacing
]
for y in separators:
    if y < h:
        draw.line((sep_x0, y, sep_x1, y), fill=(245,245,245), width=1)

# Top rounded notch effect on header right side (to suggest toolbar controls area)
# subtle rounded rectangle on the right to indicate filter icon background (no icon drawn)
ctrl_w = 86
ctrl_h = 44
ctrl_x = w - card_margin - ctrl_w
ctrl_y = header_top + 18
draw.rounded_rectangle((ctrl_x, ctrl_y, ctrl_x+ctrl_w, ctrl_y+ctrl_h), radius=10, fill=(255,255,255), outline=divider_color)

# End of structural drawing. All icons/text/images are expected to be pasted on top of this background.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 294, 97)
    canvas.paste(_c0, (21, 335), _c0)
except Exception:
    pass
layout["Broadway"] = [21, 335, 315, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/01_icon_Dance.png
try:
    _c1 = get_crop(1, 224, 97)
    canvas.paste(_c1, (624, 335), _c1)
except Exception:
    pass
layout["Dance"] = [624, 335, 848, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/02_icon_Family.png
try:
    _c2 = get_crop(2, 221, 97)
    canvas.paste(_c2, (872, 335), _c2)
except Exception:
    pass
layout["Family"] = [872, 335, 1093, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/03_icon_Comedy.png
try:
    _c3 = get_crop(3, 261, 97)
    canvas.paste(_c3, (339, 335), _c3)
except Exception:
    pass
layout["Comedy"] = [339, 335, 600, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/04_icon_Cirque_Du_Sole.png
try:
    _c4 = get_crop(4, 323, 97)
    canvas.paste(_c4, (1117, 335), _c4)
except Exception:
    pass
layout["Cirque_Du_Sole"] = [1117, 335, 1440, 432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/05_icon_Track.png
try:
    _c5 = get_crop(5, 267, 185)
    canvas.paste(_c5, (0, 1382), _c5)
except Exception:
    pass
layout["Track"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/06_icon_Track.png
try:
    _c6 = get_crop(6, 267, 185)
    canvas.paste(_c6, (0, 2517), _c6)
except Exception:
    pass
layout["Track"] = [0, 2517, 267, 2702]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/07_icon_Share.png
try:
    _c7 = get_crop(7, 248, 162)
    canvas.paste(_c7, (267, 1398), _c7)
except Exception:
    pass
layout["Share"] = [267, 1398, 515, 1560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/08_icon_Los_Angeles_CA.png
try:
    _c8 = get_crop(8, 1440, 1135)
    canvas.paste(_c8, (0, 1591), _c8)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1591, 1440, 2726]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/09_icon_Share.png
try:
    _c9 = get_crop(9, 248, 162)
    canvas.paste(_c9, (267, 2533), _c9)
except Exception:
    pass
layout["Share"] = [267, 2533, 515, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/10_icon_884.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/11_icon_Comedy.png
try:
    _c11 = get_crop(11, 62, 58)
    canvas.paste(_c11, (242, 5), _c11)
except Exception:
    pass
layout["Comedy"] = [242, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/12_icon_7_11_mY.png
try:
    _c12 = get_crop(12, 60, 61)
    canvas.paste(_c12, (112, 2), _c12)
except Exception:
    pass
layout["7:11_mY"] = [112, 2, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/13_icon_7_11_mY.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (0, 72), _c13)
except Exception:
    pass
layout["7:11_mY"] = [0, 72, 144, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/14_icon_Comedy.png
try:
    _c14 = get_crop(14, 56, 60)
    canvas.paste(_c14, (313, 5), _c14)
except Exception:
    pass
layout["Comedy"] = [313, 5, 369, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/15_icon_884.png
try:
    _c15 = get_crop(15, 100, 62)
    canvas.paste(_c15, (1215, 2), _c15)
except Exception:
    pass
layout["884"] = [1215, 2, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/16_icon_7_11_mY.png
try:
    _c16 = get_crop(16, 50, 58)
    canvas.paste(_c16, (183, 4), _c16)
except Exception:
    pass
layout["7:11_mY"] = [183, 4, 233, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 49, 66)
    canvas.paste(_c17, (1153, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1153, 1, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 58)
    canvas.paste(_c18, (1320, 4), _c18)
except Exception:
    pass
layout["icon_18"] = [1320, 4, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/19_icon_Dance.png
try:
    _c19 = get_crop(19, 1440, 1135)
    canvas.paste(_c19, (0, 456), _c19)
except Exception:
    pass
layout["Dance"] = [0, 456, 1440, 1591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 45, 63)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 2, 429, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/21_icon_7_11_mY.png
try:
    _c21 = get_crop(21, 96, 62)
    canvas.paste(_c21, (10, 1), _c21)
except Exception:
    pass
layout["7:11_mY"] = [10, 1, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/22_text_Comedy.png
try:
    _c22 = get_crop(22, 251, 77)
    canvas.paste(_c22, (185, 132), _c22)
except Exception:
    pass
layout["Comedy"] = [185, 132, 436, 209]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/23_text_date.png
try:
    _c23 = get_crop(23, 116, 52)
    canvas.paste(_c23, (671, 208), _c23)
except Exception:
    pass
layout["date"] = [671, 208, 787, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/24_text_Netflix_Is_A_Joke.png
try:
    _c24 = get_crop(24, 267, 185)
    canvas.paste(_c24, (0, 1382), _c24)
except Exception:
    pass
layout["Netflix_Is_A_Joke"] = [0, 1382, 267, 1567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/25_text_Shane_Gillis.png
try:
    _c25 = get_crop(25, 290, 54)
    canvas.paste(_c25, (460, 1203), _c25)
except Exception:
    pass
layout["Shane_Gillis"] = [460, 1203, 750, 1257]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/26_text_Sat.png
try:
    _c26 = get_crop(26, 91, 49)
    canvas.paste(_c26, (42, 1277), _c26)
except Exception:
    pass
layout["Sat,"] = [42, 1277, 133, 1326]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/27_text_4_8_PM.png
try:
    _c27 = get_crop(27, 165, 48)
    canvas.paste(_c27, (243, 1281), _c27)
except Exception:
    pass
layout["4,8_PM"] = [243, 1281, 408, 1329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/28_text_Los_Angeles_CA.png
try:
    _c28 = get_crop(28, 348, 54)
    canvas.paste(_c28, (428, 1279), _c28)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [428, 1279, 776, 1333]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/29_text_The_Greek_Theatre.png
try:
    _c29 = get_crop(29, 406, 49)
    canvas.paste(_c29, (798, 1277), _c29)
except Exception:
    pass
layout["The_Greek_Theatre"] = [798, 1277, 1204, 1326]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/30_text_Los_A..png
try:
    _c30 = get_crop(30, 124, 43)
    canvas.paste(_c30, (1241, 1282), _c30)
except Exception:
    pass
layout["Los_A."] = [1241, 1282, 1365, 1325]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_09_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-12/31_text_May.png
try:
    _c31 = get_crop(31, 107, 64)
    canvas.paste(_c31, (135, 1273), _c31)
except Exception:
    pass
layout["May"] = [135, 1273, 242, 1337]
