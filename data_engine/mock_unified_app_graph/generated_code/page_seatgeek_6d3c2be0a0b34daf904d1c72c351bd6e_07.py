# page_id: page_seatgeek_6d3c2be0a0b34daf904d1c72c351bd6e_07
# screenshot: 2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10.png
# step_index: 7/9
# task: Open SeatGeek. Look up "Phoenix Suns" tickets for next upcoming event. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant app background)
draw.rectangle([(0, 0), (1440, 2960)], fill=(242, 244, 246))

# STATUS BAR (top area)
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill=(222, 224, 226))

# Header / top pill (rounded) behind title and icons (do not draw text/icons)
header_rect = (40, 80, 1400, 220)
draw.rounded_rectangle(header_rect, radius=60, fill=(249, 250, 251), outline=(210, 213, 216), width=2)

# Subtle divider line below header pill
draw.line([(40, 232), (1400, 232)], fill=(224, 226, 228), width=1)

# Large map/content area background (rounded card behind arena diagram)
map_rect = (80, 300, 1360, 1560)
draw.rounded_rectangle(map_rect, radius=28, fill=(235, 237, 239), outline=(195, 197, 199), width=3)

# Thin divider above modal sheet area
sheet_top_guess = 1700
draw.line([(20, sheet_top_guess), (1420, sheet_top_guess)], fill=(220, 222, 224), width=1)

# Modal sheet shadow (soft band) to imply elevation
shadow_top = sheet_top_guess - 30
draw.rectangle([(40, shadow_top), (1400, sheet_top_guess)], fill=(210, 212, 214))

# Modal sheet (rounded white panel)
sheet_rect = (30, sheet_top_guess, 1410, 2960)
draw.rounded_rectangle(sheet_rect, radius=36, fill=(255, 255, 255), outline=None)

# Grabber bar at top of modal sheet
grabber_w = 140
grabber_h = 10
grabber_x = (1440 - grabber_w) // 2
grabber_y = sheet_top_guess + 14
draw.rounded_rectangle(
    [(grabber_x, grabber_y), (grabber_x + grabber_w, grabber_y + grabber_h)],
    radius=6,
    fill=(220, 222, 224),
)

# Cards inside modal sheet (rounded rect backgrounds and borders)
card_x1 = 60
card_x2 = 1380
# Card 1 (Deal Score) - subtle border
card1 = (card_x1, sheet_top_guess + 100, card_x2, sheet_top_guess + 340)
draw.rounded_rectangle(card1, radius=20, fill=(255, 255, 255), outline=(225, 227, 229), width=2)

# Card 2 (Price) - selected style with darker/bold outline
card2 = (card_x1, sheet_top_guess + 380, card_x2, sheet_top_guess + 620)
draw.rounded_rectangle(card2, radius=20, fill=(255, 255, 255), outline=(18, 18, 18), width=4)

# Card 3 (Best Seats) - subtle border
card3 = (card_x1, sheet_top_guess + 660, card_x2, sheet_top_guess + 900)
draw.rounded_rectangle(card3, radius=20, fill=(255, 255, 255), outline=(225, 227, 229), width=2)

# Inner separators / subtle shadows between card groups for depth
sep_x1 = card_x1 + 12
sep_x2 = card_x2 - 12
draw.line([(sep_x1, card1[3] + 22), (sep_x2, card1[3] + 22)], fill=(242, 244, 245), width=1)
draw.line([(sep_x1, card2[3] + 22), (sep_x2, card2[3] + 22)], fill=(242, 244, 245), width=1)

# Subtle bottom area tint to match screenshot gradient feel
draw.rectangle([(0, 2880), (1440, 2960)], fill=(250, 250, 250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/00_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c0 = get_crop(0, 1320, 267)
    canvas.paste(_c0, (60, 2318), _c0)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 2318, 1380, 2585]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/01_icon_Include.png
try:
    _c1 = get_crop(1, 341, 117)
    canvas.paste(_c1, (537, 309), _c1)
except Exception:
    pass
layout["Include"] = [537, 309, 878, 426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/02_icon_Courtside.png
try:
    _c2 = get_crop(2, 295, 119)
    canvas.paste(_c2, (908, 308), _c2)
except Exception:
    pass
layout["Courtside"] = [908, 308, 1203, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/03_icon_Quantity.png
try:
    _c3 = get_crop(3, 280, 120)
    canvas.paste(_c3, (231, 307), _c3)
except Exception:
    pass
layout["Quantity"] = [231, 307, 511, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/04_icon_Tit.png
try:
    _c4 = get_crop(4, 169, 121)
    canvas.paste(_c4, (37, 308), _c4)
except Exception:
    pass
layout["Tit"] = [37, 308, 206, 429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/05_icon_Center.png
try:
    _c5 = get_crop(5, 211, 119)
    canvas.paste(_c5, (1229, 308), _c5)
except Exception:
    pass
layout["Center"] = [1229, 308, 1440, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/06_icon_We_rate_each_ticket_based_on_location_pr.png
try:
    _c6 = get_crop(6, 1320, 329)
    canvas.paste(_c6, (60, 1941), _c6)
except Exception:
    pass
layout["We_rate_each_ticket_based"] = [60, 1941, 1380, 2270]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/07_icon_7.07_my.png
try:
    _c7 = get_crop(7, 67, 63)
    canvas.paste(_c7, (110, 1), _c7)
except Exception:
    pass
layout["7.07_my"] = [110, 1, 177, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 63, 60)
    canvas.paste(_c8, (241, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [241, 3, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 57, 62)
    canvas.paste(_c9, (312, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [312, 3, 369, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/10_icon_Center.png
try:
    _c10 = get_crop(10, 106, 111)
    canvas.paste(_c10, (1253, 145), _c10)
except Exception:
    pass
layout["Center"] = [1253, 145, 1359, 256]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/11_icon_7.07_my.png
try:
    _c11 = get_crop(11, 54, 61)
    canvas.paste(_c11, (182, 2), _c11)
except Exception:
    pass
layout["7.07_my"] = [182, 2, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 65)
    canvas.paste(_c12, (1152, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [1152, 1, 1205, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 55)
    canvas.paste(_c13, (1320, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [1320, 5, 1372, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 48, 65)
    canvas.paste(_c14, (383, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [383, 1, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 83, 59)
    canvas.paste(_c15, (1234, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [1234, 3, 1317, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/16_icon_W_Conf_Ist_Rnd_Suns_at_Timberwolves_Gm_2.png
try:
    _c16 = get_crop(16, 1360, 162)
    canvas.paste(_c16, (41, 120), _c16)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Suns_at_T"] = [41, 120, 1401, 282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/17_text_-228.png
try:
    _c17 = get_crop(17, 48, 27)
    canvas.paste(_c17, (467, 858), _c17)
except Exception:
    pass
layout["-228"] = [467, 858, 515, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/18_text_-230.png
try:
    _c18 = get_crop(18, 50, 27)
    canvas.paste(_c18, (615, 858), _c18)
except Exception:
    pass
layout["-230"] = [615, 858, 665, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/19_text_232.png
try:
    _c19 = get_crop(19, 48, 27)
    canvas.paste(_c19, (772, 858), _c19)
except Exception:
    pass
layout["232"] = [772, 858, 820, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/20_text_234.png
try:
    _c20 = get_crop(20, 48, 27)
    canvas.paste(_c20, (920, 858), _c20)
except Exception:
    pass
layout["234"] = [920, 858, 968, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/21_text_226.png
try:
    _c21 = get_crop(21, 48, 29)
    canvas.paste(_c21, (361, 886), _c21)
except Exception:
    pass
layout["226"] = [361, 886, 409, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/22_text_236.png
try:
    _c22 = get_crop(22, 48, 29)
    canvas.paste(_c22, (1029, 886), _c22)
except Exception:
    pass
layout["236"] = [1029, 886, 1077, 915]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/23_text_B32.png
try:
    _c23 = get_crop(23, 48, 25)
    canvas.paste(_c23, (476, 902), _c23)
except Exception:
    pass
layout["B32"] = [476, 902, 524, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/24_text_S40.png
try:
    _c24 = get_crop(24, 48, 30)
    canvas.paste(_c24, (647, 899), _c24)
except Exception:
    pass
layout["S40"] = [647, 899, 695, 929]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/25_text_S41.png
try:
    _c25 = get_crop(25, 45, 28)
    canvas.paste(_c25, (724, 899), _c25)
except Exception:
    pass
layout["S41_"] = [724, 899, 769, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/26_text_B5O.png
try:
    _c26 = get_crop(26, 51, 28)
    canvas.paste(_c26, (781, 899), _c26)
except Exception:
    pass
layout["B5O_"] = [781, 899, 832, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/27_text_B5Z.png
try:
    _c27 = get_crop(27, 45, 25)
    canvas.paste(_c27, (916, 902), _c27)
except Exception:
    pass
layout["B5Z"] = [916, 902, 961, 927]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/28_text_129.png
try:
    _c28 = get_crop(28, 46, 29)
    canvas.paste(_c28, (543, 939), _c28)
except Exception:
    pass
layout["129"] = [543, 939, 589, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/29_text_133.png
try:
    _c29 = get_crop(29, 46, 29)
    canvas.paste(_c29, (846, 939), _c29)
except Exception:
    pass
layout["133"] = [846, 939, 892, 968]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/30_text_523.png
try:
    _c30 = get_crop(30, 45, 27)
    canvas.paste(_c30, (347, 962), _c30)
except Exception:
    pass
layout["523"] = [347, 962, 392, 989]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/31_text_S62.png
try:
    _c31 = get_crop(31, 52, 31)
    canvas.paste(_c31, (1043, 961), _c31)
except Exception:
    pass
layout["S62"] = [1043, 961, 1095, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/32_text_224.png
try:
    _c32 = get_crop(32, 48, 28)
    canvas.paste(_c32, (245, 1003), _c32)
except Exception:
    pass
layout["224"] = [245, 1003, 293, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/33_text_238.png
try:
    _c33 = get_crop(33, 45, 30)
    canvas.paste(_c33, (1147, 1001), _c33)
except Exception:
    pass
layout["238"] = [1147, 1001, 1192, 1031]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/34_text_124.png
try:
    _c34 = get_crop(34, 48, 27)
    canvas.paste(_c34, (312, 1031), _c34)
except Exception:
    pass
layout["124"] = [312, 1031, 360, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/35_text_138.png
try:
    _c35 = get_crop(35, 46, 27)
    canvas.paste(_c35, (1082, 1031), _c35)
except Exception:
    pass
layout["138"] = [1082, 1031, 1128, 1058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/36_text_124.png
try:
    _c36 = get_crop(36, 46, 27)
    canvas.paste(_c36, (372, 1054), _c36)
except Exception:
    pass
layout["124"] = [372, 1054, 418, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/37_text_138.png
try:
    _c37 = get_crop(37, 46, 27)
    canvas.paste(_c37, (1029, 1050), _c37)
except Exception:
    pass
layout["138"] = [1029, 1050, 1075, 1077]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/38_text_S71.png
try:
    _c38 = get_crop(38, 43, 30)
    canvas.paste(_c38, (1133, 1068), _c38)
except Exception:
    pass
layout["S71"] = [1133, 1068, 1176, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/39_text_222.png
try:
    _c39 = get_crop(39, 48, 27)
    canvas.paste(_c39, (139, 1117), _c39)
except Exception:
    pass
layout["222"] = [139, 1117, 187, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/40_text_222.png
try:
    _c40 = get_crop(40, 46, 27)
    canvas.paste(_c40, (215, 1117), _c40)
except Exception:
    pass
layout["222"] = [215, 1117, 261, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/41_text_VISITORS.png
try:
    _c41 = get_crop(41, 80, 21)
    canvas.paste(_c41, (570, 1115), _c41)
except Exception:
    pass
layout["VISITORS"] = [570, 1115, 650, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/42_text_ISCORERS.png
try:
    _c42 = get_crop(42, 85, 21)
    canvas.paste(_c42, (676, 1115), _c42)
except Exception:
    pass
layout["ISCORERS"] = [676, 1115, 761, 1136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/43_text_240.png
try:
    _c43 = get_crop(43, 48, 27)
    canvas.paste(_c43, (1177, 1117), _c43)
except Exception:
    pass
layout["240"] = [1177, 1117, 1225, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/44_text_240.png
try:
    _c44 = get_crop(44, 48, 27)
    canvas.paste(_c44, (1251, 1117), _c44)
except Exception:
    pass
layout["240"] = [1251, 1117, 1299, 1144]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/45_text_CS.png
try:
    _c45 = get_crop(45, 34, 27)
    canvas.paste(_c45, (539, 1165), _c45)
except Exception:
    pass
layout["CS"] = [539, 1165, 573, 1192]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/46_text_iCS.png
try:
    _c46 = get_crop(46, 51, 29)
    canvas.paste(_c46, (839, 1161), _c46)
except Exception:
    pass
layout["iCS"] = [839, 1161, 890, 1190]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/47_text_221.png
try:
    _c47 = get_crop(47, 41, 27)
    canvas.paste(_c47, (199, 1193), _c47)
except Exception:
    pass
layout["221"] = [199, 1193, 240, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/48_text_TI8.png
try:
    _c48 = get_crop(48, 39, 25)
    canvas.paste(_c48, (263, 1179), _c48)
except Exception:
    pass
layout["TI8"] = [263, 1179, 302, 1204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/49_text_S74.png
try:
    _c49 = get_crop(49, 46, 29)
    canvas.paste(_c49, (1133, 1177), _c49)
except Exception:
    pass
layout["S74"] = [1133, 1177, 1179, 1206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/50_text_201.png
try:
    _c50 = get_crop(50, 46, 27)
    canvas.paste(_c50, (1193, 1193), _c50)
except Exception:
    pass
layout["201"] = [1193, 1193, 1239, 1220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/51_text_220.png
try:
    _c51 = get_crop(51, 48, 28)
    canvas.paste(_c51, (215, 1269), _c51)
except Exception:
    pass
layout["220"] = [215, 1269, 263, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/52_text_CS5.png
try:
    _c52 = get_crop(52, 46, 21)
    canvas.paste(_c52, (646, 1284), _c52)
except Exception:
    pass
layout["CS5"] = [646, 1284, 692, 1305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/53_text_202.png
try:
    _c53 = get_crop(53, 46, 30)
    canvas.paste(_c53, (1177, 1267), _c53)
except Exception:
    pass
layout["202"] = [1177, 1267, 1223, 1297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/54_text_113.png
try:
    _c54 = get_crop(54, 46, 27)
    canvas.paste(_c54, (550, 1325), _c54)
except Exception:
    pass
layout["113"] = [550, 1325, 596, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/55_text_219.png
try:
    _c55 = get_crop(55, 46, 27)
    canvas.paste(_c55, (113, 1346), _c55)
except Exception:
    pass
layout["219"] = [113, 1346, 159, 1373]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/56_text_18.png
try:
    _c56 = get_crop(56, 34, 27)
    canvas.paste(_c56, (307, 1339), _c56)
except Exception:
    pass
layout["18"] = [307, 1339, 341, 1366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/57_text_104.png
try:
    _c57 = get_crop(57, 46, 27)
    canvas.paste(_c57, (1098, 1341), _c57)
except Exception:
    pass
layout["104"] = [1098, 1341, 1144, 1368]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/58_text_203.png
try:
    _c58 = get_crop(58, 46, 28)
    canvas.paste(_c58, (1281, 1343), _c58)
except Exception:
    pass
layout["203"] = [1281, 1343, 1327, 1371]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/59_text_218.png
try:
    _c59 = get_crop(59, 48, 30)
    canvas.paste(_c59, (245, 1387), _c59)
except Exception:
    pass
layout["218"] = [245, 1387, 293, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/60_text_204.png
try:
    _c60 = get_crop(60, 48, 27)
    canvas.paste(_c60, (1147, 1390), _c60)
except Exception:
    pass
layout["204"] = [1147, 1390, 1195, 1417]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/61_text_522.png
try:
    _c61 = get_crop(61, 48, 29)
    canvas.paste(_c61, (347, 1429), _c61)
except Exception:
    pass
layout["522"] = [347, 1429, 395, 1458]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/62_text_217.png
try:
    _c62 = get_crop(62, 45, 28)
    canvas.paste(_c62, (287, 1454), _c62)
except Exception:
    pass
layout["217"] = [287, 1454, 332, 1482]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/63_text_CHAIRMANS.png
try:
    _c63 = get_crop(63, 110, 25)
    canvas.paste(_c63, (664, 1482), _c63)
except Exception:
    pass
layout["~CHAIRMANS"] = [664, 1482, 774, 1507]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/64_text_216.png
try:
    _c64 = get_crop(64, 48, 27)
    canvas.paste(_c64, (361, 1505), _c64)
except Exception:
    pass
layout["216"] = [361, 1505, 409, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/65_text_C1Z.png
try:
    _c65 = get_crop(65, 48, 28)
    canvas.paste(_c65, (448, 1491), _c65)
except Exception:
    pass
layout["C1Z"] = [448, 1491, 496, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/66_text_C15.png
try:
    _c66 = get_crop(66, 48, 28)
    canvas.paste(_c66, (525, 1491), _c66)
except Exception:
    pass
layout["C15_"] = [525, 1491, 573, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/67_text_C13.png
try:
    _c67 = get_crop(67, 48, 28)
    canvas.paste(_c67, (601, 1491), _c67)
except Exception:
    pass
layout["C13_"] = [601, 1491, 649, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/68_text_C9.png
try:
    _c68 = get_crop(68, 34, 28)
    canvas.paste(_c68, (835, 1491), _c68)
except Exception:
    pass
layout["C9"] = [835, 1491, 869, 1519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/69_text_206.png
try:
    _c69 = get_crop(69, 48, 27)
    canvas.paste(_c69, (1031, 1505), _c69)
except Exception:
    pass
layout["206"] = [1031, 1505, 1079, 1532]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/70_text_214.png
try:
    _c70 = get_crop(70, 48, 27)
    canvas.paste(_c70, (472, 1535), _c70)
except Exception:
    pass
layout["214"] = [472, 1535, 520, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/71_text_208.png
try:
    _c71 = get_crop(71, 48, 27)
    canvas.paste(_c71, (927, 1535), _c71)
except Exception:
    pass
layout["208"] = [927, 1535, 975, 1562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/72_text_213.png
try:
    _c72 = get_crop(72, 48, 29)
    canvas.paste(_c72, (541, 1552), _c72)
except Exception:
    pass
layout["213"] = [541, 1552, 589, 1581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/73_text_Sort_by.png
try:
    _c73 = get_crop(73, 188, 68)
    canvas.paste(_c73, (626, 1740), _c73)
except Exception:
    pass
layout["Sort_by"] = [626, 1740, 814, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/74_text_Best_Seats.png
try:
    _c74 = get_crop(74, 269, 55)
    canvas.paste(_c74, (118, 2703), _c74)
except Exception:
    pass
layout["Best_Seats"] = [118, 2703, 387, 2758]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6d3c2be0a0b34daf904d1c72c351bd6e/step_07_2024_4_22_19_6_6d3c2be0a0b34daf904d1c72c351bd6e-10/75_text_Get_close_to_the_action_with_the_best_se.png
try:
    _c75 = get_crop(75, 1320, 267)
    canvas.paste(_c75, (60, 2633), _c75)
except Exception:
    pass
layout["Get_close_to_the_action_w"] = [60, 2633, 1380, 2900]
