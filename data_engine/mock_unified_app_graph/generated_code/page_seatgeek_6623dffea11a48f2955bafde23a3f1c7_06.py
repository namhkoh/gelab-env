# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_06
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9.png
# step_index: 6/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall page background (dominant color: white/off-white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top ~90px) - subtle off-white so icons pasted later remain visible
status_h = 90
draw.rectangle((0, 0, 1440, status_h), fill=(250, 250, 250))

# Top app header / toolbar (below status bar)
header_y0 = status_h
header_y1 = header_y0 + 86
draw.rectangle((0, header_y0, 1440, header_y1), fill=(255, 255, 255))

# Thin divider / shadow under header
draw.line((24, header_y1, 1440 - 24, header_y1), fill=(230, 230, 230), width=1)
# A faint second line to give subtle depth
draw.line((24, header_y1 + 1, 1440 - 24, header_y1 + 1), fill=(245, 245, 245), width=1)

# Top summary card background (behind the event summary at y ~292)
top_card_x0 = 24
top_card_x1 = 1440 - 24
top_card_y0 = 220
top_card_y1 = top_card_y0 + 420
draw.rounded_rectangle((top_card_x0, top_card_y0, top_card_x1, top_card_y1),
                       radius=20, fill=(248, 248, 248), outline=None)

# Slight inner highlight band on the card to suggest elevation
draw.line((top_card_x0 + 2, top_card_y0 + 2, top_card_x1 - 2, top_card_y0 + 2), fill=(255, 255, 255), width=1)

# Define the major list row top positions (these areas will have cropped content pasted on top)
row_tops = [804, 1171, 1538, 1905, 2272, 2639]

# Draw subtle rounded background blocks for each list item (these are the "section card backgrounds")
card_margin_x = 24
card_width_right = 1440 - card_margin_x
card_height = 340  # approximate height of each item card (detected crops are 367 high)
for i, top in enumerate(row_tops):
    y0 = top - 16
    y1 = y0 + card_height
    # Alternate very subtle background tint for visual separation
    if i % 2 == 0:
        fill_col = (255, 255, 255)
    else:
        fill_col = (250, 250, 250)
    draw.rounded_rectangle((card_margin_x, y0, card_width_right, y1),
                           radius=18, fill=fill_col, outline=None)
    # Inner bottom divider for each card
    draw.line((card_margin_x + 8, y1, card_width_right - 8, y1), fill=(240, 240, 240), width=1)

# Large vertical content area background (keeps the page feeling continuous)
content_y0 = header_y1 + 12
content_y1 = 2639  # leave space above bottom nav area which will be pasted over
draw.rectangle((0, content_y0, 1440, content_y1), fill=(255, 255, 255))

# Draw separators between major sections (subtle)
# For example, separator above the "All Games" list (placed just above first row)
if row_tops:
    sep_y = row_tops[0] - 36
    draw.line((24, sep_y, 1440 - 24, sep_y), fill=(235, 235, 235), width=1)

# Draw thin separators at each row boundary to visually group items
for top in row_tops:
    # top boundary
    draw.line((24, top - 6, 1440 - 24, top - 6), fill=(245, 245, 245), width=1)
    # bottom boundary (approx)
    draw.line((24, top + 367 - 6, 1440 - 24, top + 367 - 6), fill=(245, 245, 245), width=1)

# Divider above bottom navigation area (leave the nav icons/text crops untouched)
bottom_nav_top = 2639
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill=(230, 230, 230), width=1)
draw.line((0, bottom_nav_top + 1, 1440, bottom_nav_top + 1), fill=(245, 245, 245), width=1)

# Subtle left page margin guideline (visual only, very light)
draw.line((24, header_y1 + 12, 24, content_y1 - 24), fill=(250, 250, 250), width=2)

# End of structural/background drawing.
# NOTE: All icons/text/content will be pasted on top of these backgrounds at their detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/00_icon_04.png
try:
    _c0 = get_crop(0, 1440, 321)
    canvas.paste(_c0, (0, 2639), _c0)
except Exception:
    pass
layout["04"] = [0, 2639, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/01_icon_02.png
try:
    _c1 = get_crop(1, 1440, 367)
    canvas.paste(_c1, (0, 2272), _c1)
except Exception:
    pass
layout["02"] = [0, 2272, 1440, 2639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/02_icon_25.png
try:
    _c2 = get_crop(2, 1440, 367)
    canvas.paste(_c2, (0, 1171), _c2)
except Exception:
    pass
layout["25"] = [0, 1171, 1440, 1538]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/03_icon_28.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 1538), _c3)
except Exception:
    pass
layout["28"] = [0, 1538, 1440, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/04_icon_30.png
try:
    _c4 = get_crop(4, 1440, 367)
    canvas.paste(_c4, (0, 1905), _c4)
except Exception:
    pass
layout["30"] = [0, 1905, 1440, 2272]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/05_icon_76ers_at_New_York_Knicks_Game_2.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 804), _c5)
except Exception:
    pass
layout["76ers_at_New_York_Knicks_"] = [0, 804, 1440, 1171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/06_icon_Track_this_performer.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1128, 84), _c6)
except Exception:
    pass
layout["Track_this_performer"] = [1128, 84, 1272, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/07_icon_22.png
try:
    _c7 = get_crop(7, 1440, 367)
    canvas.paste(_c7, (0, 804), _c7)
except Exception:
    pass
layout["22"] = [0, 804, 1440, 1171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 1440, 367)
    canvas.paste(_c8, (0, 1905), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1905, 1440, 2272]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/09_icon_Knicks_at_Philadelphia_76ers_Game_3.png
try:
    _c9 = get_crop(9, 1440, 367)
    canvas.paste(_c9, (0, 1171), _c9)
except Exception:
    pass
layout["Knicks_at_Philadelphia_76"] = [0, 1171, 1440, 1538]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/10_icon_TBD.png
try:
    _c10 = get_crop(10, 192, 228)
    canvas.paste(_c10, (66, 316), _c10)
except Exception:
    pass
layout["TBD"] = [66, 316, 258, 544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/11_icon_Share_this_performer.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1278, 84), _c11)
except Exception:
    pass
layout["Share_this_performer"] = [1278, 84, 1422, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/12_icon_Knicks_at_Philadelphia_76ers_Game_4.png
try:
    _c12 = get_crop(12, 1440, 367)
    canvas.paste(_c12, (0, 1538), _c12)
except Exception:
    pass
layout["Knicks_at_Philadelphia_76"] = [0, 1538, 1440, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/13_icon_6.57_Wy.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (18, 84), _c13)
except Exception:
    pass
layout["6.57_Wy"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 99, 70)
    canvas.paste(_c14, (1215, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1215, 1, 1314, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/15_icon_6.57_Wy.png
try:
    _c15 = get_crop(15, 62, 65)
    canvas.paste(_c15, (112, 3), _c15)
except Exception:
    pass
layout["6.57_Wy"] = [112, 3, 174, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/16_icon_6.57_Wy.png
try:
    _c16 = get_crop(16, 54, 64)
    canvas.paste(_c16, (181, 4), _c16)
except Exception:
    pass
layout["6.57_Wy"] = [181, 4, 235, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 67, 65)
    canvas.paste(_c17, (241, 4), _c17)
except Exception:
    pass
layout["icon_17"] = [241, 4, 308, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 52, 71)
    canvas.paste(_c18, (1149, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1149, 2, 1201, 73]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 51, 65)
    canvas.paste(_c19, (1320, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [1320, 4, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/20_icon_New_York_Knicks.png
try:
    _c20 = get_crop(20, 58, 65)
    canvas.paste(_c20, (314, 3), _c20)
except Exception:
    pass
layout["New_York_Knicks"] = [314, 3, 372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/21_icon_New_York_Knicks.png
try:
    _c21 = get_crop(21, 52, 64)
    canvas.paste(_c21, (381, 1), _c21)
except Exception:
    pass
layout["New_York_Knicks"] = [381, 1, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/22_icon_Knicks_at_Philadelphia_76ers_Game_6.png
try:
    _c22 = get_crop(22, 1440, 367)
    canvas.paste(_c22, (0, 2272), _c22)
except Exception:
    pass
layout["Knicks_at_Philadelphia_76"] = [0, 2272, 1440, 2639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/23_icon_Madison_Square_Garden.png
try:
    _c23 = get_crop(23, 1440, 367)
    canvas.paste(_c23, (0, 292), _c23)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 292, 1440, 659]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/24_icon_6.57_Wy.png
try:
    _c24 = get_crop(24, 102, 69)
    canvas.paste(_c24, (8, 1), _c24)
except Exception:
    pass
layout["6.57_Wy"] = [8, 1, 110, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/25_text_New_York_Knicks.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (18, 84), _c25)
except Exception:
    pass
layout["New_York_Knicks"] = [18, 84, 162, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/26_text_Eastern_Conference_First_Round_Philadelp.png
try:
    _c26 = get_crop(26, 1440, 321)
    canvas.paste(_c26, (0, 2639), _c26)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [0, 2639, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/27_text_76ers_at_New_York_Knicks_Game_7.png
try:
    _c27 = get_crop(27, 1440, 321)
    canvas.paste(_c27, (0, 2639), _c27)
except Exception:
    pass
layout["76ers_at_New_York_Knicks_"] = [0, 2639, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/28_text_Home_Ga.png
try:
    _c28 = get_crop(28, 253, 52)
    canvas.paste(_c28, (1129, 2747), _c28)
except Exception:
    pass
layout["Home_Ga_"] = [1129, 2747, 1382, 2799]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/29_text_Time_TBD.png
try:
    _c29 = get_crop(29, 213, 52)
    canvas.paste(_c29, (309, 2829), _c29)
except Exception:
    pass
layout["Time_TBD"] = [309, 2829, 522, 2881]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/30_text_New_York_NY.png
try:
    _c30 = get_crop(30, 1440, 321)
    canvas.paste(_c30, (0, 2639), _c30)
except Exception:
    pass
layout["New_York;_NY"] = [0, 2639, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_06_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-9/31_text_From_S482.png
try:
    _c31 = get_crop(31, 235, 45)
    canvas.paste(_c31, (316, 2915), _c31)
except Exception:
    pass
layout["From_S482"] = [316, 2915, 551, 2960]
