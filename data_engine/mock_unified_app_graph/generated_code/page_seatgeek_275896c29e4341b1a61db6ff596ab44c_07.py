# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_07
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10.png
# step_index: 7/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for SeatGeek-like page
# Uses provided variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg = (241, 244, 245)          # overall app background
status_bg = (226, 229, 231)   # status bar background
header_bg = (250, 251, 252)   # header pill background
header_border = (220, 224, 226)
modal_shadow = (230, 232, 233)
modal_bg = (255, 255, 255)
muted_divider = (236, 238, 239)
handle_color = (214, 217, 219)

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bg)

# Subtle bottom divider under status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=(214, 217, 219), width=1)

# Header/toolbar pill (rounded)
pill_x0, pill_x1 = 60, W - 60
pill_y0, pill_y1 = 90, 210
pill_radius = 64
draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)],
                       radius=pill_radius, fill=header_bg,
                       outline=header_border, width=2)

# Subtle divider inside the header pill on the right (visual separator for info icon area)
sep_x = pill_x1 - 120
draw.line([(sep_x, pill_y0 + 16), (sep_x, pill_y1 - 16)], fill=(235, 237, 238), width=2)

# Light shadow under the header pill
draw.rectangle([(pill_x0, pill_y1 + 2), (pill_x1, pill_y1 + 8)], fill=(238, 240, 241))

# Chips background area (light subtle band behind filter chips, do not draw chips themselves)
chips_band_y0 = pill_y1 + 16
chips_band_y1 = chips_band_y0 + 120
draw.rectangle([(0, chips_band_y0), (W, chips_band_y1)], fill=bg)

# Large modal sheet: draw subtle shadow then the rounded white modal
modal_x0, modal_x1 = 40, W - 40
modal_y0, modal_y1 = 320, H - 40
modal_radius = 40

# Shadow (slightly bigger, offset)
shadow_offset = 10
draw.rounded_rectangle([(modal_x0 + shadow_offset, modal_y0 + shadow_offset),
                        (modal_x1 + shadow_offset, modal_y1 + shadow_offset)],
                       radius=modal_radius, fill=modal_shadow)

# Modal background (rounded white sheet)
draw.rounded_rectangle([(modal_x0, modal_y0), (modal_x1, modal_y1)],
                       radius=modal_radius, fill=modal_bg)

# Modal top handle (small rounded pill)
handle_w, handle_h = 160, 10
handle_x0 = (W - handle_w) // 2
handle_y0 = modal_y0 + 18
draw.rounded_rectangle([(handle_x0, handle_y0), (handle_x0 + handle_w, handle_y0 + handle_h)],
                       radius=6, fill=handle_color)

# Thin divider under modal handle/header area
divider_y = modal_y0 + 72
draw.line([(modal_x0 + 24, divider_y), (modal_x1 - 24, divider_y)], fill=muted_divider, width=1)

# Top section card background inside modal (group background behind the title area)
card_x0 = modal_x0 + 20
card_x1 = modal_x1 - 20
card_y0 = modal_y0 + 36
card_y1 = card_y0 + 120
card_radius = 20
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=card_radius, fill=(255, 255, 255), outline=(242, 244, 245), width=1)

# Subtle separator line below the top section card
sep_y = card_y1 + 18
draw.line([(modal_x0 + 12, sep_y), (modal_x1 - 12, sep_y)], fill=muted_divider, width=1)

# Secondary section band (e.g., category container) - do not draw individual items
section_band_y0 = sep_y + 12
section_band_y1 = section_band_y0 + 120
draw.rectangle([(modal_x0 + 12, section_band_y0), (modal_x1 - 12, section_band_y1)], fill=modal_bg, outline=(245,246,247))

# A faint vertical divider at the far right inside modal to imply action area (non-icon)
v_div_x = modal_x1 - 88
draw.line([(v_div_x, modal_y0 + 12), (v_div_x, modal_y1 - 12)], fill=(245, 246, 247), width=1)

# Bottom area subtle fade band (visual end of content)
fade_band_h = 120
draw.rectangle([(modal_x0 + 6, modal_y1 - fade_band_h), (modal_x1 - 6, modal_y1)], fill=(250, 250, 250))

# Final subtle border around modal top edge to separate from background
draw.line([(modal_x0 + 6, modal_y0 + 1), (modal_x1 - 6, modal_y0 + 1)], fill=(242, 244, 245), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/00_icon_Quantity.png
try:
    _c0 = get_crop(0, 1320, 157)
    canvas.paste(_c0, (60, 693), _c0)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/01_icon_tickets.png
try:
    _c1 = get_crop(1, 1320, 157)
    canvas.paste(_c1, (60, 1513), _c1)
except Exception:
    pass
layout["tickets"] = [60, 1513, 1380, 1670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 1320, 157)
    canvas.paste(_c2, (60, 1308), _c2)
except Exception:
    pass
layout["3_tickets"] = [60, 1308, 1380, 1465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/03_icon_5_tickets.png
try:
    _c3 = get_crop(3, 1320, 157)
    canvas.paste(_c3, (60, 1718), _c3)
except Exception:
    pass
layout["5_tickets"] = [60, 1718, 1380, 1875]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/04_icon_9_tickets.png
try:
    _c4 = get_crop(4, 1320, 157)
    canvas.paste(_c4, (60, 2538), _c4)
except Exception:
    pass
layout["9_tickets"] = [60, 2538, 1380, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/05_icon_8_tickets.png
try:
    _c5 = get_crop(5, 1320, 157)
    canvas.paste(_c5, (60, 2333), _c5)
except Exception:
    pass
layout["8_tickets"] = [60, 2333, 1380, 2490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/06_icon_2_tickets.png
try:
    _c6 = get_crop(6, 1320, 157)
    canvas.paste(_c6, (60, 1103), _c6)
except Exception:
    pass
layout["2_tickets"] = [60, 1103, 1380, 1260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/07_icon_7_tickets.png
try:
    _c7 = get_crop(7, 1320, 157)
    canvas.paste(_c7, (60, 2128), _c7)
except Exception:
    pass
layout["7_tickets"] = [60, 2128, 1380, 2285]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/08_icon_6_tickets.png
try:
    _c8 = get_crop(8, 1320, 157)
    canvas.paste(_c8, (60, 1923), _c8)
except Exception:
    pass
layout["6_tickets"] = [60, 1923, 1380, 2080]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/09_icon_quantity.png
try:
    _c9 = get_crop(9, 1320, 157)
    canvas.paste(_c9, (60, 898), _c9)
except Exception:
    pass
layout["quantity"] = [60, 898, 1380, 1055]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/10_icon_Include_fees.png
try:
    _c10 = get_crop(10, 344, 126)
    canvas.paste(_c10, (535, 309), _c10)
except Exception:
    pass
layout["Include_fees"] = [535, 309, 879, 435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/11_icon_Home_plate.png
try:
    _c11 = get_crop(11, 334, 127)
    canvas.paste(_c11, (908, 308), _c11)
except Exception:
    pass
layout["Home_plate"] = [908, 308, 1242, 435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/12_icon_Quantity.png
try:
    _c12 = get_crop(12, 279, 130)
    canvas.paste(_c12, (231, 308), _c12)
except Exception:
    pass
layout["Quantity"] = [231, 308, 510, 438]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/13_icon_10_tickets.png
try:
    _c13 = get_crop(13, 1320, 157)
    canvas.paste(_c13, (60, 2743), _c13)
except Exception:
    pass
layout["10+_tickets"] = [60, 2743, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/14_icon_Ist_ba.png
try:
    _c14 = get_crop(14, 103, 108)
    canvas.paste(_c14, (1257, 147), _c14)
except Exception:
    pass
layout["Ist_ba"] = [1257, 147, 1360, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 66, 67)
    canvas.paste(_c15, (241, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [241, 1, 307, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/16_icon_Mariners_at_Angels.png
try:
    _c16 = get_crop(16, 64, 67)
    canvas.paste(_c16, (310, 1), _c16)
except Exception:
    pass
layout["Mariners_at_Angels"] = [310, 1, 374, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 66)
    canvas.paste(_c17, (1152, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1152, 1, 1203, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 54, 59)
    canvas.paste(_c18, (1320, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1320, 2, 1374, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 106, 65)
    canvas.paste(_c19, (1211, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1211, 0, 1317, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/20_icon_7.50_my.png
try:
    _c20 = get_crop(20, 67, 65)
    canvas.paste(_c20, (111, 0), _c20)
except Exception:
    pass
layout["7.50_my"] = [111, 0, 178, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/21_icon_7.50_my.png
try:
    _c21 = get_crop(21, 55, 64)
    canvas.paste(_c21, (181, 1), _c21)
except Exception:
    pass
layout["7.50_my"] = [181, 1, 236, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/22_icon_Tit.png
try:
    _c22 = get_crop(22, 167, 128)
    canvas.paste(_c22, (40, 307), _c22)
except Exception:
    pass
layout["Tit"] = [40, 307, 207, 435]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/23_icon_Ist_ba.png
try:
    _c23 = get_crop(23, 171, 137)
    canvas.paste(_c23, (1269, 308), _c23)
except Exception:
    pass
layout["Ist_ba"] = [1269, 308, 1440, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/24_icon_Mariners_at_Angels.png
try:
    _c24 = get_crop(24, 49, 67)
    canvas.paste(_c24, (383, 1), _c24)
except Exception:
    pass
layout["Mariners_at_Angels"] = [383, 1, 432, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/25_icon_Quantity.png
try:
    _c25 = get_crop(25, 1320, 157)
    canvas.paste(_c25, (60, 693), _c25)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/26_text_Mariners_at_Angels.png
try:
    _c26 = get_crop(26, 381, 56)
    canvas.paste(_c26, (201, 143), _c26)
except Exception:
    pass
layout["Mariners_at_Angels"] = [201, 143, 582, 199]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_07_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-10/27_text_Thu_Jul_11_6.38_PM.png
try:
    _c27 = get_crop(27, 355, 43)
    canvas.paste(_c27, (200, 207), _c27)
except Exception:
    pass
layout["Thu,_Jul_11,_6.38_PM"] = [200, 207, 555, 250]
