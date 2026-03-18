# page_id: page_seatgeek_2c8f932b941840c18364dd035f1c8473_07
# screenshot: 2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10.png
# step_index: 7/8
# task: Open SeatGeek. Search "Beatles Love". Select the soonest upcoming event. Choose 2 tickets and continue. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#f3f5f7")

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#eceff1")

# Slight separator under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#d7d9db", width=1)

# Top header "pill" (background for event title bar)
header_x0, header_y0 = 40, 80
header_x1, header_y1 = 1400, 200
draw.rounded_rectangle([(header_x0, header_y0), (header_x1, header_y1)],
                       radius=60, fill="#ffffff", outline="#e0e0e0", width=1)

# Thin divider under the header pill
draw.line([(48, header_y1 + 4), (1392, header_y1 + 4)], fill="#e9e9ea", width=1)

# Tabs / filter pills row (background pills only)
tabs_y0, tabs_y1 = 252, 388
# positions chosen to align with typical layout; icons/text will be pasted on top later
tabs = [
    (60, tabs_y0, 260, tabs_y1),    # leftmost small pill
    (232, tabs_y0, 510, tabs_y1),   # Quantity (unselected)
    (536, tabs_y0, 878, tabs_y1),   # Include fees (selected)
    (908, tabs_y0, 1220, tabs_y1),  # Best seats
    (1220, tabs_y0, 1380, tabs_y1)  # Low pri / extra
]
# Draw neutral pills
for i, (x0, y0, x1, y1) in enumerate(tabs):
    if i == 2:
        # selected pill darker background
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=64, fill="#0b0b0b")
    else:
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=64, fill="#f6f7f8", outline="#e6e6e6", width=1)

# Shadow line under tabs row
draw.line([(40, tabs_y1 + 6), (1400, tabs_y1 + 6)], fill="#ececec", width=1)

# Main modal / sheet with rounded top corners (white)
modal_x0, modal_y0 = 30, 340
modal_x1, modal_y1 = 1410, 2920
draw.rounded_rectangle([(modal_x0, modal_y0), (modal_x1, modal_y1)],
                       radius=40, fill="#ffffff", outline="#e8e8e8", width=1)

# subtle inner divider under modal header area
draw.line([(modal_x0 + 20, modal_y0 + 80), (modal_x1 - 20, modal_y0 + 80)], fill="#f1f1f1", width=1)

# Draw list item card backgrounds (rounded rectangles) for quantity options
list_left = 60
list_right = 1380
item_height = 157
item_radius = 20
# Starting Y approximated to match typical layout under modal header
start_y = modal_y0 + 140
gap = 30  # vertical gap between items

# We'll draw a sequence of rounded cards down the modal. Icons/text will be pasted later.
for i in range(11):
    y0 = start_y + i * (item_height + gap)
    y1 = y0 + item_height
    # Slight variation: the top-most option has a more pronounced border (selection frame style)
    if i == 0:
        # selected/outlined card (no text)
        draw.rounded_rectangle([(list_left, y0), (list_right, y1)],
                               radius=item_radius, fill="#ffffff", outline="#cfcfcf", width=4)
    else:
        draw.rounded_rectangle([(list_left, y0), (list_right, y1)],
                               radius=item_radius, fill="#ffffff", outline="#ececec", width=1)

# Separator thin line near the very top of the modal title area
draw.line([(modal_x0 + 20, modal_y0 + 40), (modal_x1 - 20, modal_y0 + 40)], fill="#f4f4f4", width=1)

# Slight bottom fade area to suggest scrollable content (light gradient effect via bands)
fade_top = modal_y1 - 220
for i, alpha in enumerate(range(0, 60, 6)):
    y = fade_top + i * 4
    shade = 245 - i  # small change in gray to imply soft fade
    draw.rectangle([(modal_x0 + 20, y), (modal_x1 - 20, y + 4)], fill=(shade, shade, shade))

# Left and right safe area thin borders for modal content
draw.line([(modal_x0 + 12, modal_y0 + 8), (modal_x0 + 12, modal_y1 - 8)], fill="#fafafa", width=1)
draw.line([(modal_x1 - 12, modal_y0 + 8), (modal_x1 - 12, modal_y1 - 8)], fill="#fafafa", width=1)

# Subtle outer shadow along modal top for depth (drawn as a thin darker arc)
shadow_y = modal_y0 - 8
draw.rectangle([(modal_x0 + 8, shadow_y), (modal_x1 - 8, modal_y0)], fill="#f0f0f0")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/00_icon_tickets.png
try:
    _c0 = get_crop(0, 1320, 157)
    canvas.paste(_c0, (60, 1513), _c0)
except Exception:
    pass
layout["tickets"] = [60, 1513, 1380, 1670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/01_icon_Quantity.png
try:
    _c1 = get_crop(1, 1320, 157)
    canvas.paste(_c1, (60, 693), _c1)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/02_icon_3_tickets.png
try:
    _c2 = get_crop(2, 1320, 157)
    canvas.paste(_c2, (60, 1308), _c2)
except Exception:
    pass
layout["3_tickets"] = [60, 1308, 1380, 1465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/03_icon_5_tickets.png
try:
    _c3 = get_crop(3, 1320, 157)
    canvas.paste(_c3, (60, 1718), _c3)
except Exception:
    pass
layout["5_tickets"] = [60, 1718, 1380, 1875]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/04_icon_9_tickets.png
try:
    _c4 = get_crop(4, 1320, 157)
    canvas.paste(_c4, (60, 2538), _c4)
except Exception:
    pass
layout["9_tickets"] = [60, 2538, 1380, 2695]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/05_icon_8_tickets.png
try:
    _c5 = get_crop(5, 1320, 157)
    canvas.paste(_c5, (60, 2333), _c5)
except Exception:
    pass
layout["8_tickets"] = [60, 2333, 1380, 2490]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/06_icon_2_tickets.png
try:
    _c6 = get_crop(6, 1320, 157)
    canvas.paste(_c6, (60, 1103), _c6)
except Exception:
    pass
layout["2_tickets"] = [60, 1103, 1380, 1260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/07_icon_7_tickets.png
try:
    _c7 = get_crop(7, 1320, 157)
    canvas.paste(_c7, (60, 2128), _c7)
except Exception:
    pass
layout["7_tickets"] = [60, 2128, 1380, 2285]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/08_icon_6_tickets.png
try:
    _c8 = get_crop(8, 1320, 157)
    canvas.paste(_c8, (60, 1923), _c8)
except Exception:
    pass
layout["6_tickets"] = [60, 1923, 1380, 2080]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/09_icon_quantity.png
try:
    _c9 = get_crop(9, 1320, 157)
    canvas.paste(_c9, (60, 898), _c9)
except Exception:
    pass
layout["quantity"] = [60, 898, 1380, 1055]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/10_icon_Include.png
try:
    _c10 = get_crop(10, 342, 128)
    canvas.paste(_c10, (536, 308), _c10)
except Exception:
    pass
layout["Include"] = [536, 308, 878, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/11_icon_Best_seats.png
try:
    _c11 = get_crop(11, 312, 128)
    canvas.paste(_c11, (908, 308), _c11)
except Exception:
    pass
layout["Best_seats"] = [908, 308, 1220, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/12_icon_Quantity.png
try:
    _c12 = get_crop(12, 278, 131)
    canvas.paste(_c12, (232, 308), _c12)
except Exception:
    pass
layout["Quantity"] = [232, 308, 510, 439]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/13_icon_10_tickets.png
try:
    _c13 = get_crop(13, 1320, 157)
    canvas.paste(_c13, (60, 2743), _c13)
except Exception:
    pass
layout["10+_tickets"] = [60, 2743, 1380, 2900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/14_icon_6.png
try:
    _c14 = get_crop(14, 107, 111)
    canvas.paste(_c14, (1254, 144), _c14)
except Exception:
    pass
layout["6_"] = [1254, 144, 1361, 255]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/15_icon_Low_pri.png
try:
    _c15 = get_crop(15, 193, 135)
    canvas.paste(_c15, (1247, 308), _c15)
except Exception:
    pass
layout["Low_pri"] = [1247, 308, 1440, 443]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 55, 67)
    canvas.paste(_c16, (1151, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1151, 1, 1206, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 68, 66)
    canvas.paste(_c17, (241, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [241, 1, 309, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 60)
    canvas.paste(_c18, (1319, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 2, 1375, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/19_icon_5.08_my.png
try:
    _c19 = get_crop(19, 56, 65)
    canvas.paste(_c19, (180, 1), _c19)
except Exception:
    pass
layout["5.08_my"] = [180, 1, 236, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 61, 67)
    canvas.paste(_c20, (313, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 1, 374, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/21_icon_Tit.png
try:
    _c21 = get_crop(21, 168, 136)
    canvas.paste(_c21, (39, 307), _c21)
except Exception:
    pass
layout["Tit"] = [39, 307, 207, 443]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/22_icon_5.08_my.png
try:
    _c22 = get_crop(22, 69, 65)
    canvas.paste(_c22, (110, 1), _c22)
except Exception:
    pass
layout["5.08_my"] = [110, 1, 179, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/23_icon_Quantity.png
try:
    _c23 = get_crop(23, 1320, 157)
    canvas.paste(_c23, (60, 693), _c23)
except Exception:
    pass
layout["Quantity"] = [60, 693, 1380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/24_icon_6.png
try:
    _c24 = get_crop(24, 107, 64)
    canvas.paste(_c24, (1212, 1), _c24)
except Exception:
    pass
layout["6_"] = [1212, 1, 1319, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/25_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c25 = get_crop(25, 51, 66)
    canvas.paste(_c25, (382, 1), _c25)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [382, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/26_text_Cirque_du_Soleil_The_Beatles.png
try:
    _c26 = get_crop(26, 582, 60)
    canvas.paste(_c26, (195, 141), _c26)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [195, 141, 777, 201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/27_text_Love.png
try:
    _c27 = get_crop(27, 101, 41)
    canvas.paste(_c27, (806, 149), _c27)
except Exception:
    pass
layout["Love"] = [806, 149, 907, 190]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/28_text_Las_Vegas.png
try:
    _c28 = get_crop(28, 213, 60)
    canvas.paste(_c28, (938, 142), _c28)
except Exception:
    pass
layout["Las_Vegas"] = [938, 142, 1151, 202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c8f932b941840c18364dd035f1c8473/step_07_2024_4_22_17_5_2c8f932b941840c18364dd035f1c8473-10/29_text_Tue_Apr_23_7_PM.png
try:
    _c29 = get_crop(29, 330, 48)
    canvas.paste(_c29, (198, 205), _c29)
except Exception:
    pass
layout["Tue,_Apr_23,7_PM"] = [198, 205, 528, 253]
