# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_08
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11.png
# step_index: 8/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
bg_color = (250, 250, 250)  # soft off-white
draw.rectangle([0, 0, 1440, 2960], fill=bg_color)

# Status bar (top)
status_h = 72
status_color = (241, 241, 243)
draw.rectangle([0, 0, 1440, status_h], fill=status_color)

# Hero/banner area with diagonal bottom edge
hero_top = status_h
hero_h = 560
hero_bottom = hero_top + hero_h
hero_rect_color = (18, 57, 115)   # deep blue base
hero_accent = (34, 110, 200)      # lighter accent for top band

# full hero base
draw.rectangle([0, hero_top, 1440, hero_bottom], fill=hero_rect_color)
# accent band near top
draw.rectangle([0, hero_top, 1440, hero_top + 32], fill=hero_accent)

# diagonal cut polygon for hero bottom to mimic screenshot angle
poly = [(0, hero_top),
        (1440, hero_top),
        (1440, hero_bottom - 80),
        (0, hero_bottom + 40)]
draw.polygon(poly, fill=hero_rect_color)

# White content card overlapping hero (holds title/date/buttons)
card_x0, card_x1 = 40, 1400
card_y0 = hero_bottom - 80
card_y1 = card_y0 + 660
card_radius = 16
card_fill = (255, 255, 255)
card_border = (230, 230, 230)
draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1],
                       radius=card_radius, fill=card_fill, outline=card_border, width=1)

# subtle shadow under the card (thin darker strip)
shadow_y = card_y1 + 2
draw.rectangle([card_x0 + 8, shadow_y, card_x1 - 8, shadow_y + 6], fill=(245, 245, 245))

# Thin divider under the title/buttons area (inside card)
divider_y = card_y1 - 120
draw.line([card_x0 + 8, divider_y, card_x1 - 8, divider_y], fill=(238, 238, 238), width=1)

# "Location" section background (keeps page organized)
loc_y0 = card_y1 + 40
loc_y1 = loc_y0 + 660
draw.rectangle([card_x0, loc_y0, card_x1, loc_y1], fill=card_fill, outline=None)

# Divider lines between location items (subtle)
sep_color = (241, 241, 241)
draw.line([card_x0 + 8, loc_y0 + 160, card_x1 - 8, loc_y0 + 160], fill=sep_color, width=1)
draw.line([card_x0 + 8, loc_y0 + 280, card_x1 - 8, loc_y0 + 280], fill=sep_color, width=1)

# Another section separator below location
section_sep_y = loc_y1 + 12
draw.line([card_x0 + 8, section_sep_y, card_x1 - 8, section_sep_y], fill=(230, 230, 230), width=1)

# Performers section card (slightly shaded background)
perf_y0 = section_sep_y + 24
perf_y1 = 2880
perf_fill = (250, 250, 251)
draw.rounded_rectangle([card_x0, perf_y0, card_x1, perf_y1],
                       radius=12, fill=perf_fill, outline=(235, 235, 235), width=1)

# Draw separators for performer rows (approximate row positions)
row_start_x = card_x0 + 24
row_end_x = card_x1 - 24
row_y = perf_y0 + 100
for i in range(5):
    draw.line([row_start_x, row_y, row_end_x, row_y], fill=(242, 242, 242), width=1)
    row_y += 160

# Bottom safe area line
draw.line([0, 2956, 1440, 2956], fill=(235, 235, 235), width=6)

# Decorative thin left/right gutters to frame content
gutter_color = (248, 248, 249)
draw.rectangle([0, card_y0, 40, perf_y1], fill=gutter_color)
draw.rectangle([1400, card_y0, 1440, perf_y1], fill=gutter_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/00_icon_Track_event.png
try:
    _c0 = get_crop(0, 444, 153)
    canvas.paste(_c0, (60, 1146), _c0)
except Exception:
    pass
layout["Track_event"] = [60, 1146, 504, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/01_icon_Share.png
try:
    _c1 = get_crop(1, 312, 153)
    canvas.paste(_c1, (552, 1146), _c1)
except Exception:
    pass
layout["Share"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/02_icon_Eastern_Conference_First_Round.png
try:
    _c2 = get_crop(2, 312, 153)
    canvas.paste(_c2, (552, 1146), _c2)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/03_icon_6.58_my.png
try:
    _c3 = get_crop(3, 61, 69)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["6.58_my"] = [112, 1, 173, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/04_icon_6.58_my.png
try:
    _c4 = get_crop(4, 51, 66)
    canvas.paste(_c4, (183, 3), _c4)
except Exception:
    pass
layout["6.58_my"] = [183, 3, 234, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 67)
    canvas.paste(_c5, (242, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [242, 3, 305, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/06_icon_Philadelphia_76ers.png
try:
    _c6 = get_crop(6, 1416, 179)
    canvas.paste(_c6, (12, 2160), _c6)
except Exception:
    pass
layout["Philadelphia_76ers"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 69)
    canvas.paste(_c7, (1154, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 2, 1202, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/08_icon_24_events.png
try:
    _c8 = get_crop(8, 1416, 179)
    canvas.paste(_c8, (12, 2697), _c8)
except Exception:
    pass
layout["24_events"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 69)
    canvas.paste(_c9, (1319, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1319, 1, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/10_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c10 = get_crop(10, 1416, 179)
    canvas.paste(_c10, (12, 2518), _c10)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/11_icon_215_events.png
try:
    _c11 = get_crop(11, 1416, 179)
    canvas.paste(_c11, (12, 2518), _c11)
except Exception:
    pass
layout["215_events"] = [12, 2518, 1428, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/12_icon_Performers.png
try:
    _c12 = get_crop(12, 1416, 179)
    canvas.paste(_c12, (12, 2160), _c12)
except Exception:
    pass
layout["Performers"] = [12, 2160, 1428, 2339]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 57, 68)
    canvas.paste(_c13, (314, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 2, 371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/14_icon_New_York_Knicks.png
try:
    _c14 = get_crop(14, 1416, 179)
    canvas.paste(_c14, (12, 2339), _c14)
except Exception:
    pass
layout["New_York_Knicks"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/15_icon_18_events.png
try:
    _c15 = get_crop(15, 1416, 179)
    canvas.paste(_c15, (12, 2339), _c15)
except Exception:
    pass
layout["18_events"] = [12, 2339, 1428, 2518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 61, 67)
    canvas.paste(_c16, (1212, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1212, 3, 1273, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/17_icon_6.58_my.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (24, 84), _c17)
except Exception:
    pass
layout["6.58_my"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/18_icon_New_York_Knicks_at_Philadelphia.png
try:
    _c18 = get_crop(18, 444, 153)
    canvas.paste(_c18, (60, 1146), _c18)
except Exception:
    pass
layout["New_York_Knicks_at_Philad"] = [60, 1146, 504, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 47, 66)
    canvas.paste(_c19, (1270, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [1270, 3, 1317, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/21_icon_NBA_Eastern_Conference_First_Round.png
try:
    _c21 = get_crop(21, 1416, 179)
    canvas.paste(_c21, (12, 2697), _c21)
except Exception:
    pass
layout["NBA_Eastern_Conference_Fi"] = [12, 2697, 1428, 2876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/22_icon_Eastern_Conference_First_Round.png
try:
    _c22 = get_crop(22, 312, 153)
    canvas.paste(_c22, (552, 1146), _c22)
except Exception:
    pass
layout["Eastern_Conference_First_"] = [552, 1146, 864, 1299]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/23_icon_New_York_Knicks.png
try:
    _c23 = get_crop(23, 359, 56)
    canvas.paste(_c23, (244, 2372), _c23)
except Exception:
    pass
layout["New_York_Knicks"] = [244, 2372, 603, 2428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/24_icon_NBA_Playoffs.png
try:
    _c24 = get_crop(24, 283, 55)
    canvas.paste(_c24, (244, 2552), _c24)
except Exception:
    pass
layout["NBA_Playoffs"] = [244, 2552, 527, 2607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/25_text_Location.png
try:
    _c25 = get_crop(25, 212, 52)
    canvas.paste(_c25, (53, 1432), _c25)
except Exception:
    pass
layout["Location"] = [53, 1432, 265, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/26_text_Wells_Fargo_Center.png
try:
    _c26 = get_crop(26, 410, 63)
    canvas.paste(_c26, (55, 1546), _c26)
except Exception:
    pass
layout["Wells_Fargo_Center"] = [55, 1546, 465, 1609]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/27_text_Philadelphia_PA_19148.png
try:
    _c27 = get_crop(27, 445, 57)
    canvas.paste(_c27, (53, 1621), _c27)
except Exception:
    pass
layout["Philadelphia,_PA_19148"] = [53, 1621, 498, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/28_text_Get_directions.png
try:
    _c28 = get_crop(28, 1440, 113)
    canvas.paste(_c28, (0, 1721), _c28)
except Exception:
    pass
layout["Get_directions"] = [0, 1721, 1440, 1834]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/29_text_More_events_at_Wells_Fargo_Center.png
try:
    _c29 = get_crop(29, 1440, 113)
    canvas.paste(_c29, (0, 1834), _c29)
except Exception:
    pass
layout["More_events_at_Wells_Farg"] = [0, 1834, 1440, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_08_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-11/30_text_Performers.png
try:
    _c30 = get_crop(30, 255, 52)
    canvas.paste(_c30, (56, 2061), _c30)
except Exception:
    pass
layout["Performers"] = [56, 2061, 311, 2113]
