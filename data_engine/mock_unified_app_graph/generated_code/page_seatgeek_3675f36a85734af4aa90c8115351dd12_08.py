# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_08
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11.png
# step_index: 8/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 50)], fill="#f2f2f2")

# Large slanted hero image background (dark)
hero_poly = [(0, 50), (1440, 50), (1440, 420), (0, 520)]
draw.polygon(hero_poly, fill="#000000")

# Slanted white sheet overlapping the hero (creates the angled cut)
slant_white = [(0, 480), (1440, 380), (1440, 980), (0, 980)]
draw.polygon(slant_white, fill="#ffffff")

# Main content card area behind title/buttons (subtle rounded card)
content_card_bbox = (24, 720, 1416, 1680)
draw.rounded_rectangle(content_card_bbox, radius=14, fill="#ffffff", outline=None)

# Light shadow under the content card (soft thin bar)
draw.rectangle([(24, 1680), (1416, 1684)], fill="#efefef")

# Separator lines between sections
separators_y = [1060, 1190, 1666, 1930, 2190, 2360]
for y in separators_y:
    draw.line([(24, y), (1416, y)], fill="#e6e6e6", width=1)

# Performers section card (rounded)
performers_bbox = (24, 1880, 1416, 2160)
draw.rounded_rectangle(performers_bbox, radius=10, fill="#ffffff", outline="#e9e9e9")

# Thin top divider for performers card to match screenshot subtlety
draw.line([(24, 1880), (1416, 1880)], fill="#e6e6e6", width=1)

# Box office area background (slightly separated block)
box_office_bbox = (24, 2200, 1416, 2740)
draw.rectangle(box_office_bbox, fill="#ffffff", outline="#eaeaea")

# Divider above box office
draw.line([(24, 2200), (1416, 2200)], fill="#e6e6e6", width=1)

# Subtle bottom footer separation
draw.line([(0, 2920), (1440, 2920)], fill="#f3f3f3", width=2)

# Accent thin rule under hero/content transition to emphasize separation
draw.line([(24, 980), (1416, 980)], fill="#ececec", width=1)

# Soft drop shadow under the hero slant (thin gradient-like bars)
shadow_bars = [
    (0, 970, 1440, 972),
    (0, 972, 1440, 975),
    (0, 975, 1440, 977)
]
shadow_colors = ["#f0f0f0", "#f1f1f1", "#f3f3f3"]
for bbox, col in zip(shadow_bars, shadow_colors):
    draw.rectangle([bbox[0:2], bbox[2:4]], fill=col)

# Header/back area background overlay (behind back arrow and status icons)
# Keep it transparent-ish by using a very light overlay color block to match screenshot feel
draw.rectangle([(0, 50), (1440, 120)], fill="#000000", outline=None)
# Make it less dominant by overlaying a semi-opaque white bar (simulated by a lighter strip)
draw.rectangle([(0, 50), (1440, 120)], fill="#121212")

# Slight top-left corner curvature hint (no icons drawn)
draw.polygon([(0, 50), (48, 50), (48, 98), (0, 118)], fill="#121212")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/00_icon_Share.png
try:
    _c0 = get_crop(0, 312, 153)
    canvas.paste(_c0, (606, 978), _c0)
except Exception:
    pass
layout["Share"] = [606, 978, 918, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/01_icon_Untrack_event.png
try:
    _c1 = get_crop(1, 498, 153)
    canvas.paste(_c1, (60, 978), _c1)
except Exception:
    pass
layout["Untrack_event"] = [60, 978, 558, 1131]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/02_icon_8.11_Wy.png
try:
    _c2 = get_crop(2, 64, 67)
    canvas.paste(_c2, (110, 0), _c2)
except Exception:
    pass
layout["8.11_Wy"] = [110, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 65)
    canvas.paste(_c3, (241, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [241, 3, 305, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/04_icon_8.11_Wy.png
try:
    _c4 = get_crop(4, 53, 64)
    canvas.paste(_c4, (183, 2), _c4)
except Exception:
    pass
layout["8.11_Wy"] = [183, 2, 236, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 46, 66)
    canvas.paste(_c5, (1155, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [1155, 3, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 57, 65)
    canvas.paste(_c6, (313, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [313, 4, 370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 66)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 65)
    canvas.paste(_c8, (1215, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1215, 3, 1268, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 44, 66)
    canvas.paste(_c9, (1272, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1272, 2, 1316, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/10_icon_Performers.png
try:
    _c10 = get_crop(10, 1416, 179)
    canvas.paste(_c10, (12, 1992), _c10)
except Exception:
    pass
layout["Performers"] = [12, 1992, 1428, 2171]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/11_icon_View_tickets_at.png
try:
    _c11 = get_crop(11, 1440, 144)
    canvas.paste(_c11, (0, 2402), _c11)
except Exception:
    pass
layout["View_tickets_at"] = [0, 2402, 1440, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/12_icon_8.11_Wy.png
try:
    _c12 = get_crop(12, 109, 67)
    canvas.paste(_c12, (0, 0), _c12)
except Exception:
    pass
layout["8.11_Wy"] = [0, 0, 109, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/13_icon_8.11_Wy.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (24, 84), _c13)
except Exception:
    pass
layout["8.11_Wy"] = [24, 84, 168, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 66)
    canvas.paste(_c14, (382, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [382, 1, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/15_text_Mkgee.png
try:
    _c15 = get_crop(15, 220, 81)
    canvas.paste(_c15, (56, 785), _c15)
except Exception:
    pass
layout["Mkgee"] = [56, 785, 276, 866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/16_text_Location.png
try:
    _c16 = get_crop(16, 209, 52)
    canvas.paste(_c16, (56, 1263), _c16)
except Exception:
    pass
layout["Location"] = [56, 1263, 265, 1315]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/17_text_The_Fonda_Theatre.png
try:
    _c17 = get_crop(17, 411, 49)
    canvas.paste(_c17, (53, 1381), _c17)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [53, 1381, 464, 1430]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/18_text_Los_Angeles_CA_90028.png
try:
    _c18 = get_crop(18, 471, 57)
    canvas.paste(_c18, (53, 1452), _c18)
except Exception:
    pass
layout["Los_Angeles,_CA_90028"] = [53, 1452, 524, 1509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/19_text_Get_directions.png
try:
    _c19 = get_crop(19, 1440, 113)
    canvas.paste(_c19, (0, 1553), _c19)
except Exception:
    pass
layout["Get_directions"] = [0, 1553, 1440, 1666]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/20_text_More_events_at_The_Fonda_Theatre.png
try:
    _c20 = get_crop(20, 1440, 113)
    canvas.paste(_c20, (0, 1666), _c20)
except Exception:
    pass
layout["More_events_at_The_Fonda_"] = [0, 1666, 1440, 1779]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/21_text_Performers.png
try:
    _c21 = get_crop(21, 256, 54)
    canvas.paste(_c21, (55, 1893), _c21)
except Exception:
    pass
layout["Performers"] = [55, 1893, 311, 1947]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/22_text_Mkgee.png
try:
    _c22 = get_crop(22, 170, 63)
    canvas.paste(_c22, (245, 2026), _c22)
except Exception:
    pass
layout["Mkgee"] = [245, 2026, 415, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/23_text_Box_office.png
try:
    _c23 = get_crop(23, 234, 54)
    canvas.paste(_c23, (54, 2302), _c23)
except Exception:
    pass
layout["Box_office"] = [54, 2302, 288, 2356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_08_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-11/24_clickable_Location.png
try:
    _c24 = get_crop(24, 1440, 1415)
    canvas.paste(_c24, (0, 1191), _c24)
except Exception:
    pass
layout["Location"] = [0, 1191, 1440, 2606]
